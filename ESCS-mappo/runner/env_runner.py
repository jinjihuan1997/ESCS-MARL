# =====================================
# Filepath: runner/env_runner.py
# -*- coding: utf-8 -*-
# =====================================

import time
import numpy as np
import torch
from runner.base_runner import Runner


def _t2n(x: torch.Tensor):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    """Runner class to perform training, evaluation and data collection for EnvCore-like envs."""

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    # --------- helpers ---------
    def _build_share_obs(self, obs, envs):
        """
        构造 centralized critic 的输入：
        - 若 use_centralized_V=False 直接返回 obs
        - 若 use_obs_instead_of_state=True 使用 obs
        - 否则尽量从 env 获取全局状态（get_state/get_global_state/get_share_obs），失败则回退为拼接 obs
        返回形状：[n_rollout_threads, num_agents, state_dim 或 agents*obs_dim]
        """
        if not self.use_centralized_V:
            return obs

        if getattr(self.all_args, "use_obs_instead_of_state", False):
            return obs

        # 优先：从每个子环境拉取 centralized 向量
        try:
            if hasattr(envs, "envs") and isinstance(envs.envs, (list, tuple)) and len(
                    envs.envs) == self.n_rollout_threads:
                per_env_vecs = []
                for e in envs.envs:
                    vec = None
                    if hasattr(e, "get_share_obs") and callable(getattr(e, "get_share_obs")):
                        g = e.get_share_obs()
                        if isinstance(g, (list, tuple)) and len(g) == self.num_agents:
                            vec = np.asarray(g[0], dtype=np.float32).reshape(-1)
                        elif isinstance(g, np.ndarray) and g.ndim == 1:
                            vec = g.astype(np.float32, copy=False)
                    if vec is None and hasattr(e, "get_global_state") and callable(getattr(e, "get_global_state")):
                        s = e.get_global_state()
                        if isinstance(s, dict) and "vector" in s:
                            vec = np.asarray(s["vector"], dtype=np.float32).reshape(-1)
                        else:
                            vec = np.asarray(s, dtype=np.float32).reshape(-1)
                    if vec is None:
                        per_env_vecs = None
                        break
                    per_env_vecs.append(vec)
                if per_env_vecs is not None:
                    state = np.stack(per_env_vecs, axis=0)  # [n_env, state_dim]
                    share_obs = np.expand_dims(state, 1).repeat(self.num_agents, axis=1)
                    return share_obs
        except Exception:
            pass

        # 次优：直接在向量化封装上尝试
        state = None
        for attr in ["get_state", "get_global_state"]:
            if hasattr(envs, attr):
                getter = getattr(envs, attr)
                try:
                    s = getter() if callable(getter) else getter
                except Exception:
                    s = None
                if s is not None:
                    state = s
                    break
        if state is None and hasattr(envs, "state"):
            try:
                state = envs.state
            except Exception:
                state = None

        if state is not None:
            state = np.asarray(state)
            if state.ndim == 1:
                state = state.reshape(self.n_rollout_threads, -1)
            else:
                state = state.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(state, 1).repeat(self.num_agents, axis=1)
            return share_obs

        # 回退：拼接各 agent 观测
        share_obs = obs.reshape(self.n_rollout_threads, -1)  # [n_env, agents*obs_dim]
        share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        return share_obs

    def _query_available_actions(self, envs):
        """
        从 env 查询可行动作掩码；若不可用返回 None。
        支持：get_available_actions / get_avail_actions / available_actions / action_masks / get_action_mask
        """
        cand = ["get_available_actions", "get_avail_actions", "available_actions", "action_masks", "get_action_mask"]
        avail = None
        for name in cand:
            if hasattr(envs, name):
                obj = getattr(envs, name)
                try:
                    avail = obj() if callable(obj) else obj
                except Exception:
                    avail = None
                if avail is not None:
                    break
        return avail

    def _flatten_available_actions(self, available_actions, n_heads=None):
        """
        将 env 返回的掩码展平成策略可接收的 [B, sum(nvec)]，B=n_rollout_threads*num_agents。
        支持：
          - ndarray/tensor: [n_env, n_agent, sum(nvec)]
          - list/tuple（按头）：len=H；每项 [n_env, n_agent, nvec[h]]
        """
        if available_actions is None:
            return None

        B = self.n_rollout_threads * self.num_agents

        # torch tensor
        if isinstance(available_actions, torch.Tensor):
            arr = available_actions
            if arr.dim() == 3 and arr.size(0) == self.n_rollout_threads and arr.size(1) == self.num_agents:
                arr = arr.reshape(B, -1)
            return arr.to(self.device, dtype=torch.float32)

        # numpy ndarray
        if isinstance(available_actions, np.ndarray):
            if (
                    available_actions.ndim == 3
                    and available_actions.shape[0] == self.n_rollout_threads
                    and available_actions.shape[1] == self.num_agents
            ):
                arr = available_actions.reshape(B, -1)
                return torch.from_numpy(arr).to(self.device, dtype=torch.float32)
            return None

        # list / tuple: 视为“按头”的列表
        if isinstance(available_actions, (list, tuple)) and len(available_actions) > 0:
            head_mats = []
            for head_mask in available_actions:
                if head_mask is None:
                    return None
                hm = torch.as_tensor(head_mask)
                if hm.dim() == 3 and hm.size(0) == self.n_rollout_threads and hm.size(1) == self.num_agents:
                    hm = hm.reshape(B, -1)
                else:
                    return None
                head_mats.append(hm)
            cat = torch.cat(head_mats, dim=-1).to(self.device, dtype=torch.float32)
            return cat

        return None

    def _split_batch(self, x):
        """将 [B, ...] 切回 [n_rollout_threads, num_agents, ...]；x 为 numpy 或 torch 均可。"""
        if isinstance(x, torch.Tensor):
            x = _t2n(x)
        chunks = np.split(x, self.n_rollout_threads, axis=0)  # list of [num_agents, ...]
        return np.stack(chunks, axis=0)  # [n_env, num_agents, ...]

    def _get_nvec(self):
        """读取 MultiDiscrete 的各头维度。非 MultiDiscrete 返回 None。"""
        space = self.envs.action_space[0]
        if space.__class__.__name__ == "MultiDiscrete":
            return [int(x) for x in list(space.nvec)]
        return None

    def _log_actions_with_mask(self, slot_idx, actions, avail_for_buffer):
        """
        打印每个时间槽、每个 env/agent 的动作，以及是否满足 action mask。
        actions:          [n_env, n_agent, H]（MultiDiscrete）
        avail_for_buffer: [n_env, n_agent, sum(nvec)] 或 None
        """
        nvec = self._get_nvec()
        if nvec is None:
            print("[act] (logging) 当前动作空间非 MultiDiscrete，跳过日志。")
            return

    def _normalize_dones(self, dones):
        """
        把 dones 统一成 [n_env, n_agent, 1] 的 float32（1.0=done, 0.0=alive）。
        兼容： [n_env, n_agent], [n_env, n_agent, 1], [n_env, n_agent, k], [n_env] 等。
        """
        d = np.asarray(dones, dtype=np.float32)
        N, A = self.n_rollout_threads, self.num_agents

        if d.ndim == 3:
            if d.shape[0] == N and d.shape[1] == A:
                if d.shape[2] == 1:
                    return d
                else:
                    return d[..., :1]
            # 其他形态强制重排
            d = d.reshape(N, A, -1)[..., :1]
            return d

        if d.ndim == 2:
            if d.shape[0] == N and d.shape[1] == A:
                return d[..., None]
            d = d.reshape(N, A)[..., None]
            return d

        if d.ndim == 1:
            if d.size == N * A:
                return d.reshape(N, A, 1)
            return np.tile(d.reshape(N, 1, 1), (1, A, 1))

        # fallback：直接 reshape
        return d.reshape(N, A, -1)[..., :1]

    # --------- main loops ---------
    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                    avail_flat_for_buffer,
                ) = self.collect(step)

                # ====== 新增：打印/校验 每槽 动作 与 掩码 ======
                # 以“全局槽号”标识（按 episode*len + step 来估计）
                global_slot = episode * self.episode_length + step + 1
                if getattr(self.all_args, "log_action_mask", True):
                    self._log_actions_with_mask(global_slot, actions, avail_flat_for_buffer)
                # ============================================

                # Step env
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    avail_flat_for_buffer,
                )
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        getattr(self.all_args, "scenario_name", self.env_name),
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / max(1e-6, (end - start))),
                    )
                )
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()  # [n_env, n_agent, obs_dim]
        # build share obs
        share_obs = self._build_share_obs(obs, self.envs)

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()

        # centralized obs for critic
        share_obs_flat = np.concatenate(self.buffer.share_obs[step])  # [n_env*num_agents, ...]
        obs_flat = np.concatenate(self.buffer.obs[step])
        rnn_states_flat = np.concatenate(self.buffer.rnn_states[step])
        rnn_states_critic_flat = np.concatenate(self.buffer.rnn_states_critic[step])
        masks_flat = np.concatenate(self.buffer.masks[step])

        # query and flatten available actions (optional)
        avail = self._query_available_actions(self.envs)
        avail_flat = self._flatten_available_actions(avail)

        # policy forward
        value, action, action_log_prob, rnn_states_new, rnn_states_critic_new = self.trainer.policy.get_actions(
            share_obs_flat, obs_flat, rnn_states_flat, rnn_states_critic_flat, masks_flat,
            available_actions=avail_flat, deterministic=False
        )

        # split back to [n_env, n_agent, ...]
        values = self._split_batch(value)  # [n_env, n_agent, 1]
        actions = self._split_batch(action)  # [n_env, n_agent, H] (MultiDiscrete) 或 [n_env, n_agent, 1] (Discrete)
        action_log_probs = self._split_batch(action_log_prob)  # [n_env, n_agent, 1]
        rnn_states = self._split_batch(rnn_states_new)  # [n_env, n_agent, rnn, hidden]
        rnn_states_critic = self._split_batch(rnn_states_critic_new)

        # actions for env:
        space = self.envs.action_space[0]
        if space.__class__.__name__ == "MultiDiscrete":
            actions_env = actions
        elif space.__class__.__name__ == "Discrete":
            actions_env = np.squeeze(np.eye(space.n)[actions], 2)
        else:
            actions_env = actions

        # 可行动作掩码 -> buffer 形状
        if avail_flat is not None:
            avail_np = _t2n(avail_flat) if isinstance(avail_flat, torch.Tensor) else np.asarray(avail_flat)
            avail_for_buffer = self._split_batch(avail_np)  # [n_env, n_agent, sum(nvec)]
        else:
            avail_for_buffer = None

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
            avail_for_buffer,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            available_actions,  # 可能为 None 或 [n_env, n_agent, sum(nvec)]
        ) = data

        # --- 规范 dones -> [N, A, 1] ---
        done_f = self._normalize_dones(dones)  # float32, 1=done, 0=alive
        keep = 1.0 - done_f  # [N, A, 1]
        keep_rnn = keep.reshape(self.n_rollout_threads, self.num_agents, 1, 1)

        # 以广播方式重置 RNN（稳定、无布尔索引）
        rnn_states *= keep_rnn
        rnn_states_critic *= keep_rnn

        masks = keep.astype(np.float32)  # [N, A, 1]
        active_masks = masks.copy()

        # centralized obs
        share_obs = self._build_share_obs(obs, self.envs)

        # 仅当 buffer 预置了 available_actions 存储时才写入，否则传 None
        if getattr(self.buffer, "available_actions", None) is None:
            available_actions = None

        # 写入 buffer
        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            active_masks=active_masks,
            available_actions=available_actions,
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32
        )
        eval_rnn_states_critic = np.zeros_like(eval_rnn_states)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for _ in range(self.episode_length):
            self.trainer.prep_rollout()

            # centralized obs（eval 阶段同样支持 centralized V）
            if self.use_centralized_V and not getattr(self.all_args, "use_obs_instead_of_state", False):
                eval_share_obs = self._build_share_obs(eval_obs, self.eval_envs)
                eval_share_obs_flat = np.concatenate(eval_share_obs)
            else:
                eval_share_obs_flat = np.concatenate(eval_obs)

            # 可行动作掩码（若有）
            avail = self._query_available_actions(self.eval_envs)
            avail_flat = self._flatten_available_actions(avail)

            eval_values, eval_action, eval_logp, eval_rnn_states_flat, eval_rnn_states_critic_flat = \
                self.trainer.policy.get_actions(
                    eval_share_obs_flat,
                    np.concatenate(eval_obs),
                    np.concatenate(eval_rnn_states),
                    np.concatenate(eval_rnn_states_critic),
                    np.concatenate(eval_masks),
                    available_actions=avail_flat,
                    deterministic=True,
                )

            eval_actions = self._split_batch(eval_action)
            eval_rnn_states = self._split_batch(eval_rnn_states_flat)
            eval_rnn_states_critic = self._split_batch(eval_rnn_states_critic_flat)

            space = self.eval_envs.action_space[0]
            if space.__class__.__name__ == "MultiDiscrete":
                eval_actions_env = eval_actions
            elif space.__class__.__name__ == "Discrete":
                eval_actions_env = np.squeeze(np.eye(space.n)[eval_actions], 2)
            else:
                eval_actions_env = eval_actions

            # Step eval env
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            # --- 规范 dones 并用广播清零 RNN ---
            done_f = self._normalize_dones(eval_dones)
            keep = 1.0 - done_f
            keep_rnn = keep.reshape(self.n_eval_rollout_threads, self.num_agents, 1, 1)
            eval_rnn_states *= keep_rnn
            eval_rnn_states_critic *= keep_rnn
            eval_masks = keep.astype(np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos["eval_average_episode_rewards"])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        all_frames = []
        for _ in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render("rgb_array")[0][0]
                all_frames.append(image)
            else:
                envs.render("human")

            rnn_states = np.zeros(
                (self.n_rollout_threads, self.num_agents, self.recurrent_N, self.actor_hidden_size),
                dtype=np.float32,
            )
            rnn_states_critic = np.zeros(
                (self.n_rollout_threads, self.num_agents, self.recurrent_N, self.critic_hidden_size),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []

            for _ in range(self.episode_length):
                self.trainer.prep_rollout()

                # 可行动作掩码（渲染也保持一致）
                avail = self._query_available_actions(envs)
                avail_flat = self._flatten_available_actions(avail)

                # centralized obs（渲染阶段）
                if self.use_centralized_V and not getattr(self.all_args, "use_obs_instead_of_state", False):
                    share_obs = self._build_share_obs(obs, envs)
                    share_obs_flat = np.concatenate(share_obs)
                else:
                    share_obs_flat = np.concatenate(obs)

                _, action_tensor, _, rnn_states_flat, rnn_states_critic_flat = \
                    self.trainer.policy.get_actions(
                        share_obs_flat,
                        np.concatenate(obs),
                        np.concatenate(rnn_states),
                        np.concatenate(rnn_states_critic),
                        np.concatenate(masks),
                        available_actions=avail_flat,
                        deterministic=False,
                    )
                actions = self._split_batch(action_tensor)
                rnn_states = self._split_batch(rnn_states_flat)
                rnn_states_critic = self._split_batch(rnn_states_critic_flat)

                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    actions_env = actions
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    actions_env = actions

                # Step env
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                # --- 规范 dones 并用广播清零 RNN ---
                done_f = self._normalize_dones(dones)
                keep = 1.0 - done_f
                keep_rnn = keep.reshape(self.n_rollout_threads, self.num_agents, 1, 1)
                rnn_states *= keep_rnn
                rnn_states_critic *= keep_rnn
                masks = keep.astype(np.float32)

                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                else:
                    envs.render("human")

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))
