# =====================================
# Filepath: runner/env_runner_2actor.py
# -*- coding: utf-8 -*-
# =====================================

import time
import numpy as np
import torch
from runner.base_runner_2actor import Runner


def _pack_actions_env(actions_list):
    """把不同 agent 的动作按 [n_env, list(per-agent action)] 方式打包给环境。"""
    assert isinstance(actions_list, (list, tuple)) and len(actions_list) > 0
    N = np.asarray(actions_list[0]).shape[0]
    A = len(actions_list)
    out = np.empty((N,), dtype=object)
    for n in range(N):
        per_env = [np.asarray(actions_list[i][n]) for i in range(A)]
        out[n] = per_env
    return out


def _t2n(x: torch.Tensor):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    """Runner for grouped (BS/SD) actors with per-group critics and a centralized critic input."""

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)
        self._policy_group_ready = False
        self._agent_groups = None  # 长度 = num_agents；元素 "BS"/"SD"

    # --------- helpers ---------
    def _build_share_obs(self, obs, envs):
        """
        构造 centralized critic 的输入：
        - 若 use_centralized_V=False 直接返回 obs
        - 若 use_obs_instead_of_state=True 使用 obs
        - 否则优先从每个子环境拿全局状态（get_share_obs/get_global_state），失败则回退拼接 obs
        返回 [n_env, num_agents, state_dim 或 agents*obs_dim]
        """
        if not self.use_centralized_V:
            return obs

        if getattr(self.all_args, "use_obs_instead_of_state", False):
            return obs

        # 优先每个子环境的 share obs / global state
        try:
            if hasattr(envs, "envs") and isinstance(envs.envs, (list, tuple)) and len(envs.envs) == self.n_rollout_threads:
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

        # 回退：直接拼接 obs
        share_obs = obs.reshape(self.n_rollout_threads, -1)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        return share_obs

    def _query_available_actions(self, envs):
        """尝试从 env 查询动作掩码；失败返回 None。"""
        cand = ["get_available_actions", "get_avail_actions", "available_actions", "action_masks", "get_action_mask"]
        for name in cand:
            if hasattr(envs, name):
                try:
                    out = getattr(envs, name)() if callable(getattr(envs, name)) else getattr(envs, name)
                except Exception:
                    out = None
                if out is not None:
                    return out
        return None

    def _split_batch(self, x):
        """把 [B, ...] 切回 [N, A, ...]；x 可为 numpy 或 torch。"""
        if isinstance(x, torch.Tensor):
            x = _t2n(x)
        chunks = np.split(x, self.n_rollout_threads, axis=0)
        return np.stack(chunks, axis=0)

    def _normalize_dones(self, dones):
        """
        把 dones 统一成 [n_env, n_agent, 1] 的 float32（1.0=done, 0.0=alive），
        自适应 n_env（训练/评估均可用）。
        """
        d = np.asarray(dones, dtype=np.float32)
        A = self.num_agents

        if d.ndim == 3:
            # [..., 1] 直接返回；否则仅取第一列
            return d[..., :1]

        if d.ndim == 2:
            # 若第二维就是 A，直接扩 1 维；否则强制 reshape(-1, A)
            if d.shape[1] == A:
                return d[..., None]
            return d.reshape(-1, A)[..., None]

        if d.ndim == 1:
            # 能整除就 (N, A, 1)，否则当成 (N, 1, 1) 广播成 (N, A, 1)
            if d.size % A == 0:
                return d.reshape(-1, A, 1)
            return np.tile(d.reshape(-1, 1, 1), (1, A, 1))

        # fallback：尝试按 A 对齐
        try:
            return d.reshape(-1, A, 1)
        except Exception:
            # 最后兜底：强行把尾维裁到 1
            return d.reshape(-1, A, -1)[..., :1]

    def _ensure_policy_groups(self):
        """构造 agent_id -> group 映射：0=BS，其余=SD。"""
        if self._policy_group_ready:
            return
        self._has_multi_policies = hasattr(self.trainer, "policies") and isinstance(self.trainer.policies, dict)
        if self._has_multi_policies:
            keys = set(self.trainer.policies.keys())
            assert keys == {"SD", "BS"}, f"trainer.policies 键应为 'SD' 和 'BS'，当前={keys}"
        self._agent_groups = ["BS"] + ["SD"] * (self.num_agents - 1)
        self._policy_group_ready = True

    # ====================== 主训练循环 ====================== #
    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            # lr decay
            if self.use_linear_lr_decay:
                if hasattr(self.trainer, "policies") and isinstance(self.trainer.policies, dict):
                    for _, p in self.trainer.policies.items():
                        if hasattr(p, "lr_decay"):
                            p.lr_decay(episode, episodes)
                elif hasattr(self.trainer, "policy"):
                    self.trainer.policy.lr_decay(episode, episodes)

            # === 每个并行环境的 shared-return 累加器（按时隙求和） ===
            ep_return_per_env = np.zeros(self.n_rollout_threads, dtype=np.float32)

            for step in range(self.episode_length):
                # 采样动作
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                    _,
                ) = self.collect(step)

                # 交互一步
                obs, rewards, dones, infos = self.envs.step(actions_env)

                # —— 累加“shared_reward”（取 BS 一列即可；若形状不标准做健壮处理）——
                r = np.asarray(rewards)
                if r.ndim == 3:  # [N, A, 1] 或 [N, A, something]
                    if r.shape[2] == 1:
                        r_env = r[:, 0, 0]
                    else:
                        r_env = r[:, 0, :].mean(axis=1)
                elif r.ndim == 2:  # [N, A]
                    r_env = r[:, 0]
                elif r.ndim == 1:  # [N]
                    r_env = r
                else:  # 兜底
                    r_env = r.reshape(self.n_rollout_threads, -1).mean(axis=1)
                ep_return_per_env += r_env.astype(np.float32)

                # 写入各自 buffer（内部会再次读取“下一步”的 SD 掩码）
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
                    None,
                )
                self.insert(data)

            # 计算回报并更新网络（分组）
            self.compute()
            train_infos = self.train()

            # —— 日志：平均 episode reward（按环境求平均）——
            avg_episode_rewards = float(ep_return_per_env.mean())
            train_infos["average_episode_rewards"] = avg_episode_rewards

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log
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
                # —— 按你的定义输出 —— #
                print("average episode rewards is {}".format(avg_episode_rewards))
                self.log_train(train_infos, total_num_steps)

            # eval（可选）
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        """重置环境，写入两个 buffer 的 t=0 条目，并保存 t=0 的 SD 动作掩码。"""
        obs = self.envs.reset()  # [N, A, O]

        share_obs = self._build_share_obs(obs, self.envs)  # [N, A, S]

        # 写入 t=0 观测
        self.buffers["BS"].share_obs[0] = share_obs[:, 0:1, :].copy()
        self.buffers["BS"].obs[0] = obs[:, 0:1, :].copy()

        if "SD" in self.buffers:
            self.buffers["SD"].share_obs[0] = share_obs[:, 1:, :].copy()
            self.buffers["SD"].obs[0] = obs[:, 1:, :].copy()

        # 读取当前时刻（t=0）的 SD 动作掩码并写入 buffer[0]
        avail_all = self._query_available_actions(self.envs)
        if isinstance(avail_all, np.ndarray) and avail_all.ndim == 3 and "SD" in self.buffers:
            sd_need = (
                int(np.sum(list(self.envs.action_space[1].nvec)))
                if self.envs.action_space[1].__class__.__name__ == "MultiDiscrete"
                else None
            )
            if sd_need is not None and self.buffers["SD"].available_actions is not None:
                if avail_all.shape[2] >= sd_need:
                    self.buffers["SD"].available_actions[0] = avail_all[:, 1:, :sd_need].astype(np.float32)

    # ------------------- 采样（支持多策略） ------------------- #
    @torch.no_grad()
    def collect(self, step):
        """
        两套 Actor（BS/SD）+ 各自 Critic（共享 centralized 输入）。
        """
        self._ensure_policy_groups()

        # ---- 1) 当前时刻 obs ----
        obs_bs = self.buffers["BS"].obs[step]  # [N, 1, O]
        obs_sd = self.buffers["SD"].obs[step] if "SD" in self.buffers else None  # [N, K, O] or None

        obs = np.concatenate([obs_bs, obs_sd], axis=1) if obs_sd is not None else obs_bs  # [N, A, O]

        # centralized obs/state 给 critic
        share = self._build_share_obs(obs, self.envs)  # [N, A, S]

        # ---- 2) 组合 rnn_states / masks（都用 t 时刻的）----
        N, A = self.n_rollout_threads, self.num_agents
        rnn = np.zeros((N, A, self.recurrent_N, self.actor_hidden_size), dtype=np.float32)
        rnnc = np.zeros((N, A, self.recurrent_N, self.critic_hidden_size), dtype=np.float32)
        masks = np.zeros((N, A, 1), dtype=np.float32)

        rnn[:, 0:1, :, :] = self.buffers["BS"].rnn_states[step]
        rnnc[:, 0:1, :, :] = self.buffers["BS"].rnn_states_critic[step]
        masks[:, 0:1, :] = self.buffers["BS"].masks[step]  # BS 的 mask 取自 BS buffer

        if "SD" in self.buffers:
            rnn[:, 1:, :, :] = self.buffers["SD"].rnn_states[step]
            rnnc[:, 1:, :, :] = self.buffers["SD"].rnn_states_critic[step]
            masks[:, 1:, :] = self.buffers["SD"].masks[step]

        # ---- 3) 当前时刻的动作掩码（仅 SD 会用），注意：训练阶段不改变 env 的动作语义 ----
        avail_all = self._query_available_actions(self.envs)
        avail_np = np.asarray(avail_all) if isinstance(avail_all, np.ndarray) else None

        values_list, actions_list, logp_list, rnn_list, rnnc_list = [], [], [], [], []

        # ---- 4) 逐 agent 前向（异构动作 OK）----
        for i in range(A):
            group = "BS" if i == 0 else "SD"
            pi = self.trainer.policies[group] if hasattr(self.trainer, "policies") else self.trainer.policy

            share_i = share[:, i, :]  # [N, S]
            obs_i = obs[:, i, :]      # [N, O]
            rnn_i = rnn[:, i, :, :]   # [N, r, h]
            rnnc_i = rnnc[:, i, :, :]
            mask_i = masks[:, i, :]   # [N, 1]

            # 掩码切片（按各自空间宽度裁）
            avail_i = None
            if isinstance(avail_np, np.ndarray) and avail_np.ndim == 3:
                space_i = self.envs.action_space[i]
                cls = space_i.__class__.__name__
                if cls == "MultiDiscrete":
                    need = int(np.sum(list(space_i.nvec)))
                    if avail_np.shape[2] >= need:
                        avail_i = avail_np[:, i, :need]
                elif cls == "Discrete":
                    need = int(space_i.n)
                    if avail_np.shape[2] >= need:
                        avail_i = avail_np[:, i, :need]

            v_i, a_i, lp_i, rnn_new_i, rnnc_new_i = pi.get_actions(
                share_i, obs_i, rnn_i, rnnc_i, mask_i, available_actions=avail_i, deterministic=False
            )

            values_list.append(_t2n(v_i))           # [N, 1]
            actions_list.append(_t2n(a_i))          # [N, H_i]
            logp_list.append(_t2n(lp_i))            # [N, 1] 或 [N, H_i]
            rnn_list.append(_t2n(rnn_new_i))        # [N, r, h]
            rnnc_list.append(_t2n(rnnc_new_i))      # [N, r, h]

        # ---- 5) 打包给训练与环境 ----
        values = np.stack(values_list, axis=1)               # [N, A, 1]
        rnn_states = np.stack(rnn_list, axis=1)              # [N, A, r, h]
        rnn_states_critic = np.stack(rnnc_list, axis=1)      # [N, A, r, h]

        # 训练用的 actions / logp：用 ragged object，交给 buffer 对齐
        actions_ragged = np.empty((N, A), dtype=object)
        logp_ragged = np.empty((N, A), dtype=object)
        for i in range(A):
            for n in range(N):
                actions_ragged[n, i] = actions_list[i][n].reshape(-1)
                logp_ragged[n, i] = logp_list[i][n].reshape(-1)

        # 环境用的动作：每个 env 一个 list[agent_i_action]，**保持整数/向量，不做 one-hot**
        actions_env = np.empty((N,), dtype=object)
        for n in range(N):
            per_env = [actions_list[i][n] for i in range(A)]
            actions_env[n] = per_env

        avail_for_buffer = None  # 保持原逻辑：下一时刻的掩码在 insert() 里 post-step 再读
        return values, actions_ragged, logp_ragged, rnn_states, rnn_states_critic, actions_env, avail_for_buffer

    # ------------------- 插入两组 Buffer ------------------- #
    def _align_logp_for_buffer(self, logp_slice, buf):
        """
        将 logp 形状对齐到 buffer 预分配的最后一维：
          - 若 buffer 期待 H（MultiDiscrete逐头），而我们是 [N,*,1]，则 repeat 到 H；
          - 若 buffer 期待 1，而我们是 H，则求和到 1。
        """
        expect = int(buf.action_log_probs.shape[-1])
        got = int(logp_slice.shape[-1])
        if got == expect:
            return logp_slice
        if got == 1 and expect > 1:
            return np.repeat(logp_slice, expect, axis=-1)
        if got > 1 and expect == 1:
            return np.sum(logp_slice, axis=-1, keepdims=True)
        # 兜底：裁切
        return logp_slice[..., :expect]

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,             # object [N, A]
            action_log_probs,    # object [N, A]
            rnn_states,          # [N, A, r, h] —— 新状态（用于 t+1）
            rnn_states_critic,   # [N, A, r, h]
            _,
        ) = data

        N, A = self.n_rollout_threads, self.num_agents

        # --- 规范 dones -> [N, A, 1] 并得到 keep/masks ---
        done_f = self._normalize_dones(dones)  # float32, 1=done, 0=alive
        keep = 1.0 - done_f                    # [N, A, 1]

        # centralized obs
        share_obs = self._build_share_obs(obs, self.envs)

        # 切分 BS / SD
        obs_bs, obs_sd = obs[:, 0:1, :], obs[:, 1:, :]
        g_bs, g_sd = share_obs[:, 0:1, :], share_obs[:, 1:, :]
        rew_bs, rew_sd = rewards[:, 0:1, :], rewards[:, 1:, :]
        keep_bs, keep_sd = keep[:, 0:1, :], keep[:, 1:, :]

        K = int(obs_sd.shape[1]) if obs_sd is not None and obs_sd.ndim == 3 else 0

        rnn_states_bs = rnn_states[:, 0:1, :, :] * keep_bs.reshape(N, 1, 1, 1)
        rnn_states_critic_bs = rnn_states_critic[:, 0:1, :, :] * keep_bs.reshape(N, 1, 1, 1)

        rnn_states_sd = None
        rnn_states_critic_sd = None
        if "SD" in self.buffers and K > 0:
            rnn_states_sd = rnn_states[:, 1:, :, :] * keep_sd.reshape(N, K, 1, 1)
            rnn_states_critic_sd = rnn_states_critic[:, 1:, :, :] * keep_sd.reshape(N, K, 1, 1)

        # 动作/对数概率（object 直接切片，交由 buffer 做 ragged 对齐）
        act_bs = actions[:, 0:1]                      # object [N, 1]
        act_sd = actions[:, 1:] if A > 1 else None
        lps_bs = action_log_probs[:, 0:1]            # object [N, 1]
        lps_sd = action_log_probs[:, 1:] if A > 1 else None

        val_bs = values[:, 0:1, :]
        val_sd = values[:, 1:, :] if A > 1 else None

        # **保持原语义**：在 env.step() 之后（也即“下一时刻”）再次查询 SD 的可行动作掩码，用于写入 buffer 的 step+1
        sd_avail_next = None
        avail_all_next = self._query_available_actions(self.envs)
        if isinstance(avail_all_next, np.ndarray) and avail_all_next.ndim == 3 and "SD" in self.buffers:
            if self.envs.action_space[1].__class__.__name__ == "MultiDiscrete":
                sd_need = int(np.sum(list(self.envs.action_space[1].nvec)))
                if self.buffers["SD"].available_actions is not None and avail_all_next.shape[2] >= sd_need:
                    sd_avail_next = avail_all_next[:, 1:, :sd_need].astype(np.float32)

        # 写入两个 buffer
        self.buffers["BS"].insert(
            g_bs,
            obs_bs,
            rnn_states_bs,
            rnn_states_critic_bs,
            act_bs,
            lps_bs,
            val_bs,
            rew_bs,
            keep_bs.astype(np.float32),
            active_masks=keep_bs.astype(np.float32),
            available_actions=None,
        )

        if "SD" in self.buffers:
            self.buffers["SD"].insert(
                g_sd,
                obs_sd,
                rnn_states_sd,
                rnn_states_critic_sd,
                act_sd,
                lps_sd,
                val_sd,
                rew_sd,
                keep_sd.astype(np.float32),
                active_masks=keep_sd.astype(np.float32),
                available_actions=sd_avail_next,
            )

    # ====================== eval / render ====================== #
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        N, A = self.n_eval_rollout_threads, self.num_agents
        eval_rnn_states = np.zeros(
            (N, A, self.recurrent_N, self.actor_hidden_size),
            dtype=np.float32,
        )
        eval_rnn_states_critic = np.zeros(
            (N, A, self.recurrent_N, self.critic_hidden_size),
            dtype=np.float32,
        )
        eval_masks = np.ones((N, A, 1), dtype=np.float32)

        for _ in range(self.episode_length):
            self.trainer  # noqa
            self._ensure_policy_groups()

            # centralized obs（eval 阶段同样支持 centralized V）
            if self.use_centralized_V and not getattr(self.all_args, "use_obs_instead_of_state", False):
                eval_share_obs = self._build_share_obs(eval_obs, self.eval_envs)
            else:
                eval_share_obs = eval_obs

            # 可行动作掩码
            avail_all = self._query_available_actions(self.eval_envs)
            avail_np = np.asarray(avail_all) if isinstance(avail_all, np.ndarray) else None

            actions_list = []

            for i in range(A):
                group = "BS" if i == 0 else "SD"
                pi = self.trainer.policies[group] if hasattr(self.trainer, "policies") else self.trainer.policy

                share_i = eval_share_obs[:, i, :]
                obs_i = eval_obs[:, i, :]
                rnn_i = eval_rnn_states[:, i, :, :]
                rnnc_i = eval_rnn_states_critic[:, i, :, :]
                mask_i = eval_masks[:, i, :]

                avail_i = None
                if isinstance(avail_np, np.ndarray) and avail_np.ndim == 3:
                    space_i = self.eval_envs.action_space[i]
                    if space_i.__class__.__name__ == "MultiDiscrete":
                        need = int(np.sum(list(space_i.nvec)))
                        if avail_np.shape[2] >= need:
                            avail_i = avail_np[:, i, :need]
                    elif space_i.__class__.__name__ == "Discrete":
                        need = int(space_i.n)
                        if avail_np.shape[2] >= need:
                            avail_i = avail_np[:, i, :need]

                _, a_i, _, rnn_new_i, rnnc_new_i = pi.get_actions(
                    share_i, obs_i, rnn_i, rnnc_i, mask_i, available_actions=avail_i, deterministic=True
                )

                a_i = _t2n(a_i)
                eval_rnn_states[:, i, :, :] = _t2n(rnn_new_i)
                eval_rnn_states_critic[:, i, :, :] = _t2n(rnnc_new_i)

                # eval/render 可按需把 Discrete 转 one-hot；MultiDiscrete 保持原样
                space_i = self.eval_envs.action_space[i]
                if space_i.__class__.__name__ == "MultiDiscrete":
                    actions_list.append(a_i)
                elif space_i.__class__.__name__ == "Discrete":
                    ai_int = a_i.reshape(N, 1).astype(np.int64)
                    one_hot = np.squeeze(np.eye(space_i.n)[ai_int], axis=1)
                    actions_list.append(one_hot)
                else:
                    actions_list.append(a_i)

            eval_actions_env = _pack_actions_env(actions_list)

            # Step eval env
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            # RNN 清零（广播）
            done_f = self._normalize_dones(eval_dones)
            keep = 1.0 - done_f
            eval_rnn_states *= keep.reshape(N, A, 1, 1)
            eval_rnn_states_critic *= keep.reshape(N, A, 1, 1)
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
                self._ensure_policy_groups()

                # 可行动作掩码
                avail_all = self._query_available_actions(envs)
                avail_np = np.asarray(avail_all) if isinstance(avail_all, np.ndarray) else None

                # centralized obs（渲染阶段）
                share_obs = self._build_share_obs(obs, envs)

                N, A = self.n_rollout_threads, self.num_agents
                actions_list = []

                for i in range(A):
                    group = "BS" if i == 0 else "SD"
                    pi = self.trainer.policies[group] if hasattr(self.trainer, "policies") else self.trainer.policy

                    share_i = share_obs[:, i, :]
                    obs_i = obs[:, i, :]
                    rnn_i = rnn_states[:, i, :, :]
                    rnnc_i = rnn_states_critic[:, i, :, :]
                    mask_i = masks[:, i, :]

                    avail_i = None
                    if isinstance(avail_np, np.ndarray) and avail_np.ndim == 3:
                        space_i = envs.action_space[i]
                        if space_i.__class__.__name__ == "MultiDiscrete":
                            need = int(np.sum(list(space_i.nvec)))
                            if avail_np.shape[2] >= need:
                                avail_i = avail_np[:, i, :need]
                        elif space_i.__class__.__name__ == "Discrete":
                            need = int(space_i.n)
                            if avail_np.shape[2] >= need:
                                avail_i = avail_np[:, i, :need]

                    _, action_tensor, _, rnn_states_flat, rnn_states_critic_flat = pi.get_actions(
                        share_i, obs_i, rnn_i, rnnc_i, mask_i, available_actions=avail_i, deterministic=False
                    )

                    a_i = _t2n(action_tensor)
                    rnn_states[:, i, :, :] = _t2n(rnn_states_flat)
                    rnn_states_critic[:, i, :, :] = _t2n(rnn_states_critic_flat)

                    space_i = envs.action_space[i]
                    if space_i.__class__.__name__ == "MultiDiscrete":
                        actions_list.append(a_i)
                    elif space_i.__class__.__name__ == "Discrete":
                        ai_int = a_i.reshape(N, 1).astype(np.int64)
                        one_hot = np.squeeze(np.eye(space_i.n)[ai_int], axis=1)
                        actions_list.append(one_hot)
                    else:
                        actions_list.append(a_i)

                actions_env = _pack_actions_env(actions_list)

                # Step env
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                # RNN 清零（广播）
                done_f = self._normalize_dones(dones)
                keep = 1.0 - done_f
                rnn_states *= keep.reshape(self.n_rollout_threads, self.num_agents, 1, 1)
                rnn_states_critic *= keep.reshape(self.n_rollout_threads, self.num_agents, 1, 1)
                masks = keep.astype(np.float32)

                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                else:
                    envs.render("human")

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

    # ---- 小工具：从 buffers 拿一个“当前 obs”占位（仅供 collect 内部调用）----
    def buffer_obs_like(self):
        if hasattr(self, "buffers") and "SD" in self.buffers:
            return self.buffers["SD"].obs[self.buffers["SD"].step]
        return self.buffer.obs[self.buffer.step]
