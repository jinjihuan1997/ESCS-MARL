# Filename: runner/base_runner_2actor.py
# -*- coding: utf-8 -*-

import os
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensorboardX import SummaryWriter
from gym.spaces import Tuple as GymTuple

from algorithms.onpolicy_utils.shared_buffer_2actor import SharedReplayBuffer


def _t2n(x: torch.Tensor):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


def _resolve_weights_dir(p: str) -> str:
    """
    将用户传入的路径解析为“可加载权重的目录”。
    - 若直接包含 actor/critic 权重文件，则认为已是 models 目录；
    - 若目录下含 models 子目录，则返回 models；
    - 其余返回空串，表示无效。
    """
    if not p:
        return ""
    P = Path(p)
    if not P.exists():
        return ""

    # 2-actor 命名
    if (P / "actor_SD.pt").exists() or (P / "actor_BS.pt").exists():
        return str(P)
    # 单 actor 命名（兜底）
    if (P / "actor.pt").exists() or (P / "critic.pt").exists():
        return str(P)
    # 传的是 run 目录，取其 models
    if (P / "models").exists():
        return str(P / "models")
    return ""


def _weight_signature(module: torch.nn.Module) -> str:
    """简洁的权重签名，便于核对是否加载成功。"""
    try:
        s, n = 0.0, 0
        for p in module.parameters():
            s += p.abs().sum().item()
            n += p.numel()
        return f"abs_sum={s:.3e}, numel={n}"
    except Exception:
        return "N/A"


class Runner(object):
    """
    Base class for training recurrent policies (2-actor version).
    支持两种模式：
      1) 单策略（历史兼容）
      2) 分组策略（BS/SD）：两套 actor，各自独立 critic（不共享），但支持 centralized critic 输入。
    """

    def __init__(self, config):
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']

        # ---------- agent 数 ----------
        if hasattr(self.envs, "num_agents"):
            self.num_agents = int(self.envs.num_agents)
        else:
            act_sp = self.envs.action_space
            if isinstance(act_sp, GymTuple):
                self.num_agents = len(act_sp.spaces)
            elif isinstance(act_sp, (list, tuple)):
                self.num_agents = len(act_sp)
            else:
                self.num_agents = 1

        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']

        # ---------- args ----------
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.actor_hidden_size = self.all_args.actor_hidden_size
        self.critic_hidden_size = self.all_args.critic_hidden_size
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # ---------- intervals ----------
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # ---------- dirs ----------
        # model_dir 可选：未提供或无效 => 空串，后续跳过 restore()
        raw_model_dir = getattr(self.all_args, "model_dir", "") or ""
        self.model_dir = _resolve_weights_dir(raw_model_dir)

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / 'models')
        os.makedirs(self.save_dir, exist_ok=True)

        # ====== IMPORTS (适配当前策略栈路径) ======
        from algorithms.on_policy.r_mappo.algorithm.r_mappo_2actor import RMAPPO as TrainAlgo
        from algorithms.on_policy.r_mappo.algorithm.rMAPPOPolicy_2actor import RMAPPOPolicy as Policy

        # ====== centralized critic 输入空间 ======
        if self.use_centralized_V:
            if getattr(self.all_args, "use_obs_instead_of_state", False):
                cent_obs_space = self.envs.observation_space[0]
            else:
                cent_obs_space = self.envs.share_observation_space[0]
        else:
            cent_obs_space = self.envs.observation_space[0]

        # ================= 分组策略（BS/SD）—— 各自独立 critic =================
        # 约定：env 的第 0 个 agent 为 BS，其余为 SD
        self.K = int(self.num_agents - 1)
        self._multi_group = self.num_agents >= 2  # 启用分组

        if self._multi_group:
            # ---- 两套 policy（各自 Actor + 各自 Critic）----
            obs_space_bs = self.envs.observation_space[0]
            act_space_bs = self.envs.action_space[0]
            obs_space_sd = self.envs.observation_space[1]  # 假设各 SD 观测一致
            act_space_sd = self.envs.action_space[1]

            policy_sd = Policy(self.all_args, obs_space_sd, cent_obs_space, act_space_sd, device=self.device)
            policy_bs = Policy(self.all_args, obs_space_bs, cent_obs_space, act_space_bs, device=self.device)
            # ！！！不再共享 critic，不做任何“指针绑定”

            # ---- 两个 trainer；各自拥有 critic_optimizer + value_normalizer ----
            trainer_sd = TrainAlgo(self.all_args, policy_sd, device=self.device)
            trainer_bs = TrainAlgo(self.all_args, policy_bs, device=self.device)

            # ---- 封装到一个“多组 trainer 容器”以兼容 EnvRunner ----
            multi = types.SimpleNamespace()
            multi.policies = {"SD": policy_sd, "BS": policy_bs}
            multi.trainers = {"SD": trainer_sd, "BS": trainer_bs}
            self.trainer: Any = multi  # 动态容器

            # ---- buffers：SD=K 个 agent；BS=1 个 agent ----
            self.buffers = {
                "SD": SharedReplayBuffer(self.all_args, self.K, obs_space_sd, cent_obs_space, act_space_sd),
                "BS": SharedReplayBuffer(self.all_args, 1,  obs_space_bs, cent_obs_space, act_space_bs),
            }
        else:
            # ================= 单策略（历史兼容）=================
            TrainAlgo_ = TrainAlgo(
                self.all_args,
                Policy(self.all_args,
                       self.envs.observation_space[0],
                       cent_obs_space,
                       self.envs.action_space[0],
                       device=self.device),
                device=self.device
            )
            self.trainer: Any = TrainAlgo_
            self.buffer = SharedReplayBuffer(
                self.all_args,
                self.num_agents,
                self.envs.observation_space[0],
                cent_obs_space,
                self.envs.action_space[0]
            )

        # ====== restore (可选) ======
        if self.model_dir:
            print(f"[INIT] restore from: {self.model_dir}")
            self.restore()
        else:
            print("[INIT] no valid model_dir provided; training from scratch.")

    # -------------------- 训练期公共接口（分组/单策略 兼容） -------------------- #
    @torch.no_grad()
    def compute(self):
        """Calculate returns for collected data."""
        if hasattr(self.trainer, "policies"):  # 分组模式
            for g in ("SD", "BS"):
                tr = self.trainer.trainers[g]  # type: ignore[attr-defined]
                buf = self.buffers[g]
                tr.prep_rollout()

                # [n_rt, n_agent_g, ...] -> [n_rt*n_agent_g, ...]
                cent = buf.share_obs[-1].reshape(-1, *buf.share_obs.shape[3:])
                rnnc = buf.rnn_states_critic[-1].reshape(-1, *buf.rnn_states_critic.shape[3:])
                msk  = buf.masks[-1].reshape(-1, 1)

                nv = tr.policy.get_values(cent, rnnc, msk)
                # 回到 [n_rt, n_agent_g, 1]
                nv = _t2n(nv).reshape(self.n_rollout_threads, -1, 1)
                buf.compute_returns(nv, tr.value_normalizer)
        else:  # 单策略
            self.trainer.prep_rollout()
            cent = self.buffer.share_obs[-1].reshape(-1, *self.buffer.share_obs.shape[3:])
            rnnc = self.buffer.rnn_states_critic[-1].reshape(-1, *self.buffer.rnn_states_critic.shape[3:])
            msk  = self.buffer.masks[-1].reshape(-1, 1)

            nv = self.trainer.policy.get_values(cent, rnnc, msk)
            nv = _t2n(nv).reshape(self.n_rollout_threads, -1, 1)
            self.buffer.compute_returns(nv, self.trainer.value_normalizer)

    def train(self):
        """Train policies with data in buffer."""
        if hasattr(self.trainer, "policies"):  # 分组
            infos = {}
            # 先训练 SD 再训练 BS（各自独立 critic）
            for g in ("SD", "BS"):
                tr = self.trainer.trainers[g]  # type: ignore[attr-defined]
                buf = self.buffers[g]
                tr.prep_training()
                ti = tr.train(buf, update_actor=True)
                buf.after_update()
                # 指标加组前缀
                for k, v in ti.items():
                    infos[f"{g}/{k}"] = v
            return infos
        else:  # 单策略
            self.trainer.prep_training()
            infos = self.trainer.train(self.buffer)
            self.buffer.after_update()
            return infos

    def save(self):
        """Save policies."""
        if hasattr(self.trainer, "policies"):  # 分组
            pols = self.trainer.policies  # type: ignore[attr-defined]
            trs  = self.trainer.trainers

            # 两个 actor
            torch.save(pols["SD"].actor.state_dict(), os.path.join(self.save_dir, "actor_SD.pt"))
            torch.save(pols["BS"].actor.state_dict(), os.path.join(self.save_dir, "actor_BS.pt"))

            # 两个 critic（不再共享）
            torch.save(pols["SD"].critic.state_dict(), os.path.join(self.save_dir, "critic_SD.pt"))
            torch.save(pols["BS"].critic.state_dict(), os.path.join(self.save_dir, "critic_BS.pt"))

            # 各自的 critic optimizer（可选保存）
            if hasattr(trs["SD"].policy, "critic_optimizer") and trs["SD"].policy.critic_optimizer is not None:
                torch.save(trs["SD"].policy.critic_optimizer.state_dict(),
                           os.path.join(self.save_dir, "critic_optim_SD.pt"))
            if hasattr(trs["BS"].policy, "critic_optimizer") and trs["BS"].policy.critic_optimizer is not None:
                torch.save(trs["BS"].policy.critic_optimizer.state_dict(),
                           os.path.join(self.save_dir, "critic_optim_BS.pt"))

            # 各自的 value_normalizer（若启用）
            if getattr(trs["SD"], "value_normalizer", None) is not None:
                torch.save(trs["SD"].value_normalizer.state_dict(),
                           os.path.join(self.save_dir, "value_norm_SD.pt"))
            if getattr(trs["BS"], "value_normalizer", None) is not None:
                torch.save(trs["BS"].value_normalizer.state_dict(),
                           os.path.join(self.save_dir, "value_norm_BS.pt"))
        else:
            torch.save(self.trainer.policy.actor.state_dict(), os.path.join(self.save_dir, "actor.pt"))
            torch.save(self.trainer.policy.critic.state_dict(), os.path.join(self.save_dir, "critic.pt"))

    def restore(self):
        """Restore policies (best-effort)."""
        if not self.model_dir:
            print("[RESTORE] skipped (empty model_dir).")
            return

        if hasattr(self.trainer, "policies"):  # 分组
            pols = self.trainer.policies  # type: ignore[attr-defined]
            trs  = self.trainer.trainers
            p_sd, p_bs = pols["SD"], pols["BS"]

            # ---- 优先加载 2-actor 命名 ----
            path_sd_actor = os.path.join(self.model_dir, 'actor_SD.pt')
            path_bs_actor = os.path.join(self.model_dir, 'actor_BS.pt')
            path_sd_cr    = os.path.join(self.model_dir, 'critic_SD.pt')
            path_bs_cr    = os.path.join(self.model_dir, 'critic_BS.pt')

            # ---- 若不存在，再尝试单 actor 命名作为兜底（两边都加载同一份）----
            single_actor = os.path.join(self.model_dir, 'actor.pt')
            single_critic = os.path.join(self.model_dir, 'critic.pt')

            # actor
            if os.path.isfile(path_sd_actor):
                p_sd.actor.load_state_dict(torch.load(path_sd_actor, map_location=self.device), strict=False)
            elif os.path.isfile(single_actor):
                p_sd.actor.load_state_dict(torch.load(single_actor, map_location=self.device), strict=False)

            if os.path.isfile(path_bs_actor):
                p_bs.actor.load_state_dict(torch.load(path_bs_actor, map_location=self.device), strict=False)
            elif os.path.isfile(single_actor):
                p_bs.actor.load_state_dict(torch.load(single_actor, map_location=self.device), strict=False)

            # critic
            if os.path.isfile(path_sd_cr):
                p_sd.critic.load_state_dict(torch.load(path_sd_cr, map_location=self.device), strict=False)
            elif os.path.isfile(single_critic):
                p_sd.critic.load_state_dict(torch.load(single_critic, map_location=self.device), strict=False)

            if os.path.isfile(path_bs_cr):
                p_bs.critic.load_state_dict(torch.load(path_bs_cr, map_location=self.device), strict=False)
            elif os.path.isfile(single_critic):
                p_bs.critic.load_state_dict(torch.load(single_critic, map_location=self.device), strict=False)

            # 各自 critic optimizer（若存在）
            path_sd_opt = os.path.join(self.model_dir, 'critic_optim_SD.pt')
            path_bs_opt = os.path.join(self.model_dir, 'critic_optim_BS.pt')
            if os.path.isfile(path_sd_opt) and hasattr(trs["SD"].policy, "critic_optimizer"):
                trs["SD"].policy.critic_optimizer.load_state_dict(
                    torch.load(path_sd_opt, map_location=self.device)
                )
            if os.path.isfile(path_bs_opt) and hasattr(trs["BS"].policy, "critic_optimizer"):
                trs["BS"].policy.critic_optimizer.load_state_dict(
                    torch.load(path_bs_opt, map_location=self.device)
                )

            # 各自 value_normalizer（若启用）
            path_sd_vn = os.path.join(self.model_dir, 'value_norm_SD.pt')
            path_bs_vn = os.path.join(self.model_dir, 'value_norm_BS.pt')
            if os.path.isfile(path_sd_vn) and getattr(trs["SD"], "value_normalizer", None) is not None:
                trs["SD"].value_normalizer.load_state_dict(
                    torch.load(path_sd_vn, map_location=self.device)
                )
            if os.path.isfile(path_bs_vn) and getattr(trs["BS"], "value_normalizer", None) is not None:
                trs["BS"].value_normalizer.load_state_dict(
                    torch.load(path_bs_vn, map_location=self.device)
                )

            print(f"[LOAD] actor_SD  : {path_sd_actor if os.path.isfile(path_sd_actor) else ('fallback:'+single_actor if os.path.isfile(single_actor) else 'NOT FOUND')}")
            print(f"[LOAD] actor_BS  : {path_bs_actor if os.path.isfile(path_bs_actor) else ('fallback:'+single_actor if os.path.isfile(single_actor) else 'NOT FOUND')}")
            print(f"[LOAD] critic_SD : {path_sd_cr if os.path.isfile(path_sd_cr) else ('fallback:'+single_critic if os.path.isfile(single_critic) else 'NOT FOUND')}")
            print(f"[LOAD] critic_BS : {path_bs_cr if os.path.isfile(path_bs_cr) else ('fallback:'+single_critic if os.path.isfile(single_critic) else 'NOT FOUND')}")

            try:
                print(f"[WEIGHT] SD/Actor : {_weight_signature(p_sd.actor)}")
                print(f"[WEIGHT] BS/Actor : {_weight_signature(p_bs.actor)}")
                print(f"[WEIGHT] SD/Critic: {_weight_signature(p_sd.critic)}")
                print(f"[WEIGHT] BS/Critic: {_weight_signature(p_bs.critic)}")
            except Exception as e:
                print(f"[WEIGHT] signature error: {e}")

        else:  # 单策略
            actor_path = os.path.join(self.model_dir, 'actor.pt')
            critic_path = os.path.join(self.model_dir, 'critic.pt')

            if os.path.isfile(actor_path):
                self.trainer.policy.actor.load_state_dict(torch.load(actor_path, map_location=self.device),
                                                          strict=False)
            if (not self.use_render) and os.path.isfile(critic_path):
                self.trainer.policy.critic.load_state_dict(torch.load(critic_path, map_location=self.device),
                                                           strict=False)

            print(f"[LOAD] Actor path : {actor_path if os.path.isfile(actor_path) else 'NOT FOUND'}")
            print(f"[LOAD] Critic path: {critic_path if os.path.isfile(critic_path) else 'NOT FOUND'}")
            try:
                print(f"[WEIGHT] Actor  : {_weight_signature(self.trainer.policy.actor)}")
                print(f"[WEIGHT] Critic : {_weight_signature(self.trainer.policy.critic)}")
            except Exception as e:
                print(f"[WEIGHT] signature error: {e}")

    def log_train(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    # ---- 以下接口交由 EnvRunner 实现 ----
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
