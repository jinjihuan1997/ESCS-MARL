# =====================================
# Filepath: algorithms/onpolicy_utils/shared_buffer_2actor.py
# -*- coding: utf-8 -*-
# =====================================

import torch
import numpy as np
from typing import List
from utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def _act_dim_from_space(act_space) -> int:
    """
    返回“动作向量最后一维长度”(int)：
      - Discrete          -> 1
      - MultiDiscrete     -> len(nvec)（头数）
      - MultiBinary       -> n
      - Box               -> shape[0]
      - Tuple             -> 递归求和
    """
    name = act_space.__class__.__name__
    if name == "Discrete":
        return 1
    if name == "MultiDiscrete":
        if hasattr(act_space, "nvec"):
            return int(len(act_space.nvec))
        if hasattr(act_space, "shape") and len(act_space.shape) > 0:
            return int(act_space.shape[0])
        if hasattr(act_space, "high"):
            return int(np.asarray(act_space.high).shape[0])
        return 1
    if name == "MultiBinary":
        if hasattr(act_space, "n"):
            return int(act_space.n)
        if hasattr(act_space, "shape") and len(act_space.shape) > 0:
            return int(act_space.shape[0])
        return 1
    if name == "Box":
        return int(act_space.shape[0])
    if name == "Tuple":
        try:
            children = getattr(act_space, "spaces", None)
            if children is None:
                children = list(act_space)
            return int(sum(_act_dim_from_space(s) for s in children))
        except Exception:
            return 1

    # 兜底：尝试 util
    try:
        raw = get_shape_from_act_space(act_space)
    except Exception:
        raw = None
    if isinstance(raw, (int, np.integer, float, np.floating)):
        return int(raw)
    if isinstance(raw, (tuple, list, np.ndarray)):
        if len(raw) == 1 and isinstance(raw[0], (int, np.integer, float, np.floating)):
            return int(raw[0])
        return int(len(raw))
    try:
        return int(act_space.shape[0])
    except Exception:
        return 1


def _sum_nvec_from_space(act_space) -> int:
    """
    返回该空间对应掩码“扁平宽度”（仅对需要掩码的空间）：
      - Discrete      -> n
      - MultiDiscrete -> sum(nvec)
      - MultiBinary   -> n
      - Box / 其他    -> 0（不需要掩码）
    """
    name = act_space.__class__.__name__
    if name == "Discrete":
        return int(getattr(act_space, "n"))
    if name == "MultiDiscrete":
        if hasattr(act_space, "nvec"):
            return int(np.sum(np.asarray(act_space.nvec, dtype=np.int64)))
        if hasattr(act_space, "high") and hasattr(act_space, "low"):
            nvec = (np.asarray(act_space.high) - np.asarray(act_space.low) + 1).astype(np.int64)
            return int(np.sum(nvec))
        return 0
    if name == "MultiBinary":
        if hasattr(act_space, "n"):
            return int(act_space.n)
        if hasattr(act_space, "shape") and len(act_space.shape) > 0:
            return int(act_space.shape[0])
        return 0
    return 0  # Box/others


# --- in algorithms/onpolicy_utils/shared_buffer.py ---

def _logp_dim_from_space(act_space) -> int:
    """
    约定：
      - Box           -> 1
      - MultiDiscrete -> 1  # 关键：与 ACTLayer 的逐头求和对齐
      - Discrete      -> 1
      - MultiBinary   -> 1
      - Tuple         -> 子空间 logp_dim 求和
    """
    name = act_space.__class__.__name__
    if name == "Box":
        return 1
    if name == "MultiDiscrete":
        return 1        # ← 改这里（原来是返回头数）
    if name in ("Discrete", "MultiBinary"):
        return 1
    if name == "Tuple":
        try:
            children = getattr(act_space, "spaces", None) or list(act_space)
            return int(sum(_logp_dim_from_space(s) for s in children))
        except Exception:
            return 1
    return 1



def _analyze_act_spaces(act_space_in, num_agents: int):
    """
    支持两种传参：
      - 同构：act_space_in 是一个 gym.Space，表示所有 agent 相同
      - 异构：act_space_in 是长度=num_agents 的 list/tuple
    返回：
      spaces_per_agent: List[Space]
      act_dims:         List[int]
      logp_dims:        List[int]
      mask_widths:      List[int]
      max_act_dim, max_logp_dim, max_mask_width
    """
    if isinstance(act_space_in, (list, tuple)):
        assert len(act_space_in) == num_agents, \
            f"异构动作空间应与 num_agents 一致：got {len(act_space_in)} vs {num_agents}"
        spaces_per_agent = list(act_space_in)
    else:
        spaces_per_agent = [act_space_in for _ in range(num_agents)]

    act_dims = [int(_act_dim_from_space(s)) for s in spaces_per_agent]
    logp_dims = [int(_logp_dim_from_space(s)) for s in spaces_per_agent]
    mask_widths = [int(_sum_nvec_from_space(s)) for s in spaces_per_agent]

    max_act_dim = int(max(act_dims)) if len(act_dims) > 0 else 1
    max_logp_dim = int(max(logp_dims)) if len(logp_dims) > 0 else 1
    max_mask_width = int(max(mask_widths)) if len(mask_widths) > 0 else 0

    return spaces_per_agent, act_dims, logp_dims, mask_widths, max_act_dim, max_logp_dim, max_mask_width


def _pad_ragged_actions(actions, N, A, max_act_dim, per_agent_dims: List[int]) -> np.ndarray:
    """
    将可能是 ragged/object 的 actions 统一为 [N, A, max_act_dim] float32/float安全类型：
      - 若已是等宽 ndarray，直接返回
      - 若是 object/列表：逐 agent 按 per_agent_dims[i] 拷贝到前缀，右侧补零
    说明：为兼容 int/float 混合，这里统一转为 float32；训练端可在用到时再根据策略需要强制转换 dtype。
    """
    # 已是规则数组
    if isinstance(actions, np.ndarray) and actions.dtype != object and actions.ndim == 3:
        if actions.shape[1] == A and actions.shape[2] == max_act_dim:
            return actions.astype(np.float32, copy=False)

    out = np.zeros((N, A, max_act_dim), dtype=np.float32)

    # List/tuple/object-ndarray：按 env、agent 逐个填
    if isinstance(actions, np.ndarray) and actions.dtype == object and actions.ndim == 2:
        assert actions.shape == (N, A), f"object actions 期望形状 (N={N}, A={A})，got {actions.shape}"
        for n in range(N):
            for i in range(A):
                ai = np.asarray(actions[n, i]).reshape(-1)
                use = min(ai.size, per_agent_dims[i], max_act_dim)
                if use > 0:
                    out[n, i, :use] = ai[:use]
        return out

    if isinstance(actions, (list, tuple)) and len(actions) == N:
        # actions[n] 期望为长度 A 的 list/tuple/object-ndarray
        for n in range(N):
            row = actions[n]
            if isinstance(row, np.ndarray) and row.dtype == object and row.ndim == 1 and row.size == A:
                for i in range(A):
                    ai = np.asarray(row[i]).reshape(-1)
                    use = min(ai.size, per_agent_dims[i], max_act_dim)
                    if use > 0:
                        out[n, i, :use] = ai[:use]
            elif isinstance(row, (list, tuple)) and len(row) == A:
                for i in range(A):
                    ai = np.asarray(row[i]).reshape(-1)
                    use = min(ai.size, per_agent_dims[i], max_act_dim)
                    if use > 0:
                        out[n, i, :use] = ai[:use]
            else:
                raise ValueError("无法解析 actions 的第 n 行结构；应为长度 A 的 list/tuple/object-ndarray")
        return out

    # 已是等维 ndarray 但列宽 != max_act_dim：尝试右侧补零
    if isinstance(actions, np.ndarray) and actions.ndim == 3 and actions.shape[1] == A:
        K = actions.shape[2]
        out[:, :, :min(K, max_act_dim)] = actions[:, :, :min(K, max_act_dim)]
        return out

    # 其他情形：尽力 reshape，再不行就抛错
    try:
        arr = np.asarray(actions, dtype=np.float32)
        arr = arr.reshape(N, A, -1)
        K = arr.shape[2]
        out[:, :, :min(K, max_act_dim)] = arr[:, :, :min(K, max_act_dim)]
        return out
    except Exception as e:
        raise ValueError(f"无法对齐 actions 到 [N,A,max_act_dim]：{type(actions)}, err={e}")


def _pad_ragged_logp(action_log_probs, N, A, max_logp_dim, per_agent_logp_dims: List[int]) -> np.ndarray:
    """
    对齐 log-probs 到 [N, A, max_logp_dim]（右侧补零）。
    """
    if isinstance(action_log_probs, np.ndarray) and action_log_probs.dtype != object and action_log_probs.ndim == 3:
        if action_log_probs.shape[1] == A and action_log_probs.shape[2] == max_logp_dim:
            return action_log_probs.astype(np.float32, copy=False)

    out = np.zeros((N, A, max_logp_dim), dtype=np.float32)

    if isinstance(action_log_probs, np.ndarray) and action_log_probs.dtype == object and action_log_probs.ndim == 2:
        assert action_log_probs.shape == (N, A)
        for n in range(N):
            for i in range(A):
                lp = np.asarray(action_log_probs[n, i]).reshape(-1)
                use = min(lp.size, per_agent_logp_dims[i], max_logp_dim)
                if use > 0:
                    out[n, i, :use] = lp[:use]
        return out

    if isinstance(action_log_probs, (list, tuple)) and len(action_log_probs) == N:
        for n in range(N):
            row = action_log_probs[n]
            if isinstance(row, (list, tuple)) and len(row) == A:
                for i in range(A):
                    lp = np.asarray(row[i]).reshape(-1)
                    use = min(lp.size, per_agent_logp_dims[i], max_logp_dim)
                    if use > 0:
                        out[n, i, :use] = lp[:use]
            else:
                raise ValueError("无法解析 action_log_probs 的第 n 行结构；应为长度 A 的 list/tuple")
        return out

    # 等维但宽度不同：补零
    if isinstance(action_log_probs, np.ndarray) and action_log_probs.ndim == 3 and action_log_probs.shape[1] == A:
        K = action_log_probs.shape[2]
        out[:, :, :min(K, max_logp_dim)] = action_log_probs[:, :, :min(K, max_logp_dim)]
        return out

    try:
        arr = np.asarray(action_log_probs, dtype=np.float32).reshape(N, A, -1)
        K = arr.shape[2]
        out[:, :, :min(K, max_logp_dim)] = arr[:, :, :min(K, max_logp_dim)]
        return out
    except Exception as e:
        raise ValueError(f"无法对齐 action_log_probs 到 [N,A,max_logp_dim]：{type(action_log_probs)}, err={e}")


def _pad_ragged_avail(avail, N, A, max_mask_width, per_agent_mask_widths: List[int]) -> np.ndarray:
    """
    将 available_actions 对齐到 [N, A, max_mask_width]；对“无需掩码”的 agent（如 Box），其行保持全 0。
    """
    out = np.zeros((N, A, max_mask_width), dtype=np.float32)
    if avail is None:
        return out  # 直接返回全 0，调用方可据需要忽略

    # 已是等宽
    if isinstance(avail, np.ndarray) and avail.dtype != object and avail.ndim == 3:
        if avail.shape == (N, A, max_mask_width):
            return avail.astype(np.float32, copy=False)

    # object 或 list
    if isinstance(avail, np.ndarray) and avail.dtype == object and avail.ndim == 2 and avail.shape == (N, A):
        for n in range(N):
            for i in range(A):
                vi = np.asarray(avail[n, i]).reshape(-1)
                use = min(vi.size, per_agent_mask_widths[i], max_mask_width)
                if use > 0:
                    out[n, i, :use] = vi[:use]
        return out

    if isinstance(avail, (list, tuple)) and len(avail) == N:
        for n in range(N):
            row = avail[n]
            if isinstance(row, (list, tuple)) and len(row) == A:
                for i in range(A):
                    vi = np.asarray(row[i]).reshape(-1)
                    use = min(vi.size, per_agent_mask_widths[i], max_mask_width)
                    if use > 0:
                        out[n, i, :use] = vi[:use]
            else:
                raise ValueError("无法解析 available_actions 的第 n 行结构；应为长度 A 的 list/tuple")
        return out

    # 等维但宽度不同：补零
    if isinstance(avail, np.ndarray) and avail.ndim == 3 and avail.shape[1] == A:
        K = avail.shape[2]
        out[:, :, :min(K, max_mask_width)] = avail[:, :, :min(K, max_mask_width)]
        return out

    try:
        arr = np.asarray(avail, dtype=np.float32).reshape(N, A, -1)
        K = arr.shape[2]
        out[:, :, :min(K, max_mask_width)] = arr[:, :, :min(K, max_mask_width)]
        return out
    except Exception as e:
        raise ValueError(f"无法对齐 available_actions 到 [N,A,max_mask_width]：{type(avail)}, err={e}")


class SharedReplayBuffer(object):
    """
    Buffer to store training data for (R)MAPPO.

    适配点：
      - 支持异构动作空间（例如 0 号 BS=Box，1..K 号 SD=MultiDiscrete）。
      - actions / action_log_probs 统一为 max 宽度，并在 insert 时自动补零对齐。
      - available_actions 亦支持异构（Box 的 mask 宽度=0，整行保留 0）。
    """

    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.actor_hidden_size = args.actor_hidden_size
        self.critic_hidden_size = args.critic_hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits

        # ---------- obs shapes ----------
        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)
        if isinstance(obs_shape[-1], list):
            obs_shape = obs_shape[:1]
        if isinstance(share_obs_shape[-1], list):
            share_obs_shape = share_obs_shape[:1]

        # ---------- 动作空间分析（支持异构） ----------
        (
            self._spaces_per_agent,
            self._act_dims_per_agent,
            self._logp_dims_per_agent,
            self._mask_widths_per_agent,
            self._max_act_dim,
            self._max_logp_dim,
            self._max_mask_width,
        ) = _analyze_act_spaces(act_space, num_agents)

        # 是否异构（仅用于参考/调试）
        self._heterogeneous = len(set(self._act_dims_per_agent)) > 1 or \
                              len(set(self._logp_dims_per_agent)) > 1 or \
                              len(set(self._mask_widths_per_agent)) > 1

        # ---------- buffers ----------
        self.share_obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape),
            dtype=np.float32
        )
        self.obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape),
            dtype=np.float32
        )

        self.rnn_states = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.actor_hidden_size),
            dtype=np.float32
        )
        self.rnn_states_critic = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.critic_hidden_size),
            dtype=np.float32
        )

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32
        )
        self.returns = np.zeros_like(self.value_preds)

        # available_actions：若所有 agent 的 mask_width 都为 0，则置 None
        if self._max_mask_width > 0:
            self.available_actions = np.zeros(
                (self.episode_length + 1, self.n_rollout_threads, num_agents, self._max_mask_width),
                dtype=np.float32
            )
        else:
            self.available_actions = None

        # actions 与 log_probs：统一 max 宽度
        # 为兼容连续/离散混合，actions 统一 float32，策略更新时可自行转换 dtype
        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, self._max_act_dim),
            dtype=np.float32
        )
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, self._max_logp_dim),
            dtype=np.float32
        )

        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32
        )
        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32
        )
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    # ----------- 插入 -----------
    def insert(self, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        """
        支持 actions / action_log_probs / available_actions 为“等宽 ndarray”
        或 “ragged/object/list”，函数会自动补零到内部 max 宽度。
        """
        N = self.n_rollout_threads
        A = share_obs.shape[1]  # num_agents

        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()

        # actions / logp 对齐
        actions_padded = _pad_ragged_actions(
            actions, N, A, self._max_act_dim, self._act_dims_per_agent
        )
        logp_padded = _pad_ragged_logp(
            action_log_probs, N, A, self._max_logp_dim, self._logp_dims_per_agent
        )
        self.actions[self.step] = actions_padded
        self.action_log_probs[self.step] = logp_padded

        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()

        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()

        if (self.available_actions is not None) and (available_actions is not None):
            avail_padded = _pad_ragged_avail(
                available_actions, N, A, self._max_mask_width, self._mask_widths_per_agent
            )
            self.available_actions[self.step + 1] = avail_padded

        self.step = (self.step + 1) % self.episode_length

    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        """Turn-based environments (e.g., Hanabi)."""
        N = self.n_rollout_threads
        A = share_obs.shape[1]

        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()

        actions_padded = _pad_ragged_actions(
            actions, N, A, self._max_act_dim, self._act_dims_per_agent
        )
        logp_padded = _pad_ragged_logp(
            action_log_probs, N, A, self._max_logp_dim, self._logp_dims_per_agent
        )
        self.actions[self.step] = actions_padded
        self.action_log_probs[self.step] = logp_padded

        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()

        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()

        if (self.available_actions is not None) and (available_actions is not None):
            avail_padded = _pad_ragged_avail(
                available_actions, N, A, self._max_mask_width, self._mask_widths_per_agent
            )
            self.available_actions[self.step] = avail_padded

        self.step = (self.step + 1) % self.episode_length

    # ----------- 更新收尾 -----------
    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def chooseafter_update(self):
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    # ----------------- Returns / GAE -----------------
    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(
                            self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    # ----------------- Generators -----------------
    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) * number of steps ({}) * number of agents ({}) = {} "
                "to be >= the number of PPO mini batches ({}).".format(
                    n_rollout_threads, episode_length, num_agents, batch_size, num_mini_batch
                )
            )
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])  # 等宽，策略端可按各自维度切片
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        available_actions = None
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])

        for indices in sampler:
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            available_actions_batch = available_actions[indices] if available_actions is not None else None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            adv_targ = advantages[indices] if advantages is not None else None

            yield (share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,
                   value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,
                   adv_targ, available_actions_batch)

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * num_agents
        assert n_rollout_threads * num_agents >= num_mini_batch, (
            "PPO requires (#proc {} * #agents {}) >= #mini-batches {}."
            .format(n_rollout_threads, num_agents, num_mini_batch)
        )
        num_envs_per_batch = batch_size // num_mini_batch
        perm = torch.randperm(batch_size).numpy()

        share_obs = self.share_obs.reshape(-1, batch_size, *self.share_obs.shape[3:])
        obs = self.obs.reshape(-1, batch_size, *self.obs.shape[3:])
        rnn_states = self.rnn_states.reshape(-1, batch_size, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic.reshape(-1, batch_size, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, batch_size, self.actions.shape[-1])
        value_preds = self.value_preds.reshape(-1, batch_size, 1)
        returns = self.returns.reshape(-1, batch_size, 1)
        masks = self.masks.reshape(-1, batch_size, 1)
        active_masks = self.active_masks.reshape(-1, batch_size, 1)
        action_log_probs = self.action_log_probs.reshape(-1, batch_size, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, batch_size, 1)

        available_actions = None
        if self.available_actions is not None:
            available_actions = self.available_actions.reshape(-1, batch_size, self.available_actions.shape[-1])

        for start_ind in range(0, batch_size, num_envs_per_batch):
            share_obs_batch, obs_batch = [], []
            rnn_states_batch, rnn_states_critic_batch = [], []
            actions_batch, available_actions_batch = [], []
            value_preds_batch, return_batch = [], []
            masks_batch, active_masks_batch = [], []
            old_action_log_probs_batch, adv_targ = [], []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(share_obs[:-1, ind])
                obs_batch.append(obs[:-1, ind])
                rnn_states_batch.append(rnn_states[0:1, ind])
                rnn_states_critic_batch.append(rnn_states_critic[0:1, ind])
                actions_batch.append(actions[:, ind])
                if available_actions is not None:
                    available_actions_batch.append(available_actions[:-1, ind])
                value_preds_batch.append(value_preds[:-1, ind])
                return_batch.append(returns[:-1, ind])
                masks_batch.append(masks[:-1, ind])
                active_masks_batch.append(active_masks[:-1, ind])
                old_action_log_probs_batch.append(action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.episode_length, num_envs_per_batch
            share_obs_batch = np.stack(share_obs_batch, 1)
            obs_batch = np.stack(obs_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            share_obs_batch = _flatten(T, N, share_obs_batch)
            obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield (share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,
                   value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,
                   adv_targ, available_actions_batch)

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        if len(self.share_obs.shape) > 4:
            share_obs = self.share_obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        rnn_states = self.rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 2, 0, 3, 4).reshape(
            -1, *self.rnn_states_critic.shape[3:]
        )

        available_actions = None
        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            share_obs_batch, obs_batch = [], []
            rnn_states_batch, rnn_states_critic_batch = [], []
            actions_batch, available_actions_batch = [], []
            value_preds_batch, return_batch = [], []
            masks_batch, active_masks_batch = [], []
            old_action_log_probs_batch, adv_targ = [], []

            for index in indices:
                ind = index * data_chunk_length
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                if available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                return_batch.append(returns[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind + data_chunk_length])
                adv_targ.append(advantages[ind:ind + data_chunk_length])
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            if available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield (share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,
                   value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,
                   adv_targ, available_actions_batch)
