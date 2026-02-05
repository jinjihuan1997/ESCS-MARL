# Filepath: algorithms/on_policy/r_mappo/algorithm/rMAPPOPolicy_2actor.py
# -*- coding: utf-8 -*-

import torch
import numpy as np
from algorithms.on_policy.r_mappo.algorithm.r_actor_critic_2actor import R_Actor, R_Critic
from utils.util import update_linear_schedule


def _act_dim_from_space(space) -> int:
    name = space.__class__.__name__
    if name == "Discrete":
        return 1
    if name == "MultiDiscrete":
        # 返回“头数”，而不是 sum(nvec)
        if hasattr(space, "nvec"):
            return int(len(space.nvec))
        if hasattr(space, "shape") and len(space.shape) > 0:
            return int(space.shape[0])
        if hasattr(space, "high"):
            return int(np.asarray(space.high).shape[0])
        return 1
    if name == "MultiBinary":
        if hasattr(space, "n"):
            return int(space.n)
        if hasattr(space, "shape") and len(space.shape) > 0:
            return int(space.shape[0])
        return 1
    if name == "Box":
        return int(space.shape[0])
    if name == "Tuple":
        try:
            children = getattr(space, "spaces", None)
            if children is None:
                children = list(space)
            return int(sum(_act_dim_from_space(s) for s in children))
        except Exception:
            return 1
    # fallback
    try:
        return int(space.shape[0])
    except Exception:
        return 1


def _mask_width_from_space(space) -> int:
    """返回扁平掩码宽度（Box=0；Discrete=n；MultiDiscrete=sum(nvec)；MultiBinary=n）"""
    name = space.__class__.__name__
    if name == "Discrete":
        return int(getattr(space, "n"))
    if name == "MultiDiscrete":
        if hasattr(space, "nvec"):
            return int(np.sum(np.asarray(space.nvec, dtype=np.int64)))
        if hasattr(space, "high") and hasattr(space, "low"):
            nvec = (np.asarray(space.high) - np.asarray(space.low) + 1).astype(np.int64)
            return int(np.sum(nvec))
        return 0
    if name == "MultiBinary":
        if hasattr(space, "n"):
            return int(space.n)
        if hasattr(space, "shape") and len(space.shape) > 0:
            return int(space.shape[0])
        return 0
    return 0  # Box / others


def _clip_last_dim(x, need):
    """
    安全裁剪最后一维到 need：
      - x 为 torch.Tensor 或 np.ndarray：若 shape[-1] > need，做 x[..., :need]
      - x 为 list/tuple（按头掩码）：直接返回（交由 ACTLayer 规范化处理）
      - 其他类型：原样返回
    """
    if x is None or need is None or need <= 0:
        return None if need <= 0 else x
    if isinstance(x, (list, tuple)):
        return x
    try:
        last = x.shape[-1]
        if last > need:
            return x[..., :need]
        return x
    except Exception:
        return x


class RMAPPOPolicy:
    """
    MAPPO Policy wrapper.

    - 兼容异构动作空间（例如 BS=Box，SD=MultiDiscrete）。
    - 不直接处理掩码细节，掩码/分布细节由 R_Actor/ACTLayer 实现；
      这里仅做“安全裁剪”：当 runner/buffer 传入对齐后的更宽张量时，裁剪到本策略实际所需维度。
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        # 预存本策略对应的“动作维度”和“掩码宽度”（用于训练阶段的安全裁剪）
        self._act_dim = _act_dim_from_space(self.act_space)
        self._mask_width = _mask_width_from_space(self.act_space)

        # 构建 Actor / Critic（内部已根据 obs 形态自动选择 MLP/CNN；Actor 内部根据 act_space 选择分布）
        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        # 优化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay
        )

    def lr_decay(self, episode, episodes):
        """线性学习率衰减。"""
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    @torch.no_grad()
    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks,
                    available_actions=None, deterministic=False):
        """
        推理：给出 actions / log_probs / values 及更新后的 RNN 隐状态。
        Note:
          - available_actions 可能为 None / 扁平张量 / 按头 list；这里只对扁平张量做宽度裁剪。
        :return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        avail = _clip_last_dim(available_actions, self._mask_width) if self._mask_width > 0 else None

        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, avail, deterministic
        )
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    @torch.no_grad()
    def get_values(self, cent_obs, rnn_states_critic, masks):
        """仅取 critic 值函数。"""
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        训练阶段：计算 log_probs / entropy 与 values。
        关键点：当 buffer 对齐后把 action/available_actions 变“更宽”时，先按本策略的实际需要裁剪。
        :return values, action_log_probs, dist_entropy
        """
        # 裁剪动作到本策略的动作维度（例如：BS=Box(M)，SD=MultiDiscrete(H)）
        act = _clip_last_dim(action, self._act_dim)

        # 裁剪可行动作掩码到本策略所需宽度（Box 时置 None）
        avail = _clip_last_dim(available_actions, self._mask_width) if self._mask_width > 0 else None

        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, act, masks, avail, active_masks
        )
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    @torch.no_grad()
    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        仅 Actor 推理（用于 rollout）。
        """
        avail = _clip_last_dim(available_actions, self._mask_width) if self._mask_width > 0 else None
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, avail, deterministic
        )
        return actions, rnn_states_actor
