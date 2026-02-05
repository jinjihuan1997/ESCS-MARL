# Filepath: algorithms/on_policy/r_mappo/algorithm/rMAPPOPolicy.py
# -*- coding: utf-8 -*-

import torch
from algorithms.on_policy.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from utils.util import update_linear_schedule


class RMAPPOPolicy:
    """
    MAPPO Policy wrapper.
    - 与 EnvCore 的 MultiDiscrete([dir_idx, hov_idx, ds_idx]) 与动作掩码直接对接；
      动作掩码的具体规范化在 R_Actor/ACTLayer 内部已处理，这里无需关心其形态。
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

        # 构建 Actor / Critic（内部已根据 obs 形态自动选择 MLP/CNN）
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

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks,
                    available_actions=None, deterministic=False):
        """
        推理：给出 actions / log_probs / values 及更新后的 RNN 隐状态。
        :return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """仅取 critic 值函数。"""
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        训练阶段：计算 log_probs / entropy 与 values。
        :return values, action_log_probs, dist_entropy
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        仅 Actor 推理（用于 rollout）。
        :return actions, rnn_states_actor
        """
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        return actions, rnn_states_actor
