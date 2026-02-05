# Filepath: algorithms/on_policy/r_mappo/algorithm/r_mappo.py
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from utils.valuenorm import ValueNorm
from algorithms.on_policy.utils.util import check


class RMAPPO():
    """
    Trainer class for MAPPO to update policies.
    适配要点：
      - 对优势adv的标准化过程增加全NaN与零方差兜底；
      - 在PPO更新中对adv_targ做nan_to_num；
      - 使用active_masks时加入防0除法保护。
    """

    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        assert not (self._use_popart and self._use_valuenorm), \
            "self._use_popart and self._use_valuenorm can not be True simultaneously"

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )

        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            denom = active_masks_batch.sum().clamp(min=1e-6)
            value_loss = (value_loss * active_masks_batch).sum() / denom
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        """
        (share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,
         value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,
         adv_targ, available_actions_batch) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        # 兜底：若adv包含NaN/Inf，替换为0，避免对数概率或比率计算产生NaN
        adv_targ = torch.nan_to_num(adv_targ, nan=0.0, posinf=0.0, neginf=0.0)

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # 批量前向：评估log_prob/entropy与当前value
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch
        )

        # === actor update ===
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            denom = active_masks_batch.sum().clamp(min=1e-6)
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / denom
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # === critic update ===
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        # 局部副本用于计算均值方差，inactive位置标NaN
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan

        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)

        # 兜底：全NaN或std=0时，使用0均值、单位方差，避免产生NaN
        if np.isnan(mean_advantages):
            mean_advantages = 0.0
        if np.isnan(std_advantages) or std_advantages == 0.0:
            std_advantages = 1.0

        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {
            'value_loss': 0.0,
            'policy_loss': 0.0,
            'dist_entropy': 0.0,
            'actor_grad_norm': 0.0,
            'critic_grad_norm': 0.0,
            'ratio': 0.0
        }

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = \
                    self.ppo_update(sample, update_actor)

                train_info['value_loss'] += float(value_loss.item())
                train_info['policy_loss'] += float(policy_loss.item())
                train_info['dist_entropy'] += float(dist_entropy.item())
                train_info['actor_grad_norm'] += float(actor_grad_norm)
                train_info['critic_grad_norm'] += float(critic_grad_norm)
                # ratio 用 imp_weights 的均值（重要性采样比率）
                train_info['ratio'] += float(imp_weights.mean().item())

        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= max(1, num_updates)

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
