import numpy as np
import torch
import torch.nn as nn

from algorithms.onpolicy_utils.util import get_gard_norm, huber_loss, mse_loss
from algorithms.on_policy.utils.popart_hatrpo import PopArt
from algorithms.on_policy.utils.util import check
from algorithms.onpolicy_utils.valuenorm import ValueNorm


class HAPPO():
    """
    HAPPO Trainer（单-actor环境适配版）

    关键改动点：
    - 支持 Actor 返回【标量】或【逐头向量】的 action_log_probs；
      在此做 expand/broadcast，确保“逐头乘积”的 HAPPO 比率正确。
    - 若 buffer 未提供 factor_batch，则在此兜底为全 1（与 old_action_log_probs 同形）。
    - 保持 active_masks 的安全均值与 PopArt/ValueNorm 的有效样本更新。
    """

    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        # --------- hyper-params ---------
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        # --------- switches ---------
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        # --------- value normalizer ---------
        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    # ---------------- value loss ----------------
    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        values: V(s) predicted by current critic
        """
        if self._use_popart or self._use_valuenorm:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -self.clip_param, self.clip_param
            )
            target_norm = self.value_normalizer.normalize(return_batch)
            error_clipped = target_norm - value_pred_clipped
            error_original = target_norm - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -self.clip_param, self.clip_param
            )
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        value_loss = torch.max(value_loss_original, value_loss_clipped) if self._use_clipped_value_loss else value_loss_original

        if self._use_value_active_masks:
            # 只在有效样本上平均
            value_loss = (value_loss * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-6)
        else:
            value_loss = value_loss.mean()

        return value_loss

    # ---------------- one ppo update ----------------
    def ppo_update(self, sample, update_actor=True):
        """
        sample = (
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,
            value_preds_batch, return_batch, masks_batch, active_masks_batch,
            old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch
        )
        """
        (share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,
         value_preds_batch, return_batch, masks_batch, active_masks_batch,
         old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch) = sample

        # ---- to torch & device ----
        share_obs_batch = check(share_obs_batch).to(**self.tpdv)
        obs_batch = check(obs_batch).to(**self.tpdv)
        rnn_states_batch = check(rnn_states_batch).to(**self.tpdv)
        rnn_states_critic_batch = check(rnn_states_critic_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        actions_batch = check(actions_batch).to(self.device)  # MultiDiscrete -> long/int（check 会保留 int64）
        if available_actions_batch is not None:
            available_actions_batch = check(available_actions_batch).to(self.device)

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        # 兜底：factor_batch 若缺省则全 1（形状与 old_action_log_probs_batch 对齐）
        if factor_batch is None:
            factor_batch = torch.ones_like(old_action_log_probs_batch, dtype=self.tpdv["dtype"],
                                           device=self.tpdv["device"])
        else:
            factor_batch = check(factor_batch).to(**self.tpdv)

        # ---- 归一化器在线更新（只用有效样本）----
        if (self._use_popart or self._use_valuenorm) and self.value_normalizer is not None:
            with torch.no_grad():
                mask = (active_masks_batch > 0.5).squeeze(-1)
                if mask.any():
                    self.value_normalizer.update(return_batch[mask])

        # ---- 前向评估 ----
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch, obs_batch,
            rnn_states_batch, rnn_states_critic_batch,
            actions_batch, masks_batch,
            available_actions_batch, active_masks_batch
        )

        # ====== 逐头比率（兼容“标量 or 向量” action_log_probs）======
        # old_action_log_probs_batch: (B, H)；若 action_log_probs 是 (B,1)/(B,)
        # 则广播为 (B, H)；若已是 (B, H) 直接使用。
        if action_log_probs.shape[-1] != old_action_log_probs_batch.shape[-1]:
            action_log_probs = action_log_probs.expand_as(old_action_log_probs_batch)

        ratio_per_head = torch.exp(action_log_probs - old_action_log_probs_batch)  # (B, H)
        imp_weights = ratio_per_head.prod(dim=-1, keepdim=True)                    # (B, 1)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / (active_masks_batch.sum() + 1e-6)
        else:
            policy_action_loss = -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        # ---- actor update ----
        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
        self.policy.actor_optimizer.step()

        # ---- critic update ----
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    # ---------------- train loop over minibatches ----------------
    def train(self, buffer, update_actor=True):
        """
        buffer: SharedReplayBuffer-like
        需要提供：
          - returns, value_preds, active_masks（形状齐全）
          - feed_forward_generator / naive_recurrent_generator / recurrent_generator
          - data样本要包含 factor_batch（没有的话可在 buffer 里填 None，或直接不产出）
        """
        if self._use_popart:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        # 只在有效样本上做均值/方差
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_adv = np.nanmean(advantages_copy)
        std_adv = np.nanstd(advantages_copy)
        advantages = (advantages - mean_adv) / (std_adv + 1e-5)

        train_info = dict(value_loss=0, policy_loss=0, dist_entropy=0,
                          actor_grad_norm=0, critic_grad_norm=0, ratio=0)

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = \
                    self.ppo_update(sample, update_actor=update_actor)

                train_info['value_loss'] += float(value_loss.item())
                train_info['policy_loss'] += float(policy_loss.item())
                train_info['dist_entropy'] += float(dist_entropy.item())
                train_info['actor_grad_norm'] += float(actor_grad_norm)
                train_info['critic_grad_norm'] += float(critic_grad_norm)
                train_info['ratio'] += float(imp_weights.mean().item())

        num_updates = self.ppo_epoch * self.num_mini_batch
        num_updates = max(1, num_updates)
        for k in train_info:
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
