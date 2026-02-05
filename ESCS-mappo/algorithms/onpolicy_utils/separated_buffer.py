# Filename: algorithms/onpolicy_utils/separated_buffer.py

import torch
import numpy as np
from algorithms.onpolicy_utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 0, 2).reshape(-1, *x.shape[2:])


class SeparatedReplayBuffer(object):
    """
    Buffer for storing trajectories experienced by a single agent interacting with the environment.
    Each agent has its own independent buffer.
    """

    def __init__(self, args, obs_space, share_obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits

        # 获取观察和动作空间的形状
        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        # 处理动作空间形状
        act_shape = get_shape_from_act_space(act_space)
        if isinstance(act_shape, (list, tuple)):
            # 如果是tuple或list，需要展平或处理
            if len(act_shape) == 1:
                act_shape = act_shape[0]
            # 对于MultiDiscrete，act_shape可能是动作维度

        # 创建缓冲区数组 - 注意形状的正确构建
        if share_obs_shape is not None:
            if isinstance(share_obs_shape, (list, tuple)):
                self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, *share_obs_shape),
                                          dtype=np.float32)
            else:
                self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, share_obs_shape),
                                          dtype=np.float32)
        else:
            self.share_obs = None

        if isinstance(obs_shape, (list, tuple)):
            self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, *obs_shape), dtype=np.float32)
        else:
            self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.returns = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)

        # 处理动作形状
        if isinstance(act_space.__class__.__name__, str):
            action_type = act_space.__class__.__name__
        else:
            action_type = act_space.__class__.__name__

        if action_type == "MultiDiscrete":
            # 对于MultiDiscrete，动作是整数向量
            if hasattr(act_space, 'nvec'):
                action_dim = len(act_space.nvec)
            elif hasattr(act_space, 'shape'):
                action_dim = act_space.shape[0] if isinstance(act_space.shape, tuple) else act_space.shape
            else:
                action_dim = act_shape if isinstance(act_shape, int) else act_shape[0]
            self.actions = np.zeros((self.episode_length, self.n_rollout_threads, action_dim), dtype=np.int64)
            self.available_actions = None
        elif action_type == "Discrete":
            # 对于Discrete，动作是单个整数
            self.actions = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.int64)
            self.available_actions = None
        elif action_type == "Box":
            # 对于Box，动作是连续值
            if isinstance(act_shape, (list, tuple)):
                self.actions = np.zeros((self.episode_length, self.n_rollout_threads, *act_shape), dtype=np.float32)
            else:
                self.actions = np.zeros((self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)
            self.available_actions = None
        else:
            # 默认处理
            if isinstance(act_shape, int):
                self.actions = np.zeros((self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)
            elif isinstance(act_shape, (list, tuple)):
                self.actions = np.zeros((self.episode_length, self.n_rollout_threads, *act_shape), dtype=np.float32)
            else:
                raise NotImplementedError(f"Unsupported action space type: {action_type}")
            self.available_actions = None

        self.action_log_probs = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        """
        Insert data into the buffer.
        """
        if self.share_obs is not None:
            self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index for next episode."""
        if self.share_obs is not None:
            self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        """
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(
                            self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[
                            step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * \
                                             value_normalizer.denormalize(self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * self.value_preds[
                                                 step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(
                            self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[
                            step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + \
                                             self.rewards[step]
                    else:
                        self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + \
                                             self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Generator that returns mini-batches for PPO training.
        """
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                f"PPO requires the batch size ({batch_size}) "
                f"to be greater than or equal to the number of mini batches ({num_mini_batch})."
            )
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        if self.share_obs is not None:
            share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        else:
            share_obs = None
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[2:])
        else:
            available_actions = None
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, 1)
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            if share_obs is not None:
                share_obs_batch = share_obs[indices]
            else:
                share_obs_batch = None
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                adv_targ, available_actions_batch

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        """
        Generator for naive recurrent PPO training.
        """
        n_rollout_threads = self.rewards.shape[1]
        assert n_rollout_threads >= num_mini_batch, (
            f"PPO requires the batch size ({n_rollout_threads}) "
            f"to be greater than or equal to the number of mini batches ({num_mini_batch})."
        )
        num_envs_per_batch = n_rollout_threads // num_mini_batch
        perm = torch.randperm(n_rollout_threads).numpy()

        for start_ind in range(0, n_rollout_threads, num_envs_per_batch):
            if self.share_obs is not None:
                share_obs_batch = []
            else:
                share_obs_batch = None
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                if self.share_obs is not None:
                    share_obs_batch.append(self.share_obs[:-1, ind])
                obs_batch.append(self.obs[:-1, ind])
                rnn_states_batch.append(self.rnn_states[0:1, ind])
                rnn_states_critic_batch.append(self.rnn_states_critic[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(self.available_actions[:-1, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                active_masks_batch.append(self.active_masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.episode_length, num_envs_per_batch
            if share_obs_batch is not None:
                share_obs_batch = np.stack(share_obs_batch, 1).reshape(T * N, -1)
            obs_batch = np.stack(obs_batch, 1).reshape(T * N, -1)
            actions_batch = np.stack(actions_batch, 1).reshape(T * N, -1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1).reshape(T * N, -1)
            else:
                available_actions_batch = None
            value_preds_batch = np.stack(value_preds_batch, 1).reshape(T * N, -1)
            return_batch = np.stack(return_batch, 1).reshape(T * N, -1)
            masks_batch = np.stack(masks_batch, 1).reshape(T * N, -1)
            active_masks_batch = np.stack(active_masks_batch, 1).reshape(T * N, -1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1).reshape(T * N, -1)
            adv_targ = np.stack(adv_targ, 1).reshape(T * N, -1)

            rnn_states_batch = np.stack(rnn_states_batch, 1).reshape(N, *self.rnn_states.shape[2:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch, 1).reshape(N, *self.rnn_states_critic.shape[2:])

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                adv_targ, available_actions_batch

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        """
        Generator for recurrent PPO training with data chunks.
        """
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch

        assert episode_length % data_chunk_length == 0, (
            f"Episode length ({episode_length}) must be divisible by data chunk length ({data_chunk_length})."
        )

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        if self.share_obs is not None:
            share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        else:
            share_obs = None
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[2:])
        else:
            available_actions = None
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, 1)
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            if share_obs is not None:
                share_obs_batch = []
            else:
                share_obs_batch = None
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:
                ind = index * data_chunk_length
                if share_obs is not None:
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
            if share_obs_batch is not None and len(share_obs_batch) > 0:
                share_obs_batch = np.stack(share_obs_batch).reshape(N * L, -1)
            else:
                share_obs_batch = None
            obs_batch = np.stack(obs_batch).reshape(N * L, -1)
            actions_batch = np.stack(actions_batch).reshape(N * L, -1)
            if available_actions is not None and len(available_actions_batch) > 0:
                available_actions_batch = np.stack(available_actions_batch).reshape(N * L, -1)
            else:
                available_actions_batch = None
            value_preds_batch = np.stack(value_preds_batch).reshape(N * L, -1)
            return_batch = np.stack(return_batch).reshape(N * L, -1)
            masks_batch = np.stack(masks_batch).reshape(N * L, -1)
            active_masks_batch = np.stack(active_masks_batch).reshape(N * L, -1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch).reshape(N * L, -1)
            adv_targ = np.stack(adv_targ).reshape(N * L, -1)
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[2:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[2:])

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                adv_targ, available_actions_batch