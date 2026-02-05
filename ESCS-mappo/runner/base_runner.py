# Filename: runner/base_runner.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from algorithms.onpolicy_utils.shared_buffer import SharedReplayBuffer
from gym.spaces import Tuple as GymTuple


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """

    def __init__(self, config):
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        # 优先用 envs.num_agents；没有就从 action_space 推断
        if hasattr(self.envs, "num_agents"):
            self.num_agents = int(self.envs.num_agents)
        else:
            act_sp = self.envs.action_space
            if isinstance(act_sp, GymTuple):
                self.num_agents = len(act_sp.spaces)
            elif isinstance(act_sp, (list, tuple)):
                self.num_agents = len(act_sp)
            else:
                # 单智能体兜底
                self.num_agents = 1
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']

        # parameters
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

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dirs
        self.model_dir = self.all_args.model_dir
        self.run_dir = config["run_dir"]

        self.log_dir = str(self.run_dir / 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.writter = SummaryWriter(self.log_dir)

        self.save_dir = str(self.run_dir / 'models')
        os.makedirs(self.save_dir, exist_ok=True)

        # ====== IMPORTS (适配当前策略栈路径) ======
        from algorithms.on_policy.r_mappo.algorithm.r_mappo import RMAPPO as TrainAlgo
        from algorithms.on_policy.r_mappo.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy

        # ====== 选择 centralized critic 的输入空间 ======
        if self.use_centralized_V:
            # 可选：使用局部观测代替全局状态
            if getattr(self.all_args, "use_obs_instead_of_state", False):
                share_observation_space = self.envs.observation_space[0]
            else:
                share_observation_space = self.envs.share_observation_space[0]
        else:
            share_observation_space = self.envs.observation_space[0]

        # ====== policy network ======
        self.policy = Policy(
            self.all_args,
            self.envs.observation_space[0],  # actor obs space（局部）
            share_observation_space,  # critic obs space（centralized or local）
            self.envs.action_space[0],  # MultiDiscrete([dir, hov, ds])
            device=self.device
        )

        # ====== restore (可选) ======
        if self.model_dir is not None:
            self.restore()

        # ====== algorithm (trainer) ======
        self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)

        # ====== buffer ======
        self.buffer = SharedReplayBuffer(
            self.all_args,
            self.num_agents,
            self.envs.observation_space[0],
            share_observation_space,
            self.envs.action_space[0]
        )

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1])
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        """Train policies with data in buffer."""
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), os.path.join(self.save_dir, "actor.pt"))
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), os.path.join(self.save_dir, "critic.pt"))

    def restore(self):
        """Restore policy's networks from a saved model."""
        actor_path = os.path.join(str(self.model_dir), 'actor.pt')
        critic_path = os.path.join(str(self.model_dir), 'critic.pt')

        if os.path.isfile(actor_path):
            policy_actor_state_dict = torch.load(actor_path, map_location=self.device)
            self.policy.actor.load_state_dict(policy_actor_state_dict, strict=False)

        if (not self.all_args.use_render) and os.path.isfile(critic_path):
            policy_critic_state_dict = torch.load(critic_path, map_location=self.device)
            self.policy.critic.load_state_dict(policy_critic_state_dict, strict=False)

        # === 打印权重加载结果（路径 + 简单签名）===
        def _weight_signature(module):
            try:
                s = 0.0
                n = 0
                for p in module.parameters():
                    s += p.abs().sum().item()
                    n += p.numel()
                return f"abs_sum={s:.3e}, numel={n}"
            except Exception:
                return "N/A"

        print(f"[LOAD] Actor path : {actor_path if os.path.isfile(actor_path) else 'NOT FOUND'}")
        print(f"[LOAD] Critic path: {critic_path if os.path.isfile(critic_path) else 'NOT FOUND'}")
        try:
            print(f"[WEIGHT] Actor  : {_weight_signature(self.policy.actor)}")
        except Exception as e:
            print(f"[WEIGHT] Actor  : signature error: {e}")
        try:
            print(f"[WEIGHT] Critic : {_weight_signature(self.policy.critic)}")
        except Exception as e:
            print(f"[WEIGHT] Critic : signature error: {e}")

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
