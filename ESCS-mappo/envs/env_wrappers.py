# Filename: envs/env_wrappers.py

import numpy as np


# single env
class DummyVecEnv():
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.num_envs = len(env_fns)
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space
        self.actions = None

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        # 1) 同步调用每个子环境
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        # 2) 拆包，但只对 obs/rews/dones 转 numpy，infos 保留为列表
        obs_list, rews_list, dones_list, infos_list = zip(*results)
        obs = np.array(obs_list)
        rews = np.array(rews_list)
        dones = np.array(dones_list)
        infos = list(infos_list)  # 保留成 [ [agent0_info, agent1_info, …], … ]

        for i, done in enumerate(dones):
            # done 可能是标量或 shape=(K,)
            if np.asarray(done).ndim == 0:
                if bool(done):
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i] = self.envs[i].reset()

        self.actions = None
        return obs, rews, dones, infos

    def get_available_actions(self):
        """聚合各子环境的 action masks -> [n_env, n_agent, sum(nvec)]"""
        masks = []
        for env in self.envs:
            if hasattr(env, "get_available_actions"):
                m = env.get_available_actions()  # 期望 (K, sum(nvec))
            elif hasattr(env, "action_masks"):
                m = env.action_masks()
            else:
                m = None
            if m is None:
                return None
            masks.append(np.asarray(m, dtype=np.float32))
        return np.stack(masks, axis=0)  # [n_env, K, sum(nvec)]

    # 兼容别名
    def action_masks(self):
        return self.get_available_actions()

    def reset(self):
        obs = [env.reset() for env in self.envs]  # [env_num, agent_num, obs_dim]
        return np.asarray(obs, dtype=np.float32)  # 现在每个env返回的是 [agent, obs_dim]，可安全堆叠

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError
