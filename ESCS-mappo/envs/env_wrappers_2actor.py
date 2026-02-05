# Filename: envs/env_wrappers_2actor.py

import numpy as np


class DummyVecEnv:
    """
    一个最小但健壮的同步向量环境封装：
    - 透传每个子环境的动作负载（支持 list/tuple/dict 或 np.ndarray(object=...)）
    - 聚合 obs/reward/done/info 到批量维度
    - 暴露 num_envs / num_agents / {observation,share_observation,action}_space
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        assert len(self.envs) > 0, "DummyVecEnv 需要至少一个子环境"
        env0 = self.envs[0]

        self.num_envs = len(self.envs)

        # —— 关键：向上层暴露 agent 数 —— #
        # 你的 DiscreteActionEnv 将 action_space/observation_space 定义为长度=1+K 的 list
        if hasattr(env0, "num_agents"):
            self.num_agents = int(env0.num_agents)
        elif isinstance(getattr(env0, "action_space", None), (list, tuple)):
            self.num_agents = len(env0.action_space)
        else:
            self.num_agents = 1  # 兜底

        # 直接复用单个环境的空间定义（所有子环境应一致）
        self.observation_space = env0.observation_space
        self.share_observation_space = getattr(env0, "share_observation_space", self.observation_space)
        self.action_space = env0.action_space

        self.actions = None

    # ------------- 矢量 API ------------- #
    def step(self, actions):
        """同步 step（兼容旧 API）"""
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        """
        保存调用方传入的批量动作。
        允许：
          - list/tuple，len = num_envs，每个元素是该 env 的动作负载（任意 Python 结构）
          - np.ndarray，shape[0] = num_envs；若 dtype=object，可异构
        """
        self.actions = actions

    def step_wait(self):
        """
        将批量动作逐 env 透传给子环境的 step。
        不对单个 env 的动作 payload 做任何“猜测性”转换，避免破坏异构动作结构。
        """
        assert self.actions is not None, "step_wait() 在未调用 step_async() 时被调用"

        # 为了同时支持 list/tuple/ndarray，这里只依赖“可迭代 + 第一维 = num_envs”的特性
        try:
            per_env_actions = list(self.actions)
            assert len(per_env_actions) == self.num_envs
        except Exception as e:
            raise ValueError(f"DummyVecEnv 期望 actions 可迭代且长度为 num_envs={self.num_envs}，当前类型/形状不兼容: {type(self.actions)}, err={e}")

        results = []
        for a, env in zip(per_env_actions, self.envs):
            # 透传给子环境；各子环境自行解析（支持 dict / list / np.ndarray(object=...) / 等）
            results.append(env.step(a))

        # 聚合结果：obs/rews/dones -> ndarray；infos -> list
        obs_list, rews_list, dones_list, infos_list = zip(*results)
        obs = np.asarray(obs_list, dtype=np.float32)
        rews = np.asarray(rews_list, dtype=np.float32)
        dones = np.asarray(dones_list)  # bool/float 皆可
        infos = list(infos_list)

        # reset 已完成的 env
        for i, done_i in enumerate(dones):
            d = np.asarray(done_i)
            # 允许标量、(n_agent,) 或 (n_agent,1) 形状
            is_all_done = bool(np.all(d))
            if is_all_done:
                obs[i] = self.envs[i].reset()

        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        """重置所有子环境并堆叠 obs -> [num_envs, num_agents, obs_dim]"""
        obs = [env.reset() for env in self.envs]
        return np.asarray(obs, dtype=np.float32)

    def close(self):
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()

    # ------------- 可行动作掩码（可选）------------- #
    def get_available_actions(self):
        """
        聚合各子环境返回的动作掩码：
          - 若任一子环境返回 None -> 整体返回 None（上层自行处理）
          - 否则堆叠为 [num_envs, num_agents, width]
        """
        masks = []
        for env in self.envs:
            m = None
            if hasattr(env, "get_available_actions"):
                m = env.get_available_actions()
            elif hasattr(env, "action_masks"):
                m = env.action_masks()
            if m is None:
                return None
            masks.append(np.asarray(m, dtype=np.float32))
        return np.stack(masks, axis=0)

    # 别名适配
    def action_masks(self):
        return self.get_available_actions()

    # ------------- 渲染 ------------- #
    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        if mode == "human":
            for env in self.envs:
                env.render(mode=mode)
            return None
        raise NotImplementedError
