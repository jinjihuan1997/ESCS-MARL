# =====================================
# Filepath: envs/env_discrete.py
# -*- coding: utf-8 -*-
# =====================================

import gym
import numpy as np
from gym import spaces
from typing import Optional, List, Union, Tuple, Dict
from envs.env_core import EnvCore


class DiscreteActionEnv(gym.Env):
    """
    多智能体离散动作环境封装（只暴露 K 个 SD 为可控 agent；BS 由环境内部处理）。

    对外（给训练器）的约定：
      - agent 索引：0..K-1 = SD-0..SD-(K-1)
      - 动作空间（同构）：MultiDiscrete([n_dir, n_fly, M+1])
      - 观测：统一到固定长度 obs_dim；可选在末尾拼接 role one-hot（此处 SD 固定为 [0,1]）
      - reset/step 形状：
          obs    -> (K, obs_dim)     float32
          reward -> (K, 1)           float32
          done   -> (K, 1)           bool
          info   -> 长度 K 的 list[dict]
      - 可行动作掩码：get_action_masks()/action_masks()/get_available_actions()
          返回 (K, sum([n_dir, n_fly, M+1])) 的扁平向量（逐头拼接）
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, seed=None, debug=False, comm_mode="SC", add_type_flag=True, local_only=False):
        super().__init__()
        self.core = EnvCore(seed=seed, debug=debug, comm_mode=comm_mode)
        self.debug = bool(debug)
        self.add_type_flag = bool(add_type_flag)
        self.local_only = bool(local_only)

        # ---------- 从底层探测一次，建立维度 ----------
        obs_all = self.core.reset()  # [obs_bs, obs_sd0, obs_sd1, ...]
        self.K = int(self.core.K)
        self.M = int(self.core.M)
        self.num_agents = self.K  # 供上层读取

        # 观测统一长度：以所有（BS+SD）原始观测的最大长度为准
        raw_lens = [int(np.asarray(o, dtype=np.float32).size) for o in obs_all]
        self.role_dim = 2 if self.add_type_flag else 0
        self.base_obs_dim = int(max(raw_lens))
        self.obs_dim = self.base_obs_dim + self.role_dim

        # ---- 动作头（SD；对外只暴露 SD 的 MultiDiscrete）----
        self.sd_nvec = [int(self.core.n_dir), int(self.core.n_fly), int(self.core.M) + 1]
        self.sd_sum_nvec = int(np.sum(self.sd_nvec))

        # BS 的动作头仅用于内部调用 core.step（对训练隐藏）
        self.bs_nvec = [self.M] * int(self.core.L_BS_ENC) + [2] * int(self.core.L_BS_TRANS)

        # ---- 对外动作/观测空间（K 个同构 SD）----
        # 注意：为避免引用同一实例导致潜在副作用，这里复制生成
        self.action_space: List[spaces.Space] = [spaces.MultiDiscrete(np.asarray(self.sd_nvec, dtype=np.int64))
                                                 for _ in range(self.K)]

        self.observation_space: List[spaces.Space] = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for _ in range(self.K)
        ]

        # ---- centralized critic 的共享空间 ----
        g = self._read_global_state_vector().astype(np.float32, copy=False)
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(g.size,), dtype=np.float32)
            for _ in range(self.K)
        ]
        self._last_share = g.copy()

        # —— 缓存上一步 infos（供掩码逻辑）——
        self._last_infos: Optional[List[Dict]] = None

        # —— 为了避免 __init__ 与外部 reset() 触发两次 core.reset()，做一次缓存 —— #
        self._cached_reset_obs = self._augment_sd_only(obs_all)  # (K, obs_dim)

    # ====================== Gym 接口 ====================== #
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.seed(seed)

        if self._cached_reset_obs is not None:
            obs_out = self._cached_reset_obs
            self._cached_reset_obs = None
            self._last_share = self._read_global_state_vector()  # 改这里
            self._last_infos = None
            return obs_out

        obs_all = self.core.reset()
        obs_out = self._augment_sd_only(obs_all)
        self._last_share = self._read_global_state_vector()  # 改这里
        self._last_infos = None
        return obs_out

    # === 修改点 4：step() 里同样直接刷新全局向量 ===
    def step(self, actions: Union[List, Tuple, Dict]):
        A_sd = self._normalize_sd_actions(actions)
        A_bs = self._default_bs_action()

        obs_all, rews_all, dones_all, infos_all = self.core.step({'bs': A_bs, 'sd': A_sd})
        self._last_infos = infos_all

        obs_out = self._augment_sd_only(obs_all)
        self._last_share = self._read_global_state_vector()  # 改这里

        rews_sd = np.asarray(rews_all[1:], dtype=np.float32).reshape(self.K, 1)
        dones_sd = np.asarray(dones_all[1:], dtype=np.bool_).reshape(self.K, 1)

        infos_sd = []
        for i in range(self.K):
            info_i = infos_all[i + 1] if isinstance(infos_all[i + 1], dict) else {}
            info_i = dict(info_i) if isinstance(info_i, dict) else {}
            info_i["share_obs"] = self._last_share
            infos_sd.append(info_i)

        return obs_out, rews_sd, dones_sd, infos_sd

    def render(self, mode: str = "human"):
        if mode == "rgb_array":
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return None

    def close(self):
        pass

    # ====================== 工具方法 ====================== #
    def _default_bs_action(self) -> np.ndarray:
        """给 BS 一个缺省的“全 0”动作（enc 不投票，tx 全关）。"""
        need_bs = len(self.bs_nvec)
        return np.zeros((need_bs,), dtype=np.int64)

    def _normalize_sd_actions(self, actions: Union[List, Tuple, np.ndarray]) -> List[np.ndarray]:
        """
        将外部输入动作规整为长度=3 的 (dir, fly, ds_idx) 向量列表，长度不足补 0，多余截断。
        兼容多种输入形状：
          - np.ndarray: (K,3) | (1,K,3) | (K,) 且 dtype=object | (K*3,)
          - list/tuple: len=K 且每个元素是长度3；或 len=1 且第0个是 (K,3) 的 ndarray
        """
        need_sd = 3
        K = self.K

        def _pad_or_clip(v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, dtype=np.int64).reshape(-1)
            if v.size < need_sd:
                v = np.pad(v, (0, need_sd - v.size), mode="constant")
            elif v.size > need_sd:
                v = v[:need_sd]
            return v

        # ---------- numpy.ndarray 路径 ----------
        if isinstance(actions, np.ndarray):
            a = actions

            # squeeze 掉多余的 batch/env 维度，如 (1,K,3) -> (K,3)
            if a.ndim == 3 and a.shape[0] == 1 and a.shape[1] == K and a.shape[2] == need_sd:
                a = a[0]

            # 典型的 (K,3)
            if a.ndim == 2 and a.shape == (K, need_sd):
                return [_pad_or_clip(a[i]) for i in range(K)]

            # (K,) 且 dtype=object（元素各自是动作向量）
            if a.ndim == 1 and a.shape[0] == K and a.dtype == object:
                return [_pad_or_clip(a[i]) for i in range(K)]

            # 扁平 (K*3,)
            if a.ndim == 1 and a.size == K * need_sd:
                a2 = a.reshape(K, need_sd)
                return [_pad_or_clip(a2[i]) for i in range(K)]

            # 其他情形：尝试自动 reshape(-1,3)
            try:
                a2 = a.reshape(-1, need_sd)
                if a2.shape[0] == K:
                    return [_pad_or_clip(a2[i]) for i in range(K)]
            except Exception:
                pass

            raise ValueError(f"无法解析 SD 动作的形状: {a.shape}，期望 (K,3)/(1,K,3)/list(len=K)。")

        # ---------- list/tuple 路径 ----------
        if isinstance(actions, (list, tuple)):
            # 形如 [np.ndarray(K,3)]
            if len(actions) == 1 and isinstance(actions[0], np.ndarray):
                return self._normalize_sd_actions(actions[0])

            if len(actions) == K:
                return [_pad_or_clip(a) for a in actions]

            raise ValueError(f"SD 动作应为长度 K({K}) 的 list/tuple，当前 len={len(actions)}。")

        # ---------- 其他类型 ----------
        raise ValueError(f"不支持的 SD 动作类型: {type(actions)}")

    def _augment_sd_only(self, obs_all: List[np.ndarray]) -> np.ndarray:
        """
        只返回 SD 的观测，并统一到 (K, obs_dim)：
          - 右侧 0 填充/截断到 base_obs_dim
          - 末尾拼接 role one-hot（SD 固定 [0,1]）
        """
        out = []
        base = self.base_obs_dim
        for i in range(self.K):
            o = np.asarray(obs_all[i + 1], dtype=np.float32).reshape(-1)  # 跳过 BS
            v = np.zeros((self.obs_dim,), dtype=np.float32)
            v[:min(o.size, base)] = o[:base]
            if self.add_type_flag:
                v[-2:] = (0.0, 1.0)  # SD
            out.append(v)
        return np.stack(out).astype(np.float32)

    def _compute_share_obs(self, obs_all: List[np.ndarray]) -> np.ndarray:
        """
        centralized critic 输入：
          - 优先 core.get_global_state()
          - 否则拼接（BS+SD）的增强观测
        """
        if hasattr(self.core, "get_global_state") and callable(getattr(self.core, "get_global_state")):
            try:
                g = self.core.get_global_state()
                if isinstance(g, dict) and "vector" in g:
                    return np.asarray(g["vector"], dtype=np.float32).reshape(-1)
                return np.asarray(g, dtype=np.float32).reshape(-1)
            except Exception:
                pass

        # 没有全局状态则拼接（包含 BS+SD）
        aug = []
        base = self.base_obs_dim
        for i, o in enumerate(obs_all):
            o = np.asarray(o, dtype=np.float32).reshape(-1)
            v = np.zeros((self.obs_dim,), dtype=np.float32)
            v[:min(o.size, base)] = o[:base]
            if self.add_type_flag:
                v[-2:] = (1.0, 0.0) if i == 0 else (0.0, 1.0)
            aug.append(v)
        return np.concatenate(aug, axis=0).astype(np.float32)

    def _read_global_state_vector(self) -> np.ndarray:
        g = self.core.get_global_state()  # EnvCore 已实现的全局状态
        if isinstance(g, dict) and "vector" in g:
            g = g["vector"]
        return np.asarray(g, dtype=np.float32).reshape(-1)

    # === 修改点 5：get_share_obs() 直接回 EnvCore 全局向量给每个 agent ===
    def get_share_obs(self) -> List[np.ndarray]:
        g = self._last_share.astype(np.float32, copy=True)
        return [g for _ in range(self.K)]

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            return [None]
        np.random.seed(seed)
        if hasattr(self.core, "rng") and self.core.rng is not None:
            self.core.rng.seed(seed)
        return [seed]

    # ---------------- 动作掩码（MultiDiscrete 扁平向量） ---------------- #
    def get_action_masks(self) -> np.ndarray:
        """
        返回形状 (K, sum(sd_nvec)) 的 float32 掩码向量（按 [dir | fly | ds] 扁平拼接）：
          - dir：全 1
          - fly：首槽可强制仅允许 index=0（悬停）；有有效连接时限制最大飞行索引以满足“最小悬停比例”，否则给出偏向“全飞”的许可
          - ds ：优先用上一时刻 info['mask_valid_ds']（长度 M+1，含 skip=M）；拿不到则只允许 skip
                 无论如何始终允许 skip=M
        """
        K = self.K
        n_dir = int(self.core.n_dir)
        n_fly = int(self.core.n_fly)
        n_pick = int(self.core.M) + 1

        masks = np.zeros((K, n_dir + n_fly + n_pick), dtype=np.float32)
        last_infos = self._last_infos if isinstance(self._last_infos, (list, tuple)) else None

        first_slot = bool(getattr(self.core, "first_slot_force_full_hover", False)) and int(getattr(self.core, "t", 1)) <= 1
        cap = int(np.floor((1.0 - float(0)) * max(n_fly - 1, 0)))
        cap = int(np.clip(cap, 0, n_fly - 1))

        for k in range(K):
            # dir
            off = 0
            masks[k, off:off + n_dir] = 1.0
            off += n_dir

            # fly
            if first_slot:
                fly_mask = np.zeros(n_fly, dtype=np.float32)
                fly_mask[0] = 1.0
            else:
                fly_mask = None  # 等会儿根据 valid ds 决定

            # ds
            pick_mask = np.zeros(n_pick, dtype=np.float32)

            # 从上一步 infos 取这个 SD 的 valid 掩码
            valid_mask = None
            if last_infos is not None and len(last_infos) == (1 + K):
                info_k = last_infos[k + 1]
                if isinstance(info_k, dict) and "mask_valid_ds" in info_k:
                    vm = np.asarray(info_k["mask_valid_ds"]).astype(np.int32).reshape(-1)
                    if vm.size == n_pick:
                        valid_mask = vm

            if valid_mask is None:
                pick_mask[-1] = 1.0  # 只允许 skip
                if fly_mask is None:
                    fly_mask = np.zeros(n_fly, dtype=np.float32)
                    allow = sorted({min(2, n_fly - 1), n_fly - 1})
                    if len(allow) > 0:
                        fly_mask[allow] = 1.0
            else:
                pick_mask = valid_mask.astype(np.float32, copy=True)
                pick_mask[-1] = 1.0  # 始终允许 skip
                if np.any(valid_mask[:-1] > 0):
                    # 有有效连接：限制最大飞行索引，满足“最小悬停比例”
                    fly_mask = np.zeros(n_fly, dtype=np.float32)
                    fly_mask[:cap + 1] = 1.0
                else:
                    fly_mask = np.zeros(n_fly, dtype=np.float32)
                    allow = sorted({min(2, n_fly - 1), n_fly - 1})
                    if len(allow) > 0:
                        fly_mask[allow] = 1.0

            masks[k, off:off + n_fly] = fly_mask
            off += n_fly
            masks[k, off:off + n_pick] = pick_mask

        return masks

    # 别名：兼容上层 Runner 的自动查询
    def action_masks(self):
        return self.get_action_masks()

    def get_available_actions(self):
        return self.get_action_masks()

    # ---------------- 便捷属性/方法（供算法配置读取） ---------------- #
    @property
    def agent_names(self) -> List[str]:
        return [f"SD-{i}" for i in range(self.K)]

    @property
    def action_nvec_per_agent(self) -> List[np.ndarray]:
        sd = np.asarray(self.sd_nvec, dtype=np.int64)
        return [sd for _ in range(self.K)]

    def get_algo_config(self) -> Dict[str, Union[int, List[int]]]:
        return {
            "role_dim": int(self.role_dim),
            "bs_nvec": list(self.bs_nvec),        # 仅供外部参考
            "sd_nvec": list(self.sd_nvec),
            "bs_enc_heads": int(self.core.L_BS_ENC),
            "bs_tx_bits": int(self.core.L_BS_TRANS),
            "sd_sc_heads": 1,
            "sd_ds_pick_semantics": getattr(self.core, "sd_ds_pick_semantics", "global"),
        }

    # 仅用于内部：让 reset() 的缓存能取到最近一次全量 obs
    def get_last_obs_all(self) -> List[np.ndarray]:
        if hasattr(self.core, "get_last_obs_all") and callable(getattr(self.core, "get_last_obs_all")):
            try:
                return self.core.get_last_obs_all()
            except Exception:
                pass
        # 取不到就重置一次（尽量避免）
        return self.core.reset()
