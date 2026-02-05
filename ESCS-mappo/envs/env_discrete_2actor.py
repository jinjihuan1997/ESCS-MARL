# =====================================
# Filepath: envs/env_discrete_2actor.py
# -*- coding: utf-8 -*-
# =====================================

import gym
import numpy as np
from gym import spaces
from typing import Optional, List, Union, Tuple, Dict
from envs.env_core_2actor import EnvCore


class DiscreteActionEnv(gym.Env):
    """
    多智能体环境封装（公开 1 个 BS + K 个 SD 为可控 agent）。

    约定：
      - agent 索引：0=BS，1..K=SD-0..SD-(K-1)
      - 动作空间：
          * BS: Box(shape=(M,), low=0.0, high=1.0) —— 连续“传输权重”向量，仅用于 Tr->SD 分配（整数化由核心完成）。
                编码投票仍然在核心内部按“轮询”实现，不作为外部动作输入。
          * SD: MultiDiscrete([n_dir, n_fly, M+1])
      - 观测：统一到固定长度 obs_dim；末尾拼接 role one-hot（BS=[1,0], SD=[0,1]）
      - reset/step 形状：
          obs    -> (1+K, obs_dim)     float32
          reward -> (1+K, 1)           float32（共享奖励时会广播）
          done   -> (1+K, 1)           bool
          info   -> 长度 (1+K) 的 list[dict]（包含 share_obs 等）
      - 可行动作掩码：仅对 SD 生效（BS 连续动作不需要 mask）
          返回 (1+K, sd_sum_nvec) 的扁平向量；第 0 行(BS)恒为 0，后 K 行为 SD 掩码。
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, seed=None, debug=False, comm_mode="SC", add_type_flag=True, local_only=False):
        super().__init__()
        self.core = EnvCore(seed=seed, debug=debug, comm_mode=comm_mode)
        self.debug = bool(debug)
        self.add_type_flag = bool(add_type_flag)
        self.local_only = bool(local_only)

        # ---------- 探测尺寸 ----------
        obs_all = self.core.reset()  # [obs_bs, obs_sd0, obs_sd1, ...]
        self.K = int(self.core.K)
        self.M = int(self.core.M)

        # 现在对外暴露 (1 + K) 个 agent（第0个是BS）
        self.num_agents = 1 + self.K

        # 观测统一长度
        raw_lens = [int(np.asarray(o, dtype=np.float32).size) for o in obs_all]
        self.role_dim = 2 if self.add_type_flag else 0
        self.base_obs_dim = int(max(raw_lens))
        self.obs_dim = self.base_obs_dim + self.role_dim

        # ---- 动作头（SD）----
        self.sd_nvec = [int(self.core.n_dir), int(self.core.n_fly), int(self.core.M) + 1]
        self.sd_sum_nvec = int(np.sum(self.sd_nvec))

        # ---- BS 原离散动作头（保留为兼容字段/日志用；不再用于 action_space）----
        self.bs_nvec = [self.M] * int(self.core.L_BS_ENC) + [2] * int(self.core.L_BS_TRANS)
        self.bs_sum_nvec = int(np.sum(self.bs_nvec))

        # ---- 对外动作/观测空间（第0个BS=连续；其后K个SD=离散）----
        self.bs_weight_shape = (self.M,)
        self.bs_action_is_continuous = True

        self.action_space: List[spaces.Space] = [
            spaces.Box(low=0.0, high=1.0, shape=self.bs_weight_shape, dtype=np.float32)  # BS 连续权重
        ] + [
            spaces.MultiDiscrete(np.asarray(self.sd_nvec, dtype=np.int64))  # SD-i
            for _ in range(self.K)
        ]

        self.observation_space: List[spaces.Space] = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]

        # centralized critic 的共享空间 —— 直接读取 EnvCore 的全局向量（有稳健回退）
        g = self._read_global_state_vector(obs_all_fallback=obs_all).astype(np.float32, copy=False)
        self.share_observation_space: List[spaces.Space] = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(g.size,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]
        self._last_share = g.copy()

        # —— 缓存上一步 infos（供掩码逻辑）——
        self._last_infos: Optional[List[Dict]] = None

        # —— reset 缓存，避免双 reset —— #
        self._cached_reset_obs = self._augment_all_agents(obs_all)  # (1+K, obs_dim)

        # ---- 掩码宽度（仅 SD 需要）----
        self.mask_width = int(self.sd_sum_nvec)

    # ====================== Gym 接口 ====================== #
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.seed(seed)

        if self._cached_reset_obs is not None:
            obs_out = self._cached_reset_obs
            self._cached_reset_obs = None
            # 直接从 EnvCore 读取全局向量（若失败回退到最近一次全量观测拼接）
            self._last_share = self._read_global_state_vector()
            self._last_infos = None
            return obs_out

        obs_all = self.core.reset()
        obs_out = self._augment_all_agents(obs_all)
        self._last_share = self._read_global_state_vector(obs_all_fallback=obs_all)
        self._last_infos = None
        return obs_out

    def step(self, actions: Union[List, Tuple, Dict, np.ndarray]):
        """
        输入（对外）：
          actions: 长度 = 1+K；
            - 第0个：BS 的 Box(M,) 连续权重（>=0）；若给了老式 MultiDiscrete 也会兼容（权重置为0）
            - 其后 K 个：SD 的 MultiDiscrete（[dir, fly, ds]）
        兼容旧接口：
            - 若只给 K 个 SD 动作，则 BS 权重自动置 0。
            - 也支持 dict 形式：{'bs': ..., 'sd': ..., 'bs_tr_weights': ...}
        """
        # 解析输入
        if isinstance(actions, dict):
            # 允许显式给连续权重
            if "bs_tr_weights" in actions:
                bs_tr_weights = self._normalize_bs_weights(actions["bs_tr_weights"])
            else:
                bs_tr_weights = np.zeros((self.M,), dtype=np.float32)

            # SD 动作
            if "sd" not in actions:
                raise ValueError("dict 形式必须包含键 'sd'（长度 K 的 SD 动作列表/数组）")
            A_sd = self._normalize_sd_actions(actions["sd"])

            # 兼容旧 'bs'（MultiDiscrete），核心内部编码投票仍轮询，这里仅占位
            if "bs" in actions:
                A_bs = self._pad_or_clip_int(actions["bs"], need=len(self.bs_nvec))
            else:
                A_bs = self._default_bs_action()

        else:
            # list/tuple/object-ndarray 路径
            try:
                A_bs, A_sd, bs_tr_weights = self._normalize_joint_actions(actions)
            except Exception:
                # 仅 SD 的老接口
                A_sd = self._normalize_sd_actions(actions)
                A_bs = self._default_bs_action()
                bs_tr_weights = np.zeros((self.M,), dtype=np.float32)

        # 交给核心：同时传入 BS 连续权重
        obs_all, rews_all, dones_all, infos_all = self.core.step({
            'bs': A_bs,                      # 兼容字段（编码投票在核心内部轮询）
            'sd': A_sd,
            'bs_tr_weights': bs_tr_weights,  # 关键：连续传输权重
        })

        # 缓存 infos（供下一步掩码）
        self._last_infos = infos_all

        # 观测
        obs_out = self._augment_all_agents(obs_all)
        # === 这里直接刷新全局向量 ===
        self._last_share = self._read_global_state_vector(obs_all_fallback=obs_all)

        # 奖励/终止
        try:
            rews = np.asarray(rews_all, dtype=np.float32).reshape(self.num_agents, 1)
        except Exception:
            r = float(rews_all) if np.isscalar(rews_all) else float(np.mean(rews_all))
            rews = np.full((self.num_agents, 1), r, dtype=np.float32)

        try:
            dones = np.asarray(dones_all, dtype=np.bool_).reshape(self.num_agents, 1)
        except Exception:
            d = bool(dones_all) if np.isscalar(dones_all) else bool(np.any(dones_all))
            dones = np.full((self.num_agents, 1), d, dtype=np.bool_)

        # info：每个 info 附加 share_obs
        infos = []
        if isinstance(infos_all, (list, tuple)) and len(infos_all) >= 1:
            src = list(infos_all)
            if len(src) < self.num_agents:
                src = ([src[0]] * self.num_agents) if len(src) == 1 else (src + [{}] * (self.num_agents - len(src)))
        else:
            src = [{} for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            d = dict(src[i]) if isinstance(src[i], dict) else {}
            d["share_obs"] = self._last_share
            infos.append(d)

        return obs_out, rews, dones, infos

    def render(self, mode: str = "human"):
        if mode == "rgb_array":
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return None

    def close(self):
        pass

    # ====================== 工具方法 ====================== #
    def _default_bs_action(self) -> np.ndarray:
        """给 BS 一个缺省的“全 0”离散向量（兼容字段；目前核心对编码投票采用轮询）。"""
        need_bs = len(self.bs_nvec)
        return np.zeros((need_bs,), dtype=np.int64)

    def _normalize_bs_weights(self, a0):
        w = np.asarray(a0, dtype=np.float32).reshape(-1)
        if w.size < self.M:
            w = np.pad(w, (0, self.M - w.size), mode="constant")
        elif w.size > self.M:
            w = w[:self.M]
        # ACT 已输出 (0,1)，这里再稳一下：夹到 [0,1]
        return np.clip(w, 0.0, 1.0).astype(np.float32)

    def _pad_or_clip_int(self, v: Union[np.ndarray, List, Tuple], need: int) -> np.ndarray:
        """把输入转成 int64 长向量并裁剪/补齐到 need 长度。"""
        arr = np.asarray(v, dtype=np.int64).reshape(-1)
        if arr.size < need:
            arr = np.pad(arr, (0, need - arr.size), mode="constant")
        elif arr.size > need:
            arr = arr[:need]
        return arr

    def _normalize_sd_actions(self, actions: Union[List, Tuple, np.ndarray]) -> List[np.ndarray]:
        """
        规整 SD 动作为长度=3 的 (dir, fly, ds_idx) 向量列表，长度不足补 0，多余截断。
        兼容：
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

        if isinstance(actions, np.ndarray):
            a = actions
            if a.ndim == 3 and a.shape[0] == 1 and a.shape[1] == K and a.shape[2] == need_sd:
                a = a[0]
            if a.ndim == 2 and a.shape == (K, need_sd):
                return [_pad_or_clip(a[i]) for i in range(K)]
            if a.ndim == 1 and a.shape[0] == K and a.dtype == object:
                return [_pad_or_clip(a[i]) for i in range(K)]
            if a.ndim == 1 and a.size == K * need_sd:
                a2 = a.reshape(K, need_sd)
                return [_pad_or_clip(a2[i]) for i in range(K)]
            try:
                a2 = a.reshape(-1, need_sd)
                if a2.shape[0] == K:
                    return [_pad_or_clip(a2[i]) for i in range(K)]
            except Exception:
                pass
            raise ValueError(f"无法解析 SD 动作的形状: {a.shape}，期望 (K,3)/(1,K,3)/list(len=K)。")

    def _normalize_joint_actions(self, actions):
        need_bs = len(self.bs_nvec)
        need_sd = 3
        K = self.K

        if isinstance(actions, (list, tuple)) and len(actions) == (1 + K):
            a0 = actions[0]
            # ---- 形状优先：一维且 size==M 且是数值型 -> 连续权重 ----
            if isinstance(a0, np.ndarray) and a0.ndim == 1 and a0.size == self.M and np.issubdtype(a0.dtype, np.number):
                bs_tr_weights = self._normalize_bs_weights(a0)
                A_bs = np.zeros((need_bs,), dtype=np.int64)  # 旧离散占位
            else:
                A_bs = self._pad_or_clip_int(a0, need_bs)
                bs_tr_weights = np.zeros((self.M,), dtype=np.float32)
            A_sd_list = [self._pad_or_clip_int(actions[i + 1], need_sd) for i in range(K)]
            return A_bs, A_sd_list, bs_tr_weights

        if isinstance(actions, np.ndarray) and actions.dtype == object and actions.size == (1 + K):
            a0 = actions[0]
            if isinstance(a0, np.ndarray) and a0.ndim == 1 and a0.size == self.M and np.issubdtype(a0.dtype, np.number):
                bs_tr_weights = self._normalize_bs_weights(a0)
                A_bs = np.zeros((need_bs,), dtype=np.int64)
            else:
                A_bs = self._pad_or_clip_int(a0, need_bs)
                bs_tr_weights = np.zeros((self.M,), dtype=np.float32)
            A_sd_list = [self._pad_or_clip_int(actions[i + 1], need_sd) for i in range(K)]
            return A_bs, A_sd_list, bs_tr_weights

        raise ValueError("无法解析 joint actions：期望长度 1+K 的 (list/tuple/object-ndarray)。")

    def _augment_all_agents(self, obs_all: List[np.ndarray]) -> np.ndarray:
        """
        返回 (1+K, obs_dim)：
          - idx=0：BS，one-hot=[1,0]
          - idx=1..K：SD，one-hot=[0,1]
        """
        out = []
        base = self.base_obs_dim
        # BS
        o_bs = np.asarray(obs_all[0], dtype=np.float32).reshape(-1)
        v_bs = np.zeros((self.obs_dim,), dtype=np.float32)
        v_bs[:min(o_bs.size, base)] = o_bs[:base]
        if self.add_type_flag:
            v_bs[-2:] = (1.0, 0.0)
        out.append(v_bs)
        # SDs
        for i in range(self.K):
            o = np.asarray(obs_all[i + 1], dtype=np.float32).reshape(-1)
            v = np.zeros((self.obs_dim,), dtype=np.float32)
            v[:min(o.size, base)] = o[:base]
            if self.add_type_flag:
                v[-2:] = (0.0, 1.0)
            out.append(v)
        return np.stack(out).astype(np.float32)

    def _compute_share_obs(self, obs_all: List[np.ndarray]) -> np.ndarray:
        """centralized critic 输入：优先 core.get_global_state()；否则拼接（BS+SD）的增强观测。"""
        if hasattr(self.core, "get_global_state") and callable(getattr(self.core, "get_global_state")):
            try:
                g = self.core.get_global_state()
                if isinstance(g, dict) and "vector" in g:
                    return np.asarray(g["vector"], dtype=np.float32).reshape(-1)
                return np.asarray(g, dtype=np.float32).reshape(-1)
            except Exception:
                pass

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

    def _read_global_state_vector(self, obs_all_fallback: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        优先从 EnvCore 读取全局向量。若失败，使用提供的 obs_all 或最近一次全量观测做拼接作为回退。
        """
        # 1) 直接尝试 EnvCore.get_global_state()
        if hasattr(self.core, "get_global_state") and callable(getattr(self.core, "get_global_state")):
            try:
                g = self.core.get_global_state()
                if isinstance(g, dict) and "vector" in g:
                    g = g["vector"]
                return np.asarray(g, dtype=np.float32).reshape(-1)
            except Exception:
                pass

        # 2) 回退：使用可用的 obs_all（优先外部传入 -> 最近一次 -> 最后兜底重置一次）
        if obs_all_fallback is None:
            try:
                if hasattr(self.core, "get_last_obs_all") and callable(getattr(self.core, "get_last_obs_all")):
                    obs_all_fallback = self.core.get_last_obs_all()
            except Exception:
                obs_all_fallback = None
        if obs_all_fallback is None:
            try:
                obs_all_fallback = self.core.reset()
            except Exception:
                obs_all_fallback = []

        return self._compute_share_obs(obs_all_fallback).astype(np.float32, copy=False)

    def get_share_obs(self) -> List[np.ndarray]:
        """给上层一个列表（长度 1+K），每个都是相同的 centralized 向量（与 _last_share 保持一致）。"""
        g = self._last_share.astype(np.float32, copy=True)
        return [g for _ in range(self.num_agents)]

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            return [None]
        np.random.seed(seed)
        if hasattr(self.core, "rng") and self.core.rng is not None:
            self.core.rng.seed(seed)
        return [seed]

    # ---------------- 动作掩码（仅 SD；BS 连续不需要） ---------------- #
    def get_action_masks(self) -> np.ndarray:
        """
        返回形状 (1+K, sd_sum_nvec) 的 float32 掩码：
          - agent 0（BS）：连续动作不需要 mask -> 全 0
          - agent 1..K（SD）：[dir | fly | ds] 的掩码
        """
        K = self.K
        n_dir = int(self.core.n_dir)
        n_fly = int(self.core.n_fly)
        n_pick = int(self.core.M) + 1
        width = self.mask_width

        out = np.zeros((1 + K, width), dtype=np.float32)  # 第 0 行(BS)保持全 0

        last_infos = self._last_infos if isinstance(self._last_infos, (list, tuple)) else None

        # 可选：首槽强制悬停到 0 档（若 core 未提供该标志，则默认 False）
        first_slot = bool(getattr(self.core, "first_slot_force_full_hover", False)) and int(
            getattr(self.core, "t", 1)) <= 1

        # 关键修复：若 core 未定义 min_hov_ratio_on_conn，则默认 0.0（不额外限制）
        rho = float(getattr(self.core, "min_hov_ratio_on_conn", 0.0))
        cap = int(np.floor((1.0 - rho) * max(n_fly - 1, 0)))
        cap = int(np.clip(cap, 0, n_fly - 1))

        for k in range(K):
            buf = np.zeros((n_dir + n_fly + n_pick,), dtype=np.float32)
            off = 0

            # dir：全开
            buf[off:off + n_dir] = 1.0
            off += n_dir

            # fly
            if first_slot:
                fly_mask = np.zeros(n_fly, dtype=np.float32)
                fly_mask[0] = 1.0
            else:
                fly_mask = None  # 稍后根据 valid ds 决定

            # ds
            pick_mask = np.zeros(n_pick, dtype=np.float32)

            # 从上一步 infos 获取该 SD 的可连 DS 掩码（若没有，则只允许 skip）
            valid_mask = None
            if last_infos is not None and len(last_infos) >= (1 + K):
                info_k = last_infos[k + 1]
                if isinstance(info_k, dict) and "mask_valid_ds" in info_k:
                    vm = np.asarray(info_k["mask_valid_ds"]).astype(np.int32).reshape(-1)
                    if vm.size == n_pick:
                        valid_mask = vm

            if valid_mask is None:
                # 没有可连 DS：仅允许 skip
                pick_mask[-1] = 1.0
                if fly_mask is None:
                    # 给个保守 fallback：允许少数几个档位（与旧逻辑一致）
                    fly_mask = np.zeros(n_fly, dtype=np.float32)
                    allow = sorted({min(2, n_fly - 1), n_fly - 1})
                    if len(allow) > 0:
                        fly_mask[allow] = 1.0
            else:
                # 有可连 DS：允许这些 DS，并始终允许 skip
                pick_mask = valid_mask.astype(np.float32, copy=True)
                pick_mask[-1] = 1.0
                # 若存在有效 DS，则限制最大飞行索引到 cap（rho 未定义时等效为全开）
                if np.any(valid_mask[:-1] > 0):
                    fly_mask = np.zeros(n_fly, dtype=np.float32)
                    fly_mask[:cap + 1] = 1.0
                else:
                    # 只有 skip：退回到保守 fallback
                    fly_mask = np.zeros(n_fly, dtype=np.float32)
                    allow = sorted({min(2, n_fly - 1), n_fly - 1})
                    if len(allow) > 0:
                        fly_mask[allow] = 1.0

            buf[off:off + n_fly] = fly_mask
            off += n_fly
            buf[off:off + n_pick] = pick_mask

            out[1 + k, :buf.size] = buf

        return out

    # 别名：兼容上层 Runner 的自动查询
    def action_masks(self):
        return self.get_action_masks()

    def get_available_actions(self):
        return self.get_action_masks()

    # ---------------- 便捷属性/方法（供算法配置读取） ---------------- #
    @property
    def agent_names(self) -> List[str]:
        return ["BS"] + [f"SD-{i}" for i in range(self.K)]

    @property
    def action_nvec_per_agent(self) -> List[np.ndarray]:
        """
        仅供外部参考：
          - BS 为连续动作，返回空 nvec（长度为 0 的 int64 数组）
          - SD 返回各自的 MultiDiscrete nvec
        """
        bs = np.asarray([], dtype=np.int64)
        sd = np.asarray(self.sd_nvec, dtype=np.int64)
        return [bs] + [sd for _ in range(self.K)]

    def get_algo_config(self) -> Dict[str, Union[int, List[int]]]:
        return {
            "role_dim": int(self.role_dim),
            "bs_nvec": list(self.bs_nvec),        # 仅供参考/兼容
            "sd_nvec": list(self.sd_nvec),
            "bs_enc_heads": int(self.core.L_BS_ENC),
            "bs_tx_bits": int(self.core.L_BS_TRANS),
            "sd_sc_heads": 1,
            "sd_ds_pick_semantics": getattr(self.core, "sd_ds_pick_semantics", "global"),
            "agent_order": "idx=0 is BS, then SD-0..SD-(K-1)",
            # 新增：BS 连续动作配置
            "bs_action_type": "box",
            "bs_weight_shape": [int(self.M)],
        }

    # 仅用于内部：让 reset() 的缓存能取到最近一次全量 obs
    def get_last_obs_all(self) -> List[np.ndarray]:
        if hasattr(self.core, "get_last_obs_all") and callable(getattr(self.core, "get_last_obs_all")):
            try:
                return self.core.get_last_obs_all()
            except Exception:
                pass
        return self.core.reset()
