# =====================================
# Filepath: envs/env_core_2actor.py
# -*- coding: utf-8 -*-
# =====================================

from typing import List, Tuple, Dict, Union
import math
import numpy as np


class EnvCore(object):
    """
    多智能体下行分发核心环境（K个SD）。
          1) 槽边界释放eRU、BS登记编码（产生未来返还事件）
          2) 解析SD动作，先判定“当前位置”的可连(valid)集合
          3) **先悬停**（若有 valid_set 至少悬停一定比例），在当前位置执行 SD→DS
          4) **后飞行**，更新位置并越界检测
          5) 飞行结束后执行 Tr→SD（轮转到活跃SD），按 BS 连续权重拉取
    """

    # ============================== 初始化 =============================== #
    def __init__(self, seed: int = None, debug: bool = True, comm_mode: str = "SC"):
        # ---- 基本参数 ----
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self.debug = bool(debug)
        self.comm_mode = str(comm_mode).upper()
        if self.comm_mode not in ("SC", "TC"):
            raise ValueError("comm_mode 必须是 'SC' 或 'TC'")
        self.demo = False  # 非demo模式下记得关掉

        # 规模与时间
        self.M = 20
        self.K = 2
        self.T = 400
        self.delta_T = 2.0

        # 打印摘要控制
        self._top_ds = self.M
        self._top_conn = self.M
        self._nz_limit = 10 ** 9

        # 地图/拓扑
        self.map_size = 3000.0
        self.cluster_center = np.array([2000.0, 2000.0], dtype=np.float32)
        self.tsd_pos = np.array([100.0, 100.0, 50.0], dtype=np.float32)  # Tr-UAV固定位置
        self.z_sd = 50.0
        self.ds_z = 0.0

        # DS 位置
        self.ds_min_sep = 50.0
        self.ds_pos = self._gen_ds_positions_around_center()
        self.ds_pos = np.hstack([self.ds_pos, np.full((self.M, 1), self.ds_z, dtype=np.float32)])  # (M,3)

        # SD 初始位置
        self.P_sd = np.zeros((self.K, 3), dtype=np.float32)
        self._place_sd_init_positions()

        # SD 运动/动作离散
        self.v_speed = 15.0
        self.n_dir = 12
        self.n_fly = 6
        self.dir_table = np.linspace(-math.pi, math.pi, num=self.n_dir, endpoint=False, dtype=np.float32)

        # 动作长度（BS侧）
        self.L_BS_ENC = 15
        self.L_BS_TRANS = self.M  # 传输位长度 = 每个 DS 一个 0/1 开关

        # ============= 物理层参数（Tr→SD） =============
        self.W_tr = 20.0
        self.G_tx_tr = 0.0
        self.G_rx_sd = 0.0
        self.f_tr_sd = 9e8
        self.B_tr_sd = 1e6
        self.F_tr_sd = 6.0
        self.gamma_tr_sd = 2.3
        self.sigma_tr_sd = 2.0
        self.N_sub_tr_sd = 24
        self.T_DFT_tr_sd = 32e-6
        self.T_GI_tr_sd = 8e-6
        self.snr_thr_tr_sd = 2.0
        self.N_ss_tr_sd = 1.0

        # ============= 物理层参数（SD→DS） =============
        self.W_sd = 20.0
        self.G_tx_sd = 0.0
        self.G_rx_ds = 0.0
        self.f_sd_ds = 2.4e9
        self.B_sd_ds = 20e6
        self.F_sd_ds = 6.0
        self.gamma_sd_ds_los = 2.3
        self.gamma_sd_ds_nlos = 3.6
        self.C_los = 1.0
        self.C_nlos = 20.0
        self.sigma_sd_ds = 1.0
        # SD->DS 整带宽到单一DS
        self.N_sub_sd_ds = 232  # 仅用于速率计算
        self.T_DFT_sd_ds = 12.8e-6
        self.T_GI_sd_ds = 3.2e-6
        self.N_ss = 1.0
        self.snr_thr_sd_ds = 7.0
        self.a_los = 4.88
        self.b_los = 0.43

        # 图像/压缩
        self.S_img = 1024 * 1024 * 3 * 8  # = 25_165_824 bits
        self.cmp_ratio = 1 / 50
        # === 新增：SC/TC 计量常量 ===
        # SC：Tr->SD 用「符号」计量；一张图所需符号数：Upsilon = BCR * 3 * N_H * N_W
        self.BCR = 1.0 / 192.0
        self.N_H_pix = 1024
        self.N_W_pix = 1024
        self.CHANNELS = 3
        self.Upsilon_sym_per_img = float(self.BCR * self.CHANNELS * self.N_H_pix * self.N_W_pix)  # = 16384
        # TC：编码固定时延（秒）
        self.tc_enc_delay_s = 1.0

        # 编码资源/释放事件
        self.RS_tot = 360
        self.RS_free = self.RS_tot
        self.tao_min = 4.0
        self.tao_max = 8.0
        self.finish_events: List[Tuple[int, int]] = []  # (t_release, m)

        # 队列
        self.Q_tr = np.zeros(self.M, dtype=int)  # Tr端 per-DS 压缩整图
        self.Q_sd = np.zeros((self.K, self.M), dtype=int)  # SD端 per-(k,m)

        # 饥饿期
        self.SP = np.zeros(self.M, dtype=int)

        # ==== 单位一律用 bit（内部） ====
        self.size_enc_img_bits = getattr(self, "size_enc_img_bits", self.cmp_ratio * self.S_img)
        self.size_orig_img_bits = getattr(self, "size_orig_img_bits", self.S_img)

        # >>> 新增（Mb/Mbps日志显示用，不影响内部计算）
        self._MBIT = 1e6  # 1 Mb = 1e6 bits

        self.bs_trans_mode = 'continuous'  # 'continuous' | 'discrete'（旧逻辑）
        self.bs_enc_by_rr = True  # True: 编码始终轮询；False: 读 enc_picks（这里保留开关但默认 True）

        # ======= Logging config (display-only) =======
        self.log_level = getattr(self, "log_level", "trace")  # 'basic'|'detail'|'trace'
        self.log_full = True
        # >>> 放宽截断参数，避免“头部显示/非零限制”
        self.img_size_Mb = float(round(self.size_orig_img_bits / self._MBIT, 3))
        self.enc_img_size_Mb = float(round(self.size_enc_img_bits / self._MBIT, 3))
        self._r1 = lambda x: float(round(float(x), 1))
        self._r2 = lambda x: float(round(float(x), 2))
        self._log_schema = {"STEP": {"basic": ["slot"]},
                            "RELEASE": {"basic": ["RS_free", "released_total", "released_nz"]},
                            "BS_ENC": {
                                "basic": ["RS_free_before", "alloc_nz", "tau_slot", "slot_release", "alloc_sum"],
                                "detail": ["enc_picks", "trans_picks", "enc_cum_nz"],
                            },
                            "SD_HOVER_THEN_FLY": {"basic": ["items"]},
                            "SD_to_DS@hover": {
                                "basic": ["delivered_total", "per_sd"],
                                "trace": ["mask_head"],
                                "detail": ["sd_cum_tx_nz"],
                            },
                            "TR_ALLOC_PER_SD": {
                                "basic": ["active", "items"],
                                "detail": [
                                    "mode", "cap_enc_imgs", "enc_img_size_Mb",
                                    "active_ds_map", "active_total_imgs", "active_total_Mb"
                                ]
                            },
                            "TR_PULL": {
                                "basic": [
                                    "active", "link_ok", "pulled_blocks",
                                    "cap_enc_imgs", "cap_Mb",
                                    "backlog_blocks", "backlog_Mb",
                                    "used_first",
                                    "d_tr_active"
                                ],
                                "detail": [
                                    "tr_cum_active_nz",
                                    "weights_in",
                                    "weights_eff",
                                    "used_by_weights_nz",
                                    "raw_tr_pull",
                                ],
                            },
                            "SP_REWARD": {
                                "basic": ["delivered_total", "avg_sp", "max_sp",
                                          "deliver_reward", "sp_penalty", "collision_penalty",
                                          "move_term", "reward"],
                                "detail": ["sp_nz",
                                           "move_pos", "move_neg",
                                           "w_move_pos", "w_move_away",
                                           "move_bonus_pos", "move_penalty_away",
                                           "components"]
                            },
                            "Q_STATE": {
                                "basic": ["Q_tr", "Q_sd", "DS_rx", "DS_rx_cum", "BS_enc_cum"],
                                "detail": ["DS_rx_from_sd", "DS_rx_cum_from_sd"]
                            },
                            "SD_LINKMAP": {"detail": ["per_sd"]}
                            }

        # 奖励
        self.w1_deliver = 10.0
        self.w2_sp = 0.01
        self.w_collision = 20.0
        self.boundary_penalty = 100.0
        self.collision_dsafe = 200.0

        # >>> 新增：SP-距离引导项参数
        self.w_move_pos = 0.1  # 朝目标移动的奖励权重
        self.w_move_away = 0.1  # 远离的惩罚权重（可略小）
        self.alpha_sp = 1  # SP 对“有效距离”的放大强度
        self.sp_norm_max = self.T  # 归一化上限跟随回合长度
        self.d0_norm = max(1e-6, self.v_speed * self.delta_T)  # 随新速度/Δt自适应

        self.decay_gamma = 0.7  # 衰减系数 γ∈(0,1]；越小衰减越快
        self.ds_streak = np.zeros(self.M, dtype=int)  # 各 DS 连续被服务计数
        # 时间步
        self.t = 1

        # 最近一次BS分配/拉取的简报（供打印）
        self._last_enc_log: Dict = {}
        self._last_pull_log: Dict = {}

        # SD 在悬停阶段选择 DS 的语义
        self.sd_ds_pick_semantics = 'global'

        # ---- 运行期统计（新增） ----
        # BS 编码
        self.bs_enc_last = np.zeros(self.M, dtype=int)
        self.cum_enc_started = np.zeros(self.M, dtype=int)

        # Tr -> SD
        self.tr_tx_last = np.zeros((self.K, self.M), dtype=int)
        self.cum_tr_to_sd = np.zeros((self.K, self.M), dtype=int)
        self.cum_tr_to_sd_total = np.zeros(self.K, dtype=int)

        # SD -> DS
        self.sd2ds_sent_last = np.zeros((self.K, self.M), dtype=int)
        self.cum_sd2ds = np.zeros((self.K, self.M), dtype=int)
        self.cum_sd2ds_total = np.zeros(self.K, dtype=int)

        # DS 接收来源（镜像 SD->DS）
        self.ds_rx_last_from_sd = np.zeros((self.K, self.M), dtype=int)
        self.cum_ds_rx_from_sd = np.zeros((self.K, self.M), dtype=int)
        # --- SNR per-slot cache（新增）---
        self._snr_cache_slot_sd_ds = None  # SD->DS 缓存属于哪个时隙 self.t
        self._snr_cache_sd_ds = None  # (K, M) 本时隙冻结的 SD->DS SNR(dB)

        self._snr_cache_slot_tr_sd = None  # Tr->SD 缓存属于哪个时隙 self.t
        self._snr_cache_tr_sd = None  # (K,) 本时隙每个 SD 一份 Tr->SD SNR(dB)

        # ==== NEW: BS 策略开关 ====
        self.use_bs_agent = True
        # 轮询指针（跨槽记忆，保证“公平性”）
        self._bs_rr_ptr_enc = 0  # 编码 eRU 轮询指针（0..M-1）
        self._bs_rr_ptr_trsd = 0  # Tr->SD 拉取轮询指针（0..M-1）
        self._apply_comm_mode()

    # =====================================
    # Modified reset() method
    # =====================================
    def reset(self):
        self.RS_free = self.RS_tot
        self.finish_events.clear()
        self.Q_tr[:] = 0
        self.Q_sd[:, :] = 0
        self.SP[:] = 0
        self.t = 1

        # 运行期统计清零（新增）
        self.bs_enc_last[:] = 0
        self.cum_enc_started[:] = 0
        self.tr_tx_last[:, :] = 0
        self.cum_tr_to_sd[:, :] = 0
        self.cum_tr_to_sd_total[:] = 0
        self.sd2ds_sent_last[:, :] = 0
        self.cum_sd2ds[:, :] = 0
        self.cum_sd2ds_total[:] = 0
        self.ds_rx_last_from_sd[:, :] = 0
        self.cum_ds_rx_from_sd[:, :] = 0

        # 清空按时隙的 SNR 缓存（新增）
        self._snr_cache_slot_sd_ds = None
        self._snr_cache_sd_ds = None
        self._snr_cache_slot_tr_sd = None
        self._snr_cache_tr_sd = None
        self._last_enc_log = {}
        self._last_pull_log = {}

        self._place_sd_init_positions()

        if self.debug:
            self._log_reset_map()
            self._log_sd_positions_slot(0)
            print(f"[CONST] img_size_Mb={self.img_size_Mb}, enc_img_size_Mb={self.enc_img_size_Mb}")

        # ==== NEW: 轮询指针复位 ====
        self._bs_rr_ptr_enc = 0
        self._bs_rr_ptr_trsd = 0
        self._apply_comm_mode()
        self.ds_streak[:] = 0
        return self._build_obs_all()

    def step(self, actions: Union[List, Tuple, Dict]):
        # per-slot 清零
        self.bs_enc_last[:] = 0
        self.tr_tx_last[:, :] = 0
        self.sd2ds_sent_last[:, :] = 0
        self.ds_rx_last_from_sd[:, :] = 0

        # =============== 1) 解包动作（兼容 2 返回值或 3 返回值） ===============
        unpacked = self._unpack_actions(actions)

        if isinstance(unpacked, (list, tuple)):
            if len(unpacked) == 3:
                bs_tr_weights, sd_actions, _ = unpacked
                enc_picks = np.empty(0, dtype=int)  # 不使用 enc_picks
            elif len(unpacked) == 2:
                bs_tr_weights, sd_actions = unpacked
                enc_picks = np.empty(0, dtype=int)
            else:
                raise ValueError(f"_unpack_actions 应返回 2 或 3 个对象，当前={len(unpacked)}")
        else:
            raise TypeError(f"_unpack_actions 返回类型应为 tuple/list，当前={type(unpacked)}")

        # 统一规范连续权重：长度=M，非负（只做一次裁剪/填充/截断）
        bs_tr_weights = np.asarray(bs_tr_weights, dtype=np.float32).reshape(-1)
        if bs_tr_weights.size != self.M:
            if bs_tr_weights.size > self.M:
                bs_tr_weights = bs_tr_weights[:self.M]
            else:
                bs_tr_weights = np.pad(bs_tr_weights, (0, self.M - bs_tr_weights.size), mode="constant")
        bs_tr_weights = np.clip(bs_tr_weights, 0.0, np.inf)

        if self.debug:
            self._p("STEP", slot=self.t)
            # 纯展示：连续权重（原始）
            self._p("BS_TR_WEIGHTS",
                    weights_in=bs_tr_weights.astype(float).tolist(),
                    nz=self._nz_summary(bs_tr_weights))

        # =============== 2) 槽边界释放 eRU -> Q_tr ===============
        released_per_m = np.zeros(self.M, dtype=int)
        pre_events = self.finish_events
        self.finish_events = []
        for t_rel, m in pre_events:
            if t_rel == self.t:
                self.RS_free += 1
                self.Q_tr[m] += 1
                released_per_m[m] += 1
            else:
                self.finish_events.append((t_rel, m))
        if self.debug:
            self._p("RELEASE",
                    RS_free=self.RS_free,
                    released_total=int(released_per_m.sum()),
                    released_vec=released_per_m,
                    released_nz=self._nz_summary(released_per_m))

        # =============== 3) BS 编码（强制轮询；忽略 enc_picks） ===============
        if self.RS_free > 0:
            # 统一采用“登记返还事件”：SC 随机时延；TC 固定 1s
            self._apply_bs_encoding_auto_rr(int(self.RS_free))
            mode_enc = "auto_rr_tc_fixed" if self.comm_mode == "TC" else "auto_rr_sc_random"
            if self.debug:
                log = (self._last_enc_log.copy() if self._last_enc_log else {})
                log.update({
                    "mode": mode_enc,
                    "enc_picks": enc_picks.tolist(),
                    "enc_cum_nz": self._nz_summary(self.cum_enc_started),
                })
                if "alloc" in log and "alloc_nz" not in log:
                    log["alloc_nz"] = self._nz_summary(log["alloc"])
                self._p("BS_ENC", **log)
            # eRU 被占用，置 0（等待未来返还）
            self.RS_free = 0
        else:
            if self.debug:
                self._p("BS_ENC", note="RS_free=0, skip")

        # =============== 4) 解析 SD 动作（名义飞/停），先在当前位置判定 valid_set ===============
        dir_idx, fly_idx, ds_idx = self._parse_sd_actions(sd_actions)
        phis = self.dir_table[dir_idx]

        fly_times_nominal = (fly_idx / max(self.n_fly - 1, 1)) * self.delta_T
        hov_times_nominal = np.clip(self.delta_T - fly_times_nominal, 0.0, None)

        pre_pos = self.P_sd.copy()  # 悬停位置=当前位置

        # 冻结本槽 SD->DS SNR
        snr_slot_pre = self._ensure_snr_cache_sd_ds().copy()

        # （日志）全量 linkmap
        linkmap_rows = []
        for k in range(self.K):
            L_all = np.linalg.norm(self.ds_pos - pre_pos[k], axis=1)  # (M,)
            S_all = snr_slot_pre[k]
            ds_d_all_str = self._as_idx_map_str(L_all, rounder=self._r1)
            ds_snr_all_str = self._as_idx_map_str(S_all, rounder=self._r1)
            conn_mask = (S_all >= self.snr_thr_sd_ds)
            conn_idx = np.where(conn_mask)[0]
            if conn_idx.size > 0:
                order = np.argsort(L_all[conn_idx])
                head = [f"{int(m)}:{self._r1(L_all[int(m)])}" for m in conn_idx[order][:self._top_conn]]
                conn_head_str = "{ " + ", ".join(head) + (" ... }" if conn_idx.size > self._top_conn else " }")
            else:
                conn_head_str = "∅"
            linkmap_rows.append({"k": int(k), "ds_d_all": ds_d_all_str, "ds_snr_all": ds_snr_all_str,
                                 "conn_ds_d_head": conn_head_str})
        if self.debug:
            self._p("SD_LINKMAP", per_sd=linkmap_rows)

        # 基于当前位置构造 conn_set 与 valid_set
        conn_sets, valid_sets = [], []
        for k in range(self.K):
            snr_all = snr_slot_pre[k]
            conn_mask = (snr_all >= self.snr_thr_sd_ds)
            conn_idx = np.where(conn_mask)[0]
            conn_sets.append(conn_idx)

            has_data_mask = (self.Q_sd[k] > 0)
            valid_mask = conn_mask & has_data_mask
            valid_idx = np.where(valid_mask)[0]
            valid_sets.append(valid_idx)

        dist_tr_pre = np.linalg.norm(pre_pos - self.tsd_pos[None, :], axis=1)  # (K,)

        # —— 最终采用：完全按动作 —— #
        fly_times = fly_times_nominal.astype(np.float32, copy=True)
        hov_times = hov_times_nominal.astype(np.float32, copy=False)
        for k in range(self.K):
            if valid_sets[k].size == 0:
                fly_times[k] = float(self.delta_T)  # 满飞
                hov_times[k] = 0.0
                ds_idx[k] = self.M  # NOOP

        if self.debug:
            fly_info = []
            for k in range(self.K):
                item = {
                    "k": int(k),
                    "fly_t": self._r1(fly_times[k]),
                    "hov_t": self._r1(hov_times[k]),
                    "valid_set_size": int(valid_sets[k].size),
                    "d_tr": self._r1(dist_tr_pre[k]),
                }
                if self.log_level != "basic":
                    item["phi_deg"] = self._r1(np.degrees(phis[k]))
                conn_idx = conn_sets[k]
                if conn_idx.size > 0:
                    d_conn = np.linalg.norm(self.ds_pos[conn_idx] - pre_pos[k], axis=1)
                    order = np.argsort(d_conn)
                    take = int(min(self._top_conn, conn_idx.size))
                    head_pairs = [f"{int(conn_idx[j])}:{self._r1(d_conn[j])}" for j in order[:take]]
                    item["conn_ds_d_head"] = "{ " + ", ".join(head_pairs) + (
                        " }" if conn_idx.size <= take else ", ... }")
                else:
                    item["conn_ds_d_head"] = "∅"
                fly_info.append(item)
            self._p("SD_HOVER_THEN_FLY", items=fly_info)

        # =============== 5) 悬停阶段执行 SD->DS ===============
        delivered_total = 0
        delivered_per_m = np.zeros(self.M, dtype=int)
        per_sd_brief: List[Dict[str, object]] = []
        T_sym = self.T_DFT_sd_ds + self.T_GI_sd_ds
        ds_exec = np.full(self.K, self.M, dtype=int)

        for k in range(self.K):
            valid_idx = valid_sets[k]

            mask_valid = np.zeros(self.M + 1, dtype=int)
            mask_valid[self.M] = 1
            mask_valid[valid_idx] = 1

            raw_pick = int(ds_idx[k])
            vmask = np.zeros(self.M, dtype=bool)
            vmask[valid_idx] = True
            pick = self._map_sd_ds_pick(raw_pick, vmask, mode=self.sd_ds_pick_semantics)

            if hov_times[k] <= 1e-8:
                per_sd_brief.append({"k": int(k), "note": "hov=0, skip"})
                ds_exec[k] = self.M
                continue

            if valid_idx.size == 0:
                per_sd_brief.append({"k": int(k), "note": "valid=∅ -> skip"})
                ds_exec[k] = self.M
                continue

            if pick == self.M:
                if hov_times[k] > 1e-8 and valid_idx.size > 0:
                    best_local = int(valid_idx[np.argmax(snr_slot_pre[k, valid_idx])])
                    pick = best_local
                else:
                    per_sd_brief.append({"k": int(k), "pick": "skip", "note": "no-op"})
                    ds_exec[k] = self.M
                    continue

            snr_db = float(snr_slot_pre[k, pick])
            if snr_db < self.snr_thr_sd_ds:
                per_sd_brief.append({"k": int(k), "pick": int(pick),
                                     "snr_db": round(snr_db, 1),
                                     "thr_db": float(self.snr_thr_sd_ds),
                                     "note": "not connectable"})
                ds_exec[k] = self.M
                continue

            bps, cr = self._lookup_mcs(snr_db)
            Rm = bps * cr * self.N_ss * float(self.N_sub_sd_ds) / T_sym
            rate_Mbps = float(Rm / self._MBIT)
            C_bits_sd_ds = Rm * float(hov_times[k])
            cap_orig_imgs_sd_ds = int(C_bits_sd_ds // self.size_orig_img_bits)
            cap_Mb_sd_ds = float(C_bits_sd_ds / self._MBIT)
            can_tx = int((Rm * float(hov_times[k])) // self.S_img)

            if can_tx <= 0:
                per_sd_brief.append({
                    "k": int(k), "pick": int(pick),
                    "snr_db": round(snr_db, 1),
                    "rate_Mbps": self._r1(rate_Mbps),
                    "cap_Mb": self._r2(cap_Mb_sd_ds),
                    "hov_s": self._r1(hov_times[k]),
                    "note": "rate too low"
                })
                ds_exec[k] = self.M
                continue

            x = min(can_tx, int(self.Q_sd[k, pick]))
            if x > 0:
                self.Q_sd[k, pick] -= x
                delivered_per_m[pick] += x
                delivered_total += int(x)
                self.sd2ds_sent_last[k, pick] += x
                ds_exec[k] = pick
                per_sd_brief.append({
                    "k": int(k), "pick": int(pick), "take": int(x),
                    "snr_db": round(snr_db, 1),
                    "rate_Mbps": self._r1(rate_Mbps),
                    "cap_Mb": self._r2(cap_Mb_sd_ds),
                    "d_pick": self._r1(float(np.linalg.norm(pre_pos[k] - self.ds_pos[pick]))),
                    "cap": {"cap_orig_imgs": int(cap_orig_imgs_sd_ds)}
                })
            else:
                per_sd_brief.append({
                    "k": int(k), "pick": int(pick),
                    "snr_db": round(snr_db, 1),
                    "rate_Mbps": self._r1(rate_Mbps),
                    "d_pick": self._r1(float(np.linalg.norm(pre_pos[k] - self.ds_pos[pick]))),
                    "note": "no data"
                })
                ds_exec[k] = self.M

            per_sd_brief[-1]["mask_valid_ds_head"] = "{ " + ", ".join(
                [f"{i}" for i in np.where(mask_valid > 0)[0][:min(10, mask_valid.sum())]]
            ) + (" ... }" if mask_valid.sum() > 10 else " }")

        # SD->DS 阶段结束：镜像到 DS，滚动累计
        self.ds_rx_last_from_sd = self.sd2ds_sent_last.copy()
        self.cum_sd2ds += self.sd2ds_sent_last
        self.cum_sd2ds_total += self.sd2ds_sent_last.sum(axis=1)
        self.cum_ds_rx_from_sd += self.sd2ds_sent_last

        if self.debug:
            per_sd_out = []
            for d in per_sd_brief:
                out = {"k": int(d["k"])}
                if d.get("note") in ("hov=0, skip", "valid=∅ -> skip"):
                    out["note"] = d["note"]
                else:
                    for k2 in ("pick", "snr_db", "rate_Mbps", "cap_Mb", "hov_s", "take", "note", "d_pick"):
                        if k2 in d:
                            out[k2] = d[k2]
                if self.log_level == "trace" and "mask_valid_ds_head" in d:
                    out["mask_head"] = d["mask_valid_ds_head"]
                per_sd_out.append(out)
            self._p("SD_to_DS@hover",
                    delivered_total=int(delivered_total),
                    per_sd=per_sd_out,
                    sd_cum_tx=self.cum_sd2ds.astype(int),
                    sd_cum_tx_nz=[f"{k}:{self._nz_summary(self.cum_sd2ds[k])}" for k in range(self.K)]
                    )

        # =============== 6) 飞行更新与越界检查 ===============
        dxy_real = np.stack([np.cos(phis), np.sin(phis)], axis=1) * fly_times[:, None] * self.v_speed
        post_pos = pre_pos.copy()
        post_pos[:, :2] = pre_pos[:, :2] + dxy_real[:, :2]
        post_pos[:, 2] = self.z_sd

        out_x = (post_pos[:, 0] < 0) | (post_pos[:, 0] > self.map_size)
        out_y = (post_pos[:, 1] < 0) | (post_pos[:, 1] > self.map_size)
        if (out_x | out_y).any():
            if self.debug:
                self._p("BOUNDARY_AFTER_FLY", violated=list(np.where((out_x | out_y))[0]))
                self.P_sd[:, :] = post_pos
                self._log_sd_positions_slot(self.t)
            obs_list = self._build_obs_all()
            rews = [-self.boundary_penalty] + [-self.boundary_penalty] * self.K
            dones = [True] * (1 + self.K)
            infos = [{'boundary_violation': True}] + [{'boundary_violation': True} for _ in range(self.K)]
            return obs_list, rews, dones, infos

        # 应用位置更新
        self.P_sd[:, :] = post_pos

        # =============== 7) Tr→SD：连续权重整数分配 or 轮询 ===============
        active = (self.t - 1) % self.K
        mid_pos = 0.5 * (pre_pos[active] + post_pos[active])
        L_tr_sd = float(np.linalg.norm(mid_pos - self.tsd_pos))
        snr_trsd = self._get_snr_tr_sd_slot(active, L_tr_sd)
        link_ok = (snr_trsd >= self.snr_thr_tr_sd) and (float(fly_times[active]) > 0.0)

        # >>> 记录本槽 TR 分配前的 backlog（用于统一的日志口径）
        qtr_sum_before = int(self.Q_tr.sum())

        rx_total = 0
        moved = np.zeros(self.M, dtype=int)  # 本槽各 DS -> active SD 的分配结果（单位：压缩整图）

        if link_ok:
            enc_img_size_Mb = float(self.size_enc_img_bits / self._MBIT)

            if self.comm_mode == "SC":
                # ===== SC：Tr->SD 用“符号数 / Upsilon”计量 =====
                T_sym_tr = float(self.T_DFT_tr_sd + self.T_GI_tr_sd)
                sy_rate = (float(self.N_sub_tr_sd) * float(self.N_ss_tr_sd)) / max(T_sym_tr, 1e-12)  # symbols/s
                tx_time_s = float(fly_times[active])
                syms_avail = sy_rate * tx_time_s
                cap_enc_imgs_tr_sd = int(syms_avail // max(self.Upsilon_sym_per_img, 1e-9))
                # —— 为强调符号计量，SC 下 MB 类字段置为 0 —— #
                cap_Mb_tr_sd = 0.0
            else:
                # ===== TC：bit 速率 + MCS =====
                T_sym_tr = float(self.T_DFT_tr_sd + self.T_GI_tr_sd)
                bps_tr, cr_tr = self._lookup_mcs(float(snr_trsd))
                Rm_tr = float(bps_tr) * float(cr_tr) * float(self.N_ss_tr_sd) * float(self.N_sub_tr_sd) / max(T_sym_tr, 1e-12)  # bits/s
                tx_time_s = float(fly_times[active])
                C_bits_tr_sd = Rm_tr * tx_time_s
                cap_enc_imgs_tr_sd = int(C_bits_tr_sd // max(self.size_enc_img_bits, 1.0))
                cap_Mb_tr_sd = float(C_bits_tr_sd / self._MBIT)

            if self.use_bs_agent:
                # === 连续权重按比例分配（满足 qmax） ===
                qmax = self.Q_tr.copy()  # 每个 DS 可用上限（背压）
                weights_in = bs_tr_weights.copy()  # 来自策略的原始权重（长度 M）
                moved, self._bs_rr_ptr_trsd = self._alloc_prop_cap_by_weights(
                    cap=int(cap_enc_imgs_tr_sd),
                    weights=weights_in,
                    qmax=qmax,
                    rr_ptr=int(self._bs_rr_ptr_trsd),
                    fallback_uniform_if_zero=True
                )

                rx_total = int(moved.sum())
                if rx_total > 0:
                    self.Q_tr -= moved
                    self.Q_sd[active] += moved

                backlog_blocks_selected = int(min(cap_enc_imgs_tr_sd, qtr_sum_before))

                # —— 日志：SC/TC 分支区分 —— #
                if self.comm_mode == "SC":
                    eligible_mask = (qmax > 0).astype(int)
                    w_eff = np.clip(weights_in, 0.0, np.inf) * eligible_mask
                    s = float(w_eff.sum())
                    w_norm = (w_eff / max(s, 1e-12)) if s > 0 else np.zeros_like(w_eff)

                    self._last_pull_log = {
                        "mode": "agent_cont_sc_symbol",
                        "quota": int(cap_enc_imgs_tr_sd),
                        "pulled": int(rx_total),
                        "used_first": moved.astype(int),
                        "cap_enc_imgs_theory": int(cap_enc_imgs_tr_sd),
                        "cap_syms_theory": float(syms_avail),
                        "Upsilon_sym_per_img": float(self.Upsilon_sym_per_img),
                        "cap_Mb_theory": 0.0,
                        "enc_img_size_Mb": float(enc_img_size_Mb),
                        "backlog_blocks_selected": int(backlog_blocks_selected),
                        "backlog_enc_Mb": 0.0,
                        "used_Mb": 0.0,
                        "weights_in": weights_in.astype(float).tolist(),
                        "eligible_mask": eligible_mask.astype(int).tolist(),
                        "weights_eff": w_eff.astype(float).tolist(),
                        "weights_norm": w_norm.astype(float).tolist(),
                        "used_by_weights_nz": self._nz_summary(moved),
                    }
                    log_mode = "agent_cont_sc_symbol"
                else:
                    eligible_mask = (qmax > 0).astype(int)
                    w_eff = np.clip(weights_in, 0.0, np.inf) * eligible_mask
                    s = float(w_eff.sum())
                    w_norm = (w_eff / max(s, 1e-12)) if s > 0 else np.zeros_like(w_eff)

                    used_Mb = float(rx_total * enc_img_size_Mb)
                    backlog_enc_Mb = float(backlog_blocks_selected * enc_img_size_Mb)
                    self._last_pull_log = {
                        "mode": "agent_cont",
                        "quota": int(cap_enc_imgs_tr_sd),
                        "pulled": int(rx_total),
                        "used_first": moved.astype(int),
                        "cap_enc_imgs_theory": int(cap_enc_imgs_tr_sd),
                        "cap_Mb_theory": float(cap_Mb_tr_sd),
                        "backlog_blocks_selected": int(backlog_blocks_selected),
                        "enc_img_size_Mb": float(enc_img_size_Mb),
                        "backlog_enc_Mb": float(backlog_enc_Mb),
                        "used_Mb": float(used_Mb),
                        "weights_in": weights_in.astype(float).tolist(),
                        "eligible_mask": eligible_mask.astype(int).tolist(),
                        "weights_eff": w_eff.astype(float).tolist(),
                        "weights_norm": w_norm.astype(float).tolist(),
                        "used_by_weights_nz": self._nz_summary(moved),
                    }
                    log_mode = "agent_cont"
            else:
                # === 自动轮询 ===
                moved = self._auto_rr_tr_pull(active=active, cap_enc_imgs=cap_enc_imgs_tr_sd)
                rx_total = int(moved.sum())
                backlog_blocks_selected = int(min(cap_enc_imgs_tr_sd, qtr_sum_before))

                if self.comm_mode == "SC":
                    self.tr_tx_last[:, :] = 0
                    self._last_pull_log = {
                        "mode": "auto_rr_sc_symbol",
                        "quota": int(cap_enc_imgs_tr_sd),
                        "pulled": int(rx_total),
                        "used_rr": moved.astype(int),
                        "cap_enc_imgs_theory": int(cap_enc_imgs_tr_sd),
                        "cap_syms_theory": float(syms_avail),
                        "Upsilon_sym_per_img": float(self.Upsilon_sym_per_img),
                        "cap_Mb_theory": 0.0,
                        "enc_img_size_Mb": float(enc_img_size_Mb),
                        "backlog_blocks_selected": int(backlog_blocks_selected),
                        "backlog_enc_Mb": 0.0,
                        "used_first": moved.astype(int)
                    }
                    log_mode = "auto_rr_sc_symbol"
                else:
                    used_Mb = float(rx_total * enc_img_size_Mb)
                    backlog_enc_Mb = float(backlog_blocks_selected * enc_img_size_Mb)

                    self.tr_tx_last[:, :] = 0
                    self._last_pull_log = {
                        "mode": "auto_rr",
                        "quota": int(cap_enc_imgs_tr_sd),
                        "pulled": int(rx_total),
                        "used_rr": moved.astype(int),
                        "cap_enc_imgs_theory": int(cap_enc_imgs_tr_sd),
                        "cap_Mb_theory": float(cap_Mb_tr_sd),
                        "backlog_blocks_selected": int(backlog_blocks_selected),
                        "enc_img_size_Mb": float(enc_img_size_Mb),
                        "backlog_enc_Mb": float(backlog_enc_Mb),
                        "used_Mb": float(used_Mb),
                        "used_first": moved.astype(int)
                    }
                    log_mode = "auto_rr"

            # 记录 Tr->SD（仅 active 行非零）+ 累计
            self.tr_tx_last[:, :] = 0
            self.tr_tx_last[active, :] = moved
            self.cum_tr_to_sd[active, :] += moved
            self.cum_tr_to_sd_total[active] += rx_total

            if self.debug:
                self._log_tr_alloc_per_sd(
                    active=active,
                    moved=moved,
                    mode=log_mode,
                    cap_enc_imgs=int(cap_enc_imgs_tr_sd)
                )
        else:
            # 无链路：保持 moved=0，并写入占位日志以免沿用旧值
            enc_img_size_Mb = float(self.size_enc_img_bits / self._MBIT)
            self._last_pull_log = {
                "mode": "no_link",
                "pulled": 0,
                "cap_enc_imgs_theory": 0,
                "cap_Mb_theory": 0.0,
                "backlog_blocks_selected": 0,
                "backlog_enc_Mb": 0.0,
                "enc_img_size_Mb": float(enc_img_size_Mb),
                "used_first": np.zeros(self.M, dtype=int),
            }
            if self.debug:
                self._log_tr_alloc_per_sd(
                    active=active,
                    moved=np.zeros(self.M, dtype=int),
                    mode="no_link",
                    cap_enc_imgs=0
                )

        if self.debug:
            log = self._last_pull_log
            self._p("TR_PULL",
                    active=active,
                    link_ok=bool(link_ok),
                    pulled_blocks=int(log.get("pulled", 0)),
                    cap_enc_imgs=int(log.get("cap_enc_imgs_theory", 0)),
                    cap_Mb=self._r2(log.get("cap_Mb_theory", 0.0)),
                    backlog_blocks=int(log.get("backlog_blocks_selected", 0)),
                    backlog_Mb=self._r2(log.get("backlog_enc_Mb", 0.0)),
                    used_first=self._nz_summary(log.get("used_first", np.zeros(self.M, dtype=int))),
                    d_tr_active=self._r1(L_tr_sd),
                    tr_cum_active_nz=self._nz_summary(self.cum_tr_to_sd[active]),
                    raw_tr_pull=self._last_pull_log
                    )

        # =============== 8) 队列总览日志 ===============
        if self.debug:
            sd_q_list = [f"{k}:{self._nz_summary(self.Q_sd[k])}" for k in range(self.K)]
            ds_rx_cum_sum = self.cum_ds_rx_from_sd.sum(axis=0)
            ds_rx_from_sd_lines = [f"{k}:{self._nz_summary(self.ds_rx_last_from_sd[k])}" for k in range(self.K)]
            ds_rx_cum_from_sd_lines = [f"{k}:{self._nz_summary(self.cum_ds_rx_from_sd[k])}" for k in range(self.K)]
            self._p("Q_STATE",
                    Q_tr=self.Q_tr,
                    Q_sd=sd_q_list,
                    Q_sd_mat=self.Q_sd.astype(int),
                    DS_rx=delivered_per_m,
                    DS_rx_cum=self._nz_summary(ds_rx_cum_sum),
                    BS_enc_cum=self._nz_summary(self.cum_enc_started),
                    DS_rx_from_sd=ds_rx_from_sd_lines,
                    DS_rx_cum_from_sd=ds_rx_cum_from_sd_lines
                    )

        # =============== 9) 更新SP与奖励 ===============
        served_now = (delivered_per_m > 0).astype(np.int32)
        self.ds_streak = (self.ds_streak + 1) * served_now  # 服务->+1；未服务->0
        decay = np.power(float(self.decay_gamma), np.maximum(self.ds_streak - 1, 0))  # 1, γ, γ^2, ...
        for m in range(self.M):
            self.SP[m] = 0 if delivered_per_m[m] > 0 else self.SP[m] + 1
        avg_sp = float(np.mean(self.SP))

        move_toward_sum = 0.0
        move_away_sum = 0.0
        for k in range(self.K):
            J_pre = self._effective_distance_J(pre_pos[k])
            J_post = self._effective_distance_J(post_pos[k])
            dJ_norm = (J_pre - J_post) / self.d0_norm
            if dJ_norm >= 0:
                move_toward_sum += dJ_norm
            else:
                move_away_sum += (-dJ_norm)

        move_bonus_pos = float(self.w_move_pos * move_toward_sum)
        move_penalty_away = float(self.w_move_away * move_away_sum)
        move_bonus = move_bonus_pos - move_penalty_away
        collision_penalty = 0.0

        self._last_move_pos = float(move_toward_sum)
        self._last_move_neg = float(move_away_sum)
        self._last_move_bonus_pos = float(move_bonus_pos)
        self._last_move_penalty_away = float(move_penalty_away)
        self._last_move_bonus = float(move_bonus)

        if self.K >= 2:
            se_xy = self.P_sd[:, :2]
            dmat = np.linalg.norm(se_xy[:, None, :] - se_xy[None, :, :], axis=2)
            iu = np.triu_indices(self.K, 1)
            pair_d = dmat[iu]
            if pair_d.size > 0:
                viol = np.maximum(self.collision_dsafe - pair_d, 0.0)
                collision_penalty = self.w_collision * float(np.mean(viol / max(self.collision_dsafe, 1e-6)))
        self._last_collision_penalty = float(collision_penalty)

        deliver_reward = float(self.w1_deliver * ((decay * delivered_per_m).sum() / max(1, self.M)))
        sp_penalty = float(self.w2_sp * (self.SP.sum() / max(1, self.M)))
        shared_reward = float(deliver_reward - sp_penalty - collision_penalty + move_bonus)

        if self.debug:
            sp_nz = self._nz_summary(self.SP)
            self._p("SP_REWARD",
                    delivered_total=int(delivered_total),
                    avg_sp=float(avg_sp),
                    max_sp=int(self.SP.max()),
                    deliver_reward=float(deliver_reward),
                    sp_penalty=float(sp_penalty),
                    collision_penalty=float(collision_penalty),
                    move_pos=float(self._last_move_pos),
                    move_neg=float(self._last_move_neg),
                    w_move_pos=float(self.w_move_pos),
                    w_move_away=float(self.w_move_away),
                    move_bonus_pos=float(self._last_move_bonus_pos),
                    move_penalty_away=float(self._last_move_penalty_away),
                    move_term=float(self._last_move_bonus),
                    reward=float(shared_reward),
                    sp_nz=sp_nz,
                    components={
                        "deliver": float(deliver_reward),
                        "minus_sp": -float(sp_penalty),
                        "minus_collision": -float(collision_penalty),
                        "move_pos": float(self._last_move_pos),
                        "move_neg": -float(self._last_move_neg),
                        "move_bonus_pos": float(self._last_move_bonus_pos),
                        "move_penalty_away": -float(self._last_move_penalty_away),
                        "move_term": float(self._last_move_bonus),
                        "total": float(shared_reward),
                    })
            self._log_sd_positions_slot(self.t)

        # =============== 10) 推进时间并返回 ===============
        self.t += 1
        done_flag = (self.t > self.T)
        obs_list = self._build_obs_all()

        rews = [shared_reward] * (1 + self.K)
        dones = [done_flag] * (1 + self.K)

        infos = self._build_info_all(
            active=active,
            rx_total=rx_total,
            delivered_total=delivered_total,
            avg_sp=avg_sp,
            conn_sets=conn_sets,
            valid_sets=valid_sets,
            ds_exec=ds_exec
        )
        return obs_list, rews, dones, infos

    # =============================== 观测/信息 ============================ #
    def _build_obs_all(self):
        obs_bs = self._obs_bs()
        obs_sd = [self._obs_sd(k) for k in range(self.K)]
        return [obs_bs] + obs_sd

    def _obs_bs(self):
        """
        BS 侧观测（原 5 块） + 连续权重分配所需的掩码/提示：
          1) Tr 队列 per-DS 的 [ds_id_norm, Q_tr[m]] -> 2M
          2) 当前 Tr↔激活 SD 的“理论可传输编码图像数量”（整图单位） -> 1
          3) 激活 SD 的 id_norm -> 1
          4) 激活 SD 的队列：[id_norm] + per-DS 的 [ds_id_norm, Q_sd[active,m]] -> (1+2M)
          5) 激活 SD 与全体 DS 的 SD→DS SNR（按需要归一化）+ SNR 阈值 -> (M+1)
          6) **新增**：mask_has_data = 1(Q_tr>0)（连续权重的“动作掩码”提示） -> M
          7) **新增**：qtr_norm = Q_tr / (1 + sum(Q_tr))（供策略感知 backlog 比例） -> M
        """
        M = int(self.M)
        K = int(self.K)

        # --------- 公共：DS 的归一化编号 ---------
        if M > 1:
            ds_id_norm = np.arange(M, dtype=np.float32) / float(M - 1)
        else:
            ds_id_norm = np.zeros(M, dtype=np.float32)

        # --------- 1) Tr 队列：2M 维 ---------
        qtr = self.Q_tr.astype(np.float32, copy=False)
        tr_pairs = np.stack([ds_id_norm, qtr], axis=1).reshape(-1)  # (2M,)

        # --------- 当前激活 SD 的索引 ---------
        active = int((self.t - 1) % max(K, 1))

        # --------- 2) Tr↔SD 理论可传输编码图像数量（1 维） ---------
        cap_imgs = 0.0
        if K > 0:
            L_tr_sd = float(np.linalg.norm(self.P_sd[active] - self.tsd_pos))
            snr_trsd = float(self._get_snr_tr_sd_slot(active, L_tr_sd))
            T_sym = float(self.T_DFT_tr_sd + self.T_GI_tr_sd)
            if snr_trsd >= float(self.snr_thr_tr_sd):
                if self.comm_mode == "SC":
                    # —— SC：符号计量 —— #
                    sy_rate = (float(self.N_sub_tr_sd) * float(self.N_ss_tr_sd)) / max(T_sym, 1e-9)  # symbols/s
                    syms_avail = sy_rate * float(self.delta_T)
                    cap_imgs = float(syms_avail // max(self.Upsilon_sym_per_img, 1e-9))
                else:
                    # —— TC：bit 速率 —— #
                    bps, cr = self._lookup_mcs(snr_trsd)
                    Rm = float(bps) * float(cr) * float(self.N_ss_tr_sd) * float(self.N_sub_tr_sd) / max(T_sym, 1e-9)
                    C_bits = Rm * float(self.delta_T)
                    cap_imgs = float(C_bits // float(max(self.size_enc_img_bits, 1.0)))
        cap_arr = np.array([cap_imgs], dtype=np.float32)

        # --------- 3) 激活 SD id_norm（1 维） ---------
        active_id_norm = (float(active) / float(K - 1)) if K > 1 else 0.0
        active_id_arr = np.array([active_id_norm], dtype=np.float32)

        # --------- 4) 激活 SD 的队列：(1+2M) 维 ---------
        qsd_active = self.Q_sd[active].astype(np.float32, copy=False) if K > 0 else np.zeros(M, dtype=np.float32)
        sd_pairs = np.stack([ds_id_norm, qsd_active], axis=1).reshape(-1)  # (2M,)
        sd_block = np.concatenate([np.array([active_id_norm], dtype=np.float32), sd_pairs], axis=0)  # (1+2M,)

        # --------- 5) 激活 SD 的 SD→DS SNR + 阈值：(M+1) 维 ---------
        if K > 0:
            snr_row = self._ensure_snr_cache_sd_ds()[active].astype(np.float32, copy=False)
        else:
            snr_row = np.zeros(M, dtype=np.float32)

        if getattr(self, "norm_obs", False):
            lo = float(getattr(self, "snr_norm_min", -5.0))
            hi = float(getattr(self, "snr_norm_max", 35.0))
            snr_row = ((np.clip(snr_row, lo, hi) - lo) / max(hi - lo, 1e-6)).astype(np.float32)
            thr_val = (float(self.snr_thr_sd_ds) - lo) / max(hi - lo, 1e-6)
        else:
            thr_val = float(self.snr_thr_sd_ds)

        thr = np.array([thr_val], dtype=np.float32)
        snr_block = np.concatenate([snr_row, thr], axis=0)  # (M+1,)

        # --------- 6) 新增：mask_has_data（M） ---------
        mask_has_data = (self.Q_tr > 0).astype(np.float32)

        # --------- 7) 新增：qtr_norm（M） ---------
        denom = 1.0 + float(self.Q_tr.sum())
        qtr = self.Q_tr.astype(np.float32)
        den = np.asarray(denom, dtype=np.float32)
        qtr_norm = np.zeros_like(qtr, dtype=np.float32)
        np.divide(qtr, den, out=qtr_norm, where=den > 0)
        # ===== 拼接并返回 =====
        return np.concatenate(
            [tr_pairs, cap_arr, active_id_arr, sd_block, snr_block, mask_has_data, qtr_norm],
            axis=0
        ).astype(np.float32)

    def _build_masks_for_sd(self, k: int):
        snr_row = self._ensure_snr_cache_sd_ds()[k]
        conn_mask = (snr_row >= float(self.snr_thr_sd_ds))
        data_mask = (self.Q_sd[k] > 0)
        valid_mask = conn_mask & data_mask
        valid_idx = np.where(valid_mask)[0]

        if getattr(self, "sd_ds_pick_semantics", "global") == "local+noop_last":
            # 局部语义：前 len(valid_idx) 是“valid_set 的局部下标”，第 len(valid_idx) 位是 NOOP，其余屏蔽
            L = int(valid_idx.size)
            mask_conn_local = np.zeros(self.M + 1, dtype=np.float32)
            if L > 0:
                mask_conn_local[:L] = 1.0
            mask_conn_local[L] = 1.0  # NOOP
            mask_fly = np.ones(self.n_fly, dtype=np.float32)
            if L == 0:
                mask_fly[:] = 0.0
                mask_fly[self.n_fly - 1] = 1.0  # 仅允许满飞
            return mask_conn_local, mask_fly
        else:
            # 兼容原始 global 语义
            mask_conn_global = np.zeros(self.M + 1, dtype=np.float32)
            mask_conn_global[:self.M] = valid_mask.astype(np.float32)
            mask_conn_global[self.M] = 1.0  # NOOP 永远允许
            mask_fly = np.ones(self.n_fly, dtype=np.float32)
            if not valid_mask.any():
                mask_fly[:] = 0.0
                mask_fly[self.n_fly - 1] = 1.0
            return mask_conn_global, mask_fly

    def _obs_sd(self, k: int):
        """
        SD 侧观测（原5块） + 两个 action mask：
          1) id_norm（1）
          2) 与其他 SD 的欧氏距离（K-1）
          3) 与所有 DS 的 [平面距离, SNR] 并追加 SNR 阈值 (2M+1)
          4) SD 内部队列：[ds_id_norm, Q_sd[k,m]] × M -> 2M
          5) 所有 DS 的 [ds_id_norm, SP[m]] × M -> 2M
          6) mask_conn_global/local: (M+1)  —— 由 sd_ds_pick_semantics 决定
          7) mask_fly: (n_fly)
        返回：拼接后的观测向量
        """
        M, K = int(self.M), int(self.K)

        # 1) id_norm
        id_norm = (float(k) / float(K - 1)) if K > 1 else 0.0
        id_arr = np.array([id_norm], dtype=np.float32)

        # 2) 与其他 SD 的欧氏距离
        if K > 1:
            diffs = self.P_sd[:, :2] - self.P_sd[k, :2][None, :]
            dists_all = np.linalg.norm(diffs, axis=1).astype(np.float32)
            other_idx = [i for i in range(K) if i != k]
            sd_dist_vec = dists_all[other_idx]
        else:
            sd_dist_vec = np.zeros(0, dtype=np.float32)

        # 公共 ds_id_norm
        ds_id_norm = (np.arange(M, dtype=np.float32) / float(M - 1)).astype(np.float32) if M > 1 else np.zeros(
            M, dtype=np.float32
        )

        # 3) [距离, SNR] + 阈值
        if M > 0:
            se_xy = self.P_sd[k, :2]
            ds_xy = self.ds_pos[:, :2]
            dist_sd_ds = np.linalg.norm(ds_xy - se_xy[None, :], axis=1).astype(np.float32)
            snr_row = self._ensure_snr_cache_sd_ds()[k].astype(np.float32, copy=False)

            if getattr(self, "norm_obs", False):
                lo = float(getattr(self, "snr_norm_min", -5.0))
                hi = float(getattr(self, "snr_norm_max", 35.0))
                # 距离：按地图尺度归一化到 [0,1]
                dist_sd_ds = np.clip(dist_sd_ds / max(self.map_size, 1.0), 0.0, 1.0).astype(np.float32)
                # SNR(dB)：线性映射到 [0,1]
                snr_row = ((np.clip(snr_row, lo, hi) - lo) / max(hi - lo, 1e-6)).astype(np.float32)
                thr_val = (float(self.snr_thr_sd_ds) - lo) / max(hi - lo, 1e-6)
            else:
                thr_val = float(self.snr_thr_sd_ds)

            # 归一化完成后再组装
            pair_dist_snr = np.empty(2 * M, dtype=np.float32)
            pair_dist_snr[0:2 * M:2] = dist_sd_ds
            pair_dist_snr[1:2 * M:2] = snr_row
            thr_arr = np.array([thr_val], dtype=np.float32)
        else:
            pair_dist_snr = np.zeros(0, dtype=np.float32)
            thr_arr = np.array([float(self.snr_thr_sd_ds)], dtype=np.float32)

        # 4) [ds_id_norm, Q_sd[k,m]]
        if M > 0:
            qsd_k = self.Q_sd[k].astype(np.float32, copy=False)
            pair_id_q = np.empty(2 * M, dtype=np.float32)
            pair_id_q[0:2 * M:2] = ds_id_norm
            pair_id_q[1:2 * M:2] = qsd_k
        else:
            pair_id_q = np.zeros(0, dtype=np.float32)

        # 5) [ds_id_norm, SP[m]]
        if M > 0:
            SP = self.SP.astype(np.float32, copy=False)
            pair_id_sp = np.empty(2 * M, dtype=np.float32)
            pair_id_sp[0:2 * M:2] = ds_id_norm
            pair_id_sp[1:2 * M:2] = SP
        else:
            pair_id_sp = np.zeros(0, dtype=np.float32)

        # 基础观测
        base_obs = np.concatenate(
            [id_arr, sd_dist_vec, pair_dist_snr, thr_arr, pair_id_q, pair_id_sp],
            axis=0
        ).astype(np.float32)

        # 统一生成 mask（会根据 sd_ds_pick_semantics 返回 global 或 local+noop_last）
        mask_conn, mask_fly = self._build_masks_for_sd(k)

        return np.concatenate(
            [base_obs, mask_conn.astype(np.float32), mask_fly.astype(np.float32)],
            axis=0
        ).astype(np.float32)

    def get_global_state(self):
        """
        全局观测（critic 输入），按用户要求的三大块顺序拼接：
          1) Tr-UAV 信息：
             - 位置 (x,y,z) -> 3
             - Tr 内部队列：[DS_id_norm(长度 M), Q_tr[m](长度 M)] -> 2M
          2) 所有 SD-UAV 信息：
             - SD_id_norm 列表（长度 K） -> K
             - 所有 SD 位置 (x,y,z) 展平 -> 3K
             - 每个 SD 的内部队列，逐台展开：
                 对该 SD： [DS_id_norm, Q_sd[k,m]] × M -> 2M
                 K 台合计 -> 2MK
          3) 所有 DS 信息：
             - DS_id_norm（长度 M） -> M
             - DS 位置 (x,y,z=0) 展平 -> 3M
             - SP[m]（长度 M） -> M
        维度总计： (3 + 2M) + (K + 3K + 2MK) + (M + 3M + M) = 3 + 7M + 4K + 2MK
        """
        M = int(self.M)
        K = int(self.K)

        # ---------- 归一化 ID ----------
        if M > 1:
            ds_id_norm = (np.arange(M, dtype=np.float32) / float(M - 1)).astype(np.float32)
        else:
            ds_id_norm = np.zeros(M, dtype=np.float32)

        if K > 1:
            sd_id_norm = (np.arange(K, dtype=np.float32) / float(K - 1)).astype(np.float32)
        else:
            sd_id_norm = np.zeros(K, dtype=np.float32)

        # ---------- 1) Tr-UAV ----------
        # 位置：tsd_pos 至少 3 维 (x,y,z)
        tr_pos = np.array(self.tsd_pos, dtype=np.float32).reshape(-1)
        if tr_pos.size < 3:
            tr_pos = np.pad(tr_pos, (0, 3 - tr_pos.size), mode="constant")
        else:
            tr_pos = tr_pos[:3]
        # 队列：DS_id_norm + Q_tr
        q_tr = self.Q_tr.astype(np.float32, copy=False) if M > 0 else np.zeros(0, dtype=np.float32)
        block_tr = np.concatenate([tr_pos, ds_id_norm, q_tr], axis=0)  # 3 + 2M

        # ---------- 2) 所有 SD-UAV ----------
        # SD id 列表（K）
        sd_id_vec = sd_id_norm  # 长度 K

        # 位置：P_sd[K,3] 展平为 (3K,)
        if K > 0:
            sd_pos = np.array(self.P_sd, dtype=np.float32)
            if sd_pos.shape[1] < 3:
                # 若只给了二维，补 z=0
                pad = np.zeros((K, 3 - sd_pos.shape[1]), dtype=np.float32)
                sd_pos = np.concatenate([sd_pos, pad], axis=1)
            sd_pos = sd_pos[:, :3].reshape(-1)  # 3K
        else:
            sd_pos = np.zeros(0, dtype=np.float32)

        # 每台 SD 的内部队列：按台循环拼接 (K * 2M)
        if K > 0 and M > 0:
            sd_q_list = []
            for k in range(K):
                qsd_k = self.Q_sd[k].astype(np.float32, copy=False)
                pair_id_q = np.empty(2 * M, dtype=np.float32)
                pair_id_q[0::2] = ds_id_norm
                pair_id_q[1::2] = qsd_k
                sd_q_list.append(pair_id_q)
            sd_q_flat = np.concatenate(sd_q_list, axis=0)  # 长度 2MK
        else:
            sd_q_flat = np.zeros(0, dtype=np.float32)

        block_sd = np.concatenate([sd_id_vec, sd_pos, sd_q_flat], axis=0)  # K + 3K + 2MK

        # ---------- 3) 所有 DS ----------
        # 位置：ds_pos[M,3]，强制 z=0
        if M > 0:
            ds_pos = np.array(self.ds_pos, dtype=np.float32)
            if ds_pos.shape[1] < 3:
                pad = np.zeros((M, 3 - ds_pos.shape[1]), dtype=np.float32)
                ds_pos = np.concatenate([ds_pos, pad], axis=1)
            ds_pos = ds_pos[:, :3]
            ds_pos[:, 2] = 0.0  # z=0
            ds_pos_flat = ds_pos.reshape(-1)  # 3M
            sp = self.SP.astype(np.float32, copy=False)  # M
        else:
            ds_pos_flat = np.zeros(0, dtype=np.float32)
            sp = np.zeros(0, dtype=np.float32)

        block_ds = np.concatenate([ds_id_norm, ds_pos_flat, sp], axis=0)  # M + 3M + M = 5M

        # ---------- 拼接全部并返回 ----------
        share_obs = np.concatenate([block_tr, block_sd, block_ds], axis=0).astype(np.float32)
        return share_obs

    def _effective_distance_J(self, pos_k: np.ndarray) -> float:
        """
        计算位置 pos_k 相对所有 DS 的“SP 加权有效距离”：
            J(pos) = min_m  d(pos, DS_m) / (1 + alpha_sp * hat_SP_m)
        其中 hat_SP_m = clip(SP_m / sp_norm_max, 0, 1)
        """
        # 3D 距离（与 _snr_sd_ds_single 的 L 一致：含高度差）
        d_all = np.linalg.norm(self.ds_pos - pos_k[None, :], axis=1)  # (M,)
        # 归一化后的饥饿度
        sp_hat = np.minimum(self.SP / max(1.0, float(self.sp_norm_max)), 1.0)
        denom = 1.0 + float(self.alpha_sp) * sp_hat
        eff = d_all / denom
        return float(np.min(eff)) if eff.size > 0 else 0.0

    def _build_info_all(self, active: int, rx_total: int, delivered_total: int, avg_sp: float,
                        conn_sets: List[np.ndarray], valid_sets: List[np.ndarray],
                        ds_exec: np.ndarray):
        """
        汇总本槽信息给 info（返回顺序：[BS_info] + [SD_info_k]*K）。
        新增：
          - reward_components：把奖励分段（带符号）打包，便于训练端记录/可视化。
            其中：
              deliver  = +w1_deliver * delivered_total
              sp       = -w2_sp * avg_sp
              collision= -collision_penalty      （从 self._last_collision_penalty 读取，缺省 0）
              move     = +move_bonus              （从 self._last_move_bonus 读取，缺省 0）
              total    = deliver + sp + collision + move
        """
        # ---- BS 侧概览 ----
        info_bs = {
            'active_tr_rx': int(active),
            'rx_total_active': int(rx_total),
            'delivered_total': int(delivered_total),
            'RS_free': int(self.RS_free),
            'Q_tr_sum': int(self.Q_tr.sum()),
            'avg_SP': float(avg_sp),
            "bs_slot_enc_alloc": self.bs_enc_last.astype(int),
            "bs_cum_enc_started": self.cum_enc_started.astype(int),
            "tr_slot_tx_to_sd_active": self.tr_tx_last[active].astype(int),
            "tr_cum_tx_to_sd_active": self.cum_tr_to_sd[active].astype(int),
        }

        # 读取（或回退）奖励分段
        # 还原本槽 per-DS 的交付向量
        delivered_vec = self.ds_rx_last_from_sd.sum(axis=0).astype(np.float32)  # 形状 (M,)
        # 与 step() 同步的连续服务衰减系数
        decay_vec = np.power(float(self.decay_gamma),
                             np.maximum(self.ds_streak - 1, 0))
        deliver_reward = float(self.w1_deliver *
                               (float((decay_vec * delivered_vec).sum()) / max(1, self.M)))
        sp_penalty = float(self.w2_sp * (self.SP.sum() / max(1, self.M)))
        collision_penalty = float(getattr(self, "_last_collision_penalty", 0.0))
        move_bonus = float(getattr(self, "_last_move_bonus", 0.0))

        shared_reward = float(deliver_reward - sp_penalty - collision_penalty + move_bonus)

        info_bs["reward_components"] = {
            "deliver": float(deliver_reward),
            "sp": -float(sp_penalty),
            "collision": -float(collision_penalty),
            "move": float(move_bonus),

            # —— 细分项（保持你现在的口径：未加权靠近/远离量与其加权后的加减项）——
            "move_pos": float(getattr(self, "_last_move_pos", 0.0)),
            "move_neg": -float(getattr(self, "_last_move_neg", 0.0)),
            "move_bonus_pos": float(getattr(self, "_last_move_bonus_pos",
                                            getattr(self, "_last_move_pos", 0.0) * float(
                                                getattr(self, "w_move_pos", 1.0)))),
            "move_penalty_away": -float(getattr(self, "_last_move_penalty_away",
                                                getattr(self, "_last_move_neg", 0.0) * float(
                                                    getattr(self, "w_move_away", 1.0)))),

            "total": float(shared_reward),
        }

        # === 兼容：Tr->SD（active 行）的容量/积压统计 ===
        if hasattr(self, "_last_pull_log") and isinstance(self._last_pull_log, dict):
            info_bs["tr_cap_enc_imgs_theory"] = int(self._last_pull_log.get("cap_enc_imgs_theory", 0))
            info_bs["tr_backlog_blocks_selected"] = int(self._last_pull_log.get("backlog_blocks_selected", 0))

        # ---- 各 SD 侧明细 ----
        infos_sd = []
        for k in range(self.K):
            mask = np.zeros(self.M + 1, dtype=int)
            mask[self.M] = 1
            mask[valid_sets[k]] = 1
            mask_conn_global, mask_fly = self._build_masks_for_sd(k)
            infos_sd.append({
                'is_active': (k == active),
                'Q_sd_sum_k': int(self.Q_sd[k].sum()),
                'conn_set_size': int(conn_sets[k].size),
                'valid_set_size': int(valid_sets[k].size),
                'mask_valid_ds': mask.astype(int),
                'ds_chosen_exec': int(ds_exec[k]),
                'sd_slot_q_snapshot_row': self.Q_sd[k].astype(int),  # 本槽末快照
                'sd_slot_tx_to_ds_row': self.sd2ds_sent_last[k].astype(int),  # 本槽发出
                'sd_cum_tx_to_ds_row': self.cum_sd2ds[k].astype(int),  # 累计发出
                'sd_cum_rx_from_tr_row': self.cum_tr_to_sd[k].astype(int),  # 累计收到（Tr->SD）
                'mask_conn_global': mask_conn_global.astype(int),
                'mask_fly': mask_fly.astype(int),
            })

        return [info_bs] + infos_sd

    # ============================== 动作与分配 ============================ #
    def _unpack_actions(self, actions):
        """
        解析上层传入的动作，返回 (bs_tr_weights, sd_actions)。
        支持：
          - dict:
              {'sd': List[np.ndarray(len=3)] , 'bs_tr_weights': array-like (M,)}
            兼容：若缺少 'bs_tr_weights'，并且 'bs' 恰好是一维数值且长度=M，则把 'bs' 当作连续权重；
                  否则权重置为全 0。
          - list/tuple: len = K+1
              第 0 个若是一维数值向量且长度=M，则视为连续权重；否则权重置 0；
              其后 K 个分别是 SD 的 (dir, fly, ds)。
          - numpy 对象数组：同上（size = K+1）。
        返回：
          bs_tr_weights: np.ndarray, 形状 (M,) 的非负 float32
          sd_actions   : List[np.ndarray]，长度 K，每个形状 (3,) 的 int
        """
        import numpy as np

        M = int(self.M)
        K = int(self.K)

        def _norm_weights(w):
            w = np.asarray(w, dtype=np.float32).reshape(-1)
            if w.size < M:
                w = np.pad(w, (0, M - w.size), mode="constant")
            elif w.size > M:
                w = w[:M]
            return np.clip(w, 0.0, np.inf).astype(np.float32)

        def _norm_sd(a):
            need = 3
            v = np.asarray(a, dtype=np.int64).reshape(-1)
            if v.size < need:
                v = np.pad(v, (0, need - v.size), mode="constant")
            elif v.size > need:
                v = v[:need]
            return v

        # ---------- dict 路径 ----------
        if isinstance(actions, dict):
            # SD 必须提供
            sd_raw = actions.get('sd', None)
            if not isinstance(sd_raw, (list, tuple)) or len(sd_raw) != K:
                raise ValueError("dict['sd'] 必须是长度 K 的列表/元组")
            sd_actions = [_norm_sd(a) for a in sd_raw]

            # 连续权重：优先 'bs_tr_weights'
            if 'bs_tr_weights' in actions:
                bs_tr_weights = _norm_weights(actions['bs_tr_weights'])
            else:
                # 兼容性回退：若 'bs' 是 1D 且 len=M 的数值向量，则暂当连续权重；否则置 0
                bsv = actions.get('bs', None)
                use_bs_as_weights = False
                if bsv is not None:
                    arr = np.asarray(bsv)
                    if np.issubdtype(arr.dtype, np.number) and arr.ndim == 1 and arr.size == M:
                        use_bs_as_weights = True
                bs_tr_weights = _norm_weights(bsv) if use_bs_as_weights else np.zeros((M,), dtype=np.float32)

            return bs_tr_weights, sd_actions

        # ---------- list/tuple 路径（len = K+1） ----------
        if isinstance(actions, (list, tuple)) and len(actions) == (K + 1):
            a0 = actions[0]
            arr0 = np.asarray(a0)
            is_numeric = np.issubdtype(arr0.dtype, np.number)
            if is_numeric and arr0.ndim == 1 and arr0.size == M:
                bs_tr_weights = _norm_weights(arr0)
            else:
                bs_tr_weights = np.zeros((M,), dtype=np.float32)
            sd_actions = [_norm_sd(actions[i + 1]) for i in range(K)]
            return bs_tr_weights, sd_actions

        # ---------- numpy 对象数组（size = K+1） ----------
        if isinstance(actions, np.ndarray) and actions.dtype == object and actions.size == (K + 1):
            a0 = actions[0]
            arr0 = np.asarray(a0)
            is_numeric = np.issubdtype(arr0.dtype, np.number)
            if is_numeric and arr0.ndim == 1 and arr0.size == M:
                bs_tr_weights = _norm_weights(arr0)
            else:
                bs_tr_weights = np.zeros((M,), dtype=np.float32)
            sd_actions = [_norm_sd(actions[i + 1]) for i in range(K)]
            return bs_tr_weights, sd_actions

        raise TypeError(f"_unpack_actions 期望 dict 或 len=K+1 的 list/tuple/ndarray，当前 type={type(actions)}")

    def _map_sd_ds_pick(self, action_pick: int, valid_idx: np.ndarray, mode: str) -> int:
        """
        把策略给的 ds_idx 映射成【全局 DS id】。
        返回 self.M 表示 NOOP/跳过；返回 [0..M-1] 表示一个有效的全局 DS。
        """
        # 没有可连可发的目标：只能 NOOP
        if valid_idx.size == 0:
            return self.M

        if mode == 'global':
            # 将 valid_idx 转为布尔掩码，避免频繁构造 Python set
            if isinstance(valid_idx, np.ndarray) and valid_idx.dtype == np.bool_ and valid_idx.size == self.M:
                # 已经是掩码
                valid_mask = valid_idx
            else:
                valid_mask = np.zeros(self.M, dtype=np.bool_)
                valid_mask[valid_idx] = True

            ap = int(action_pick)
            if 0 <= ap < self.M and valid_mask[ap]:
                return ap
            return self.M

        if mode == 'local+noop_last':
            # 策略给“valid 的下标”，且【最后一格】代表 NOOP
            # 下标范围：[0 .. len(valid_idx)]；等于 len(valid_idx) => NOOP
            if action_pick == len(valid_idx):
                return self.M
            if 0 <= action_pick < len(valid_idx):
                return int(valid_idx[action_pick])
            return self.M  # 越界一律当 NOOP

        # 兜底
        return self.M

    def _parse_sd_actions(self, sd_actions: List[np.ndarray]):
        dir_idx = np.zeros(self.K, dtype=int)
        fly_idx = np.zeros(self.K, dtype=int)
        ds_idx = np.zeros(self.K, dtype=int)
        for k in range(self.K):
            dir_idx[k] = int(np.clip(sd_actions[k][0], 0, self.n_dir - 1))
            fly_idx[k] = int(np.clip(sd_actions[k][1], 0, self.n_fly - 1))
            ds_idx[k] = int(np.clip(sd_actions[k][2], 0, self.M))  # 0..M，M表示skip
        return dir_idx, fly_idx, ds_idx

    def _int_alloc_largest_remainder(self, weights: np.ndarray, total: int) -> np.ndarray:
        total = int(total)
        w = np.asarray(weights, dtype=np.float64)
        w[~np.isfinite(w)] = 0.0
        w[w < 0] = 0.0

        n = w.size
        alloc = np.zeros(n, dtype=np.int64)
        if total <= 0 or n == 0:
            return alloc.astype(int)

        nz = (w > 0)
        if not nz.any():
            q, r = divmod(total, n)
            alloc[:] = q
            if r > 0:
                order = np.arange(n)
                alloc[order[:r]] += 1
            return alloc.astype(int)

        w_nz = w[nz]
        p_nz = w_nz / w_nz.sum()
        exact_nz = p_nz * total
        base_nz = np.floor(exact_nz).astype(np.int64)
        r = int(total - base_nz.sum())

        if r > 0:
            frac_nz = exact_nz - base_nz
            idx_nz = np.arange(frac_nz.size)
            order_nz = np.lexsort((idx_nz, -w_nz, -frac_nz))
            base_nz[order_nz[:r]] += 1

        alloc[nz] = base_nz
        return alloc.astype(int)

    def _apply_bs_encoding(self, enc_picks: List[int], n_assign: int) -> None:
        n_assign = int(n_assign)
        if n_assign <= 0:
            return

        counts = np.zeros(self.M, dtype=np.float64)
        for p in enc_picks:
            m = int(np.clip(p, 0, self.M - 1))
            counts[m] += 1.0

        if counts.sum() <= 0:
            alloc = self._int_alloc_largest_remainder(np.ones(self.M, dtype=np.float64), n_assign)
        else:
            alloc = np.zeros(self.M, dtype=int)
            mask = counts > 0
            alloc_sub = self._int_alloc_largest_remainder(counts[mask], n_assign)
            alloc[mask] = alloc_sub

        diff = int(n_assign - int(alloc.sum()))
        if diff != 0:
            order = np.lexsort((np.arange(self.M), -counts))
            if diff > 0:
                for m in order[:diff]:
                    alloc[m] += 1
            else:
                for m in order[:(-diff)]:
                    if alloc[m] > 0:
                        alloc[m] -= 1

        # SC：随机；TC：固定 1s
        if self.comm_mode == "TC":
            tao = float(self.tc_enc_delay_s)
        else:
            tao = float(self.rng.uniform(self.tao_min, self.tao_max))
        tau_slot = int(math.ceil(tao / self.delta_T))
        t_rel = self.t + tau_slot

        for m in range(self.M):
            q = int(alloc[m])
            if q <= 0:
                continue
            for _ in range(q):
                self.finish_events.append((t_rel, int(m)))

        self._last_enc_log = {
            "RS_free_before": int(n_assign),
            "alloc": alloc.astype(int),
            "tao_sec": float(round(tao, 1)),
            "tau_slot": int(tau_slot),
            "slot_release": int(t_rel),
            "alloc_sum": int(alloc.sum())
        }

        self.bs_enc_last = alloc.astype(int)
        self.cum_enc_started += self.bs_enc_last

    def _apply_comm_mode(self):
        """
        根据 self.comm_mode 配置显示用的“编码后尺寸”。
        注意：两种模式下编码大小一致（沿用压缩比）；区别体现在：
          - TC：编码返还时延固定为 tc_enc_delay_s（在 _apply_bs_encoding / _apply_bs_encoding_auto_rr 中处理）；
          - SC：Tr→SD 的容量以“符号数 / Upsilon_sym_per_img”计量（在 step() / _obs_bs() 中处理）。
        """
        self.size_enc_img_bits = float(getattr(self, "size_enc_img_bits", self.cmp_ratio * self.S_img))
        self.enc_img_size_Mb = float(round(self.size_enc_img_bits / self._MBIT, 3))

    # ============================== SNR 时隙缓存工具（新增） ========================== #
    def _ensure_snr_cache_sd_ds(self) -> np.ndarray:
        """
        按时隙冻结 SD->DS 的 SNR：
        - 第一次在某时隙 self.t 被调用时，用当前 self.P_sd 作为本时隙的冻结位置，
          计算出 (K,M) 的 SNR(dB) 并缓存；
        - 同一时隙内后续任意位置调用，均直接复用缓存；
        - 下一时隙自动失效并重算。
        """
        if self._snr_cache_sd_ds is None or self._snr_cache_slot_sd_ds != self.t:
            arr = np.zeros((self.K, self.M), dtype=np.float32)
            for k in range(self.K):
                for m in range(self.M):
                    arr[k, m] = float(self._snr_sd_ds_single(self.P_sd[k], m))
            self._snr_cache_sd_ds = arr
            self._snr_cache_slot_sd_ds = self.t
        return self._snr_cache_sd_ds

    def _apply_bs_encoding_auto_rr(self, n_assign: int) -> None:
        """
        把本槽可用的编码资源 n_assign 用“轮询”平均分给所有 DS：
        第 1 个 eRU 给 ds=ptr, 下一个给 (ptr+1)%M ...，直到用完；跨槽保留轮询起点。
        产生的事件在同一释放时刻 t_rel 返还到 Q_tr。
        """
        n_assign = int(n_assign)
        if n_assign <= 0 or self.M <= 0:
            self._last_enc_log = {"RS_free_before": int(n_assign), "alloc": np.zeros(self.M, dtype=int)}
            return

        counts = np.zeros(self.M, dtype=int)
        ptr = int(self._bs_rr_ptr_enc) % self.M

        # 一个批次共用一个延迟：SC 随机；TC 固定 1s
        if self.comm_mode == "TC":
            tao = float(self.tc_enc_delay_s)
        else:
            tao = float(self.rng.uniform(self.tao_min, self.tao_max))
        tau_slot = int(math.ceil(tao / self.delta_T))
        t_rel = self.t + tau_slot

        for _ in range(n_assign):
            counts[ptr] += 1
            ptr = (ptr + 1) % self.M

        # 登记返还事件
        for m in range(self.M):
            q = int(counts[m])
            for _ in range(q):
                self.finish_events.append((t_rel, int(m)))

        # 滚动统计 & 指针推进
        self.bs_enc_last = counts.astype(int)
        self.cum_enc_started += self.bs_enc_last
        self._bs_rr_ptr_enc = ptr

        self._last_enc_log = {
            "mode": "auto_rr",
            "RS_free_before": int(n_assign),
            "alloc": self.bs_enc_last.astype(int),
            "alloc_sum": int(self.bs_enc_last.sum()),
            "tao_sec": float(round(tao, 1)),
            "tau_slot": int(tau_slot),
            "slot_release": int(t_rel),
        }

    def _apply_bs_encoding_tc_immediate(self, n_assign: int, enc_picks: List[int] = None) -> None:
        """
        （预留）实验函数：当前未使用。
        若需要“TC 立即入队”的行为，可启用该函数；但默认逻辑采用“固定 1s 编码时延 + 事件返还”。
        """
        n_assign = int(n_assign)
        if n_assign <= 0 or self.M <= 0:
            self._last_enc_log = {"mode": "tc_immediate", "RS_free_before": int(n_assign),
                                  "alloc": np.zeros(self.M, dtype=int)}
            return

        alloc = np.zeros(self.M, dtype=int)

        if self.use_bs_agent and enc_picks is not None:
            # 复用与 _apply_bs_encoding 相同的“按票分配”的策略，但不登记事件
            counts = np.zeros(self.M, dtype=np.float64)
            for p in enc_picks:
                m = int(np.clip(p, 0, self.M - 1))
                counts[m] += 1.0
            if counts.sum() <= 0:
                alloc = self._int_alloc_largest_remainder(np.ones(self.M, dtype=np.float64), n_assign)
            else:
                alloc = np.zeros(self.M, dtype=int)
                mask = counts > 0
                alloc_sub = self._int_alloc_largest_remainder(counts[mask], n_assign)
                alloc[mask] = alloc_sub
            # 填平误差
            diff = int(n_assign - int(alloc.sum()))
            if diff != 0:
                order = np.lexsort((np.arange(self.M), -counts))
                if diff > 0:
                    for m in order[:diff]:
                        alloc[m] += 1
                else:
                    for m in order[:(-diff)]:
                        if alloc[m] > 0:
                            alloc[m] -= 1
        else:
            # 自动轮询：把 n_assign 平均发给 DS（不登记事件，直接入 Q_tr）
            ptr = int(self._bs_rr_ptr_enc) % self.M
            for _ in range(n_assign):
                alloc[ptr] += 1
                ptr = (ptr + 1) % self.M
            self._bs_rr_ptr_enc = ptr

        # 关键区别：不登记 finish_events，直接入队
        self.Q_tr += alloc

        # 统计与日志
        self.bs_enc_last = alloc.astype(int)
        self.cum_enc_started += self.bs_enc_last

        # 由于“编码时间=0”，eRU 立即释放：RS_free 保持不变（不置 0）
        self._last_enc_log = {
            "mode": "tc_immediate",
            "RS_free_before": int(n_assign),
            "alloc": self.bs_enc_last.astype(int),
            "alloc_sum": int(self.bs_enc_last.sum()),
            "note": "TC: encode time=0, directly filled Q_tr; no events"
        }

    def _auto_rr_tr_pull(self, active: int, cap_enc_imgs: int) -> np.ndarray:
        """
        把本槽 Tr->active_SD 的传输容量 cap_enc_imgs（单位：压缩整图，匹配 Q_tr 的计数）
        用“轮询”在所有有 backlog 的 DS 之间平均分配：
            例：cap=40 且 20 个 DS 有货 -> 先给每个 DS 1 张；再来一轮...
        返回 moved[M]：每个 DS 本槽从 Tr 拉取到 active SD 的张数。
        """
        moved = np.zeros(self.M, dtype=int)
        cap = int(cap_enc_imgs)
        if cap <= 0 or self.M <= 0:
            return moved

        ptr = int(self._bs_rr_ptr_trsd) % self.M
        remain = min(cap, int(self.Q_tr.sum()))

        while remain > 0:
            # 从当前指针起，找到下一个有货的 DS
            found = False
            for _ in range(self.M):
                if self.Q_tr[ptr] > 0:
                    # 分配 1 张
                    self.Q_tr[ptr] -= 1
                    self.Q_sd[active, ptr] += 1
                    moved[ptr] += 1
                    remain -= 1
                    ptr = (ptr + 1) % self.M
                    found = True
                    break
                ptr = (ptr + 1) % self.M
            if not found:
                break  # 没有 DS 有货，提前结束

        self._bs_rr_ptr_trsd = ptr
        return moved

    def _get_snr_tr_sd_slot(self, k: int, L: float) -> float:
        """
        按时隙冻结 Tr->SD 的 SNR：
        - 对同一时隙 self.t、同一 SD k，第一次调用用传入的 L 计算并缓存；
        - 本时隙内后续对同一 k 的调用直接返回第一次的结果（即使 L 不同，也以首次为准）；
        - 下一时隙自动失效并重算。
        """
        if self._snr_cache_tr_sd is None or self._snr_cache_slot_tr_sd != self.t:
            self._snr_cache_tr_sd = np.full(self.K, np.nan, dtype=np.float32)
            self._snr_cache_slot_tr_sd = self.t
        if not np.isfinite(self._snr_cache_tr_sd[k]):
            self._snr_cache_tr_sd[k] = float(self._snr_tr_sd(float(L)))
        return float(self._snr_cache_tr_sd[k])

    # ============================== 物理层/信道 ========================== #
    def _lookup_mcs(self, snr_db: float) -> Tuple[float, float]:
        table = [
            (2, 1, 1 / 2), (5, 2, 1 / 2), (9, 2, 3 / 4), (11, 4, 1 / 2), (15, 4, 3 / 4),
            (18, 6, 2 / 3), (20, 6, 3 / 4), (25, 6, 5 / 6), (29, 8, 3 / 4), (31, 8, 5 / 6),
        ]
        best = (1.0, 0.5)
        for thr, bits, rate in table:
            if snr_db >= thr:
                best = (float(bits), float(rate))
        return best

    def _path_loss_db(self, L: float, freq: float, gamma: float, shadow_std: float, C_extra: float = 0.0) -> float:
        L_eff = max(float(L), 1.0)
        pl0 = 20.0 * math.log10(4.0 * math.pi * float(freq) / 3e8)
        shadow = float(self.rng.normal(0.0, shadow_std))
        return float(pl0 + 10.0 * float(gamma) * math.log10(L_eff) + float(C_extra) + shadow)

    def _snr_tr_sd(self, L: float) -> float:
        PL = self._path_loss_db(float(L), float(self.f_tr_sd), float(self.gamma_tr_sd), float(self.sigma_tr_sd), 0.0)
        noise_db = -174.0 + 10.0 * math.log10(float(self.B_tr_sd)) + float(self.F_tr_sd)
        return float(self.W_tr + self.G_tx_tr + self.G_rx_sd - PL - noise_db)

    def _snr_sd_ds_single(self, se_pos: np.ndarray, m: int) -> float:
        # 距离
        L = float(np.linalg.norm(se_pos - self.ds_pos[m]))
        ratio = float(self.z_sd) / max(L, 1.0)
        theta = math.asin(max(-1.0, min(1.0, ratio)))
        P_los = 1.0 / (
                1.0 + float(self.a_los) * math.exp(-float(self.b_los) * (math.degrees(theta) - float(self.a_los))))

        pl0 = 20.0 * math.log10(4.0 * math.pi * float(self.f_sd_ds) / 3e8)
        PL_los = pl0 + 10.0 * float(self.gamma_sd_ds_los) * math.log10(max(L, 1.0)) + float(self.C_los) + float(
            self.rng.normal(0.0, float(self.sigma_sd_ds)))
        PL_nlos = pl0 + 10.0 * float(self.gamma_sd_ds_nlos) * math.log10(max(L, 1.0)) + float(self.C_nlos) + float(
            self.rng.normal(0.0, float(self.sigma_sd_ds)))
        PL = float(P_los * PL_los + (1.0 - P_los) * PL_nlos)

        noise_db = -174.0 + 10.0 * math.log10(float(self.B_sd_ds)) + float(self.F_sd_ds)
        return float(self.W_sd + self.G_tx_sd + self.G_rx_ds - PL - noise_db)

    # ============================== 位置/生成 ============================ #
    def _place_sd_init_positions(self):
        # ===== DEMO: 硬写入初始位置（self.demo=True 时启用）=====
        if getattr(self, "demo", False):
            # 这里写死你的坐标；支持 [x,y] 或 [x,y,z]，数量≥你要用的 K 个
            coords = np.array([
                [1000.0, 2000, 50.0],  # SD-0
                [1000.0, 1000.0, 50.0],  # SD-1
                [2000.0, 1000.0, 50.0],  # SD-2
            ], dtype=np.float32)

            # 若只给 (x,y)，自动补 z=self.z_sd
            if coords.shape[1] == 2:
                zcol = np.full((coords.shape[0], 1), float(self.z_sd), dtype=np.float32)
                coords = np.concatenate([coords, zcol], axis=1)

            n = int(min(self.K, coords.shape[0]))
            # 限幅到地图范围
            coords[:n, 0] = np.clip(coords[:n, 0], 0.0, self.map_size)
            coords[:n, 1] = np.clip(coords[:n, 1], 0.0, self.map_size)

            # 先填入已提供的 K' 个
            self.P_sd[:n, :3] = coords[:n, :3]

            # 若 K 比硬写多，剩余的按原逻辑补齐
            if self.K > n:
                sd_xy = self.tsd_pos[:2].astype(np.float32)
                cen_xy = self.cluster_center.astype(np.float32)
                midpoint = (sd_xy + cen_xy) / 2.0
                vec = cen_xy - sd_xy
                nrm = float(np.linalg.norm(vec))
                if nrm < 1e-6:
                    u_perp = np.array([0.0, 1.0], dtype=np.float32)
                else:
                    u = vec / nrm
                    u_perp = np.array([-u[1], u[0]], dtype=np.float32)
                spacing = 300.0
                for i, k in enumerate(range(n, self.K)):
                    s = 1.0 if (i % 2) == 0 else -1.0
                    pos = midpoint + s * (1 + i // 2) * spacing * u_perp
                    pos[0] = float(np.clip(pos[0], 0.0, self.map_size))
                    pos[1] = float(np.clip(pos[1], 0.0, self.map_size))
                    self.P_sd[k, 0] = pos[0]
                    self.P_sd[k, 1] = pos[1]
                    self.P_sd[k, 2] = float(self.z_sd)
            else:
                self.P_sd[:, 2] = self.z_sd
            return

        # ===== 非 DEMO：保留你的原始初始化逻辑（以下是原代码，不变）=====
        sd_xy = self.tsd_pos[:2].astype(np.float32)
        cen_xy = self.cluster_center.astype(np.float32)
        midpoint = (sd_xy + cen_xy) / 2.0
        vec = cen_xy - sd_xy
        nrm = float(np.linalg.norm(vec))
        if nrm < 1e-6:
            u_perp = np.array([0.0, 1.0], dtype=np.float32)
        else:
            u = vec / nrm
            u_perp = np.array([-u[1], u[0]], dtype=np.float32)
        spacing = 300.0
        if self.K == 1:
            pos = midpoint
            pos[0] = float(np.clip(pos[0], 0.0, self.map_size))
            pos[1] = float(np.clip(pos[1], 0.0, self.map_size))
            self.P_sd[0, :2] = pos
        else:
            offs = []
            for i in range(self.K):
                n = (i // 2) + 1
                s = 1.0 if (i % 2) == 0 else -1.0
                offs.append(s * n * spacing)
            for k, off in enumerate(offs):
                pos = midpoint + off * u_perp
                pos[0] = float(np.clip(pos[0], 0.0, self.map_size))
                pos[1] = float(np.clip(pos[1], 0.0, self.map_size))
                self.P_sd[k, :2] = pos
        self.P_sd[:, 2] = self.z_sd

    def _gen_ds_positions_around_center(self):
        sigma = 220.0
        min_sep = float(self.ds_min_sep)
        pts = []
        trials = 0
        max_trials = max(5000, 50 * int(self.M))
        relax = 0.85
        floor = 0.6 * min_sep
        while len(pts) < int(self.M) and trials < max_trials:
            trials += 1
            cand = self.rng.normal(loc=self.cluster_center, scale=sigma, size=(2,)).astype(np.float32)
            cand[0] = np.clip(cand[0], 0, self.map_size)
            cand[1] = np.clip(cand[1], 0, self.map_size)
            if not pts:
                pts.append((float(cand[0]), float(cand[1])))
                continue
            diffs = np.array(pts, dtype=np.float32) - cand[None, :]
            d2 = np.sum(diffs * diffs, axis=1)
            if np.all(d2 >= (min_sep * min_sep)):
                pts.append((float(cand[0]), float(cand[1])))
                continue
            if trials >= max_trials and len(pts) < int(self.M) and min_sep > floor:
                min_sep = max(min_sep * relax, floor)
                trials = 0
                max_trials += 50 * int(self.M)
        while len(pts) < int(self.M):
            cands = self.rng.normal(loc=self.cluster_center, scale=sigma, size=(128, 2)).astype(np.float32)
            cands[:, 0] = np.clip(cands[:, 0], 0, self.map_size)
            cands[:, 1] = np.clip(cands[:, 1], 0, self.map_size)
            if pts:
                base = np.array(pts, dtype=np.float32)[None, :, :]
                diff = cands[:, None, :] - base
                dist = np.sqrt(np.sum(diff * diff, axis=2))
                score = dist.min(axis=1)
            else:
                score = np.full((len(cands),), np.inf, dtype=np.float32)
            j = int(np.argmax(score))
            pts.append((float(cands[j, 0]), float(cands[j, 1])))
        return np.array(pts, dtype=np.float32)

    # ============================== 打印工具 ============================ #
    def _p(self, tag: str, **data):
        """精简打印：按schema过滤字段；数组用摘要；列表只打头部。"""

        # >>> 新增：全量模式（最小入侵，其他逻辑都不动）
        if getattr(self, "log_full", False):
            import numpy as _np
            def to_plain(v):
                if isinstance(v, _np.ndarray):
                    return v.tolist()
                if isinstance(v, (_np.integer, _np.floating, _np.bool_)):
                    return v.item()
                if isinstance(v, (list, tuple)):
                    return [to_plain(x) for x in v]
                if isinstance(v, dict):
                    return {k: to_plain(x) for k, x in v.items()}
                return v

            # 不做 schema 过滤
            full = {k: to_plain(v) for k, v in data.items()}
            print(f"[{tag}] {full}")
            return

        # —— 原有摘要模式：保留不变 —— #
        allow = self._log_schema.get(tag)
        if allow:
            allow_keys = set(allow.get("basic", []))
            if self.log_level in ("detail", "trace"):
                allow_keys |= set(allow.get("detail", []))
            if self.log_level == "trace":
                allow_keys |= set(allow.get("trace", []))
            data = {k: v for k, v in data.items() if k in allow_keys}

        def fmt(v):
            if isinstance(v, np.ndarray):
                return self._nz_summary(v)
            elif isinstance(v, list):
                if v and isinstance(v[0], dict) and len(v) <= self._nz_limit:
                    return "[" + ", ".join(str(d) for d in v) + "]"
                if len(v) <= self._nz_limit and (not v or not isinstance(v[0], dict)):
                    return str(v)
                return f"list(len={len(v)})"
            elif isinstance(v, dict):
                keys = list(v.keys())
                return f"dict(keys={keys[:min(6, len(keys))]}...)"
            else:
                return v

        items = ", ".join([f"{k}={fmt(v)}" for k, v in data.items()])
        print(f"[{tag}] {items}")

    def _nz_summary(self, v: np.ndarray):
        """非零摘要：{idx:val,...} + (nz/len)。"""
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        v = v.reshape(-1)
        idx = np.nonzero(v)[0]
        nz = idx.size
        if nz == 0:
            return "∅"
        show = []
        for i in idx[:self._nz_limit]:
            show.append(f"{int(i)}:{int(v[i])}")
        more = "" if nz <= self._nz_limit else f", ... (nz={nz}/{v.size})"
        return "{ " + ", ".join(show) + "}" + more

    # ====================== NEW: 每槽 SD 位置日志 ====================== #
    def _log_sd_positions_slot(self, slot: int) -> None:
        """
        打印本槽（已结束/或当前编号）的所有 SD 位置。
        格式示例：
          [SD_POS] slot=12 items=[{'k':0,'x':123.4,'y':567.8,'z':50.0}, ...]
        """
        try:
            P = getattr(self, "P_sd", None)
            if P is None:
                return
            P = np.asarray(P, dtype=float)
            if P.ndim != 2 or P.shape[1] < 3:
                return
            items = []
            for k in range(P.shape[0]):
                x, y, z = float(P[k, 0]), float(P[k, 1]), float(P[k, 2])
                items.append({
                    "k": int(k),
                    "x": round(x, 1),
                    "y": round(y, 1),
                    "z": round(z, 1),
                })
            # 直接 print，独立于 self._p()/log_full，避免被 schema 或模式改写
            print(f"[SD_POS] slot={int(slot)} items={items}")
        except Exception:
            # 不让日志影响仿真
            pass

    def _as_idx_map_str(self, arr, rounder=None):
        """
        把一个一维数组/列表转成 '{ 0:1.2, 1:3.4, ... }' 字符串。
        默认用 self._r1 做四舍五入。
        """
        v = np.asarray(arr).reshape(-1)
        rfun = rounder if rounder is not None else self._r1
        parts = [f"{i}:{rfun(v[i])}" for i in range(v.size)]
        return "{ " + ", ".join(parts) + " }" if parts else "∅"

    def _log_reset_map(self):
        """reset 时打印一次初始地图信息。"""
        sd_lines = [f"SD-{k}=({self.P_sd[k, 0]:.1f},{self.P_sd[k, 1]:.1f},{self.P_sd[k, 2]:.1f})"
                    for k in range(self.K)]
        show = min(self.M, self._top_ds)
        ds_lines = [f"DS-{i}=({self.ds_pos[i, 0]:.1f},{self.ds_pos[i, 1]:.1f},{self.ds_pos[i, 2]:.1f})"
                    for i in range(show)]
        print("[RESET] Tr-UAV pos=({:.1f},{:.1f},{:.1f}), K={}, M={}".format(
            self.tsd_pos[0], self.tsd_pos[1], self.tsd_pos[2], self.K, self.M))
        print("[RESET] SD positions: " + "; ".join(sd_lines))
        print("[RESET] DS positions (head {} of {}): ".format(show, self.M) + "; ".join(ds_lines))

    def _log_tr_alloc_per_sd(self, *, active: int, moved: np.ndarray,
                             mode: str, cap_enc_imgs: int) -> None:
        """
        打印本槽 Tr->SD 的分配结果：
          - items: 仅包含“实际收到数据”的 SD（连接成功的）
          - active_ds_map: 活跃 SD 从各 DS 拉取的分解（非零摘要）
        """
        try:
            items = []
            enc_img_size_Mb = float(self.size_enc_img_bits / self._MBIT)

            # 汇总每个 SD（只保留 >0 的，代表“连接且收到数据”）
            for k in range(self.K):
                imgs_k = int(self.tr_tx_last[k].sum())
                if imgs_k > 0:
                    mb_k = float(imgs_k * enc_img_size_Mb)
                    items.append({"k": int(k), "imgs": int(imgs_k), "Mb": self._r2(mb_k)})

            # 活跃 SD 的 DS 分解与总量
            moved_vec = np.asarray(moved, dtype=int).reshape(-1)
            active_total_imgs = int(moved_vec.sum())
            active_total_Mb = float(active_total_imgs * enc_img_size_Mb)

            self._p("TR_ALLOC_PER_SD",
                    active=int(active),
                    mode=str(mode),
                    cap_enc_imgs=int(cap_enc_imgs),
                    enc_img_size_Mb=self._r2(enc_img_size_Mb),
                    items=items,
                    active_ds_map=self._nz_summary(moved_vec),
                    active_total_imgs=int(active_total_imgs),
                    active_total_Mb=self._r2(active_total_Mb))
        except Exception:
            # 不让日志影响仿真
            pass

    def _alloc_prop_cap_by_weights(self, cap: int, weights: np.ndarray, qmax: np.ndarray, rr_ptr: int,
                                   fallback_uniform_if_zero: bool = True):
        """
        基于“归一化权重 × 总容量”的整数分配（满足各自 qmax），支持轮询指针打散平局：

        cap     : 本槽 Tr->SD 的总可分配名额（整数，单位=编码整图）
        weights : 策略输出的连续权重（>=0），长度 M
        qmax    : 每个 DS 的可用上限（这里=Q_tr[m]）
        rr_ptr  : 轮询指针（用于在余数相等时打散）
        fallback_uniform_if_zero:
                  当 eligible 上权重和=0 时，是否在 eligible 上均匀分配（True）；
                  若为 False，则直接全 0（完全按策略来）。

        返回: moved[M]（整数分配），new_ptr
        """
        M = int(self.M)
        cap = int(max(0, cap))
        moved = np.zeros(M, dtype=int)

        total_supply = int(np.asarray(qmax, dtype=int).sum())
        if cap == 0 or M == 0 or total_supply <= 0:
            return moved, rr_ptr

        w = np.asarray(weights, dtype=np.float64).copy()
        w[~np.isfinite(w)] = 0.0
        w[w < 0] = 0.0

        eligible = (np.asarray(qmax, dtype=int) > 0)
        w_eff = w * eligible

        s = float(w_eff.sum())
        if s <= 0.0:
            if not fallback_uniform_if_zero:
                return moved, rr_ptr
            # 均匀回退：在 eligible 上均分
            w_eff = eligible.astype(np.float64)
            s = float(w_eff.sum())

        # 归一化到 1（只在 eligible 上）
        w_norm = w_eff / max(s, 1e-12)

        quota = int(min(cap, total_supply))  # 最多只能分配到 backlog 总量
        frac_target = w_norm * quota

        # 1) 基础整数部分（不超过各自 qmax）
        base = np.floor(frac_target).astype(int)
        base = np.minimum(base, qmax.astype(int))

        remain = int(quota - int(base.sum()))
        if remain <= 0:
            return base.astype(int), rr_ptr

        # 2) 最大余数法 + 轮询指针打散；同时受 qmax 剩余约束
        remainder = frac_target - base
        headroom = (qmax.astype(int) - base).astype(int)

        idx = np.arange(M)
        shift = int(rr_ptr) % max(M, 1)
        idx_shifted = np.roll(idx, -shift)

        order = np.lexsort((idx_shifted, -w_norm, -remainder))  # 余数优先，其次看权重，再看轮询顺序

        for j in order:
            if remain <= 0:
                break
            if headroom[j] <= 0:
                continue
            add = int(min(headroom[j], remain))
            if add > 0:
                base[j] += add
                headroom[j] -= add
                remain -= add

        new_ptr = int((rr_ptr + 1) % max(M, 1))
        return base.astype(int), new_ptr
