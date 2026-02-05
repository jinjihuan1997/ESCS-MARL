# =====================================
# Filepath: envs/env_core.py
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
          5) 飞行结束后执行 Tr→SD（轮转到活跃SD），按 BS trans_picks 拉取
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
        self.tag = ""  # 正常运行：""；人工初始化：设为 "demo"
        self.sd_init_override_xy = None  # 人工给定的坐标 (K,2) 或 (K,3)
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
        self.W_sd = 10.0
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
        self.snr_thr_sd_ds = 7.0
        self.a_los = 4.88
        self.b_los = 0.43

        # 图像/压缩
        self.S_img = 1024 * 1024 * 3 * 8  # = 25_165_824 bits
        self.cmp_ratio = 1 / 50  # 压缩了98%的数据
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
                                "detail": ["enc_picks", "trans_picks", "enc_cum_nz"],  # <<< 新增：累计编码摘要
                            },
                            "SD_HOVER_THEN_FLY": {"basic": ["items"]},
                            "SD_to_DS@hover": {
                                "basic": ["delivered_total", "per_sd"],
                                "trace": ["mask_head"],
                                "detail": ["sd_cum_tx_nz"],  # <<< 新增：各 SD 累计发往各 DS 的摘要
                            },
                            "TR_PULL": {
                                "basic": [
                                    "active", "link_ok", "pulled_blocks",
                                    "cap_enc_imgs", "cap_Mb",
                                    "backlog_blocks", "backlog_Mb",
                                    "used_first",
                                    "d_tr_active"
                                ],
                                "detail": ["tr_cum_active_nz"],  # <<< 新增：active SD 的累计 Tr->SD 摘要
                            },
                            "SP_REWARD": {
                                "basic": ["delivered_total", "avg_sp", "max_sp",
                                          "deliver_reward", "sp_penalty", "collision_penalty",
                                          "move_term", "reward"],  # 用 move_term 代替旧的 move_bonus
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

        # SD 在悬停阶段选择 DS 的语义：'global' | 'local+noop_last'
        # - 'global'        : 策略输出的是全局 DS id，M 表示 NOOP（当前行为）
        # - 'local+noop_last': 策略输出的是“valid_set 的下标”，且最后一格是 NOOP
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
        self._snr_cache_sd_ds = None  # 形状 (K, M)，本时隙冻结的 SD->DS SNR(dB)

        self._snr_cache_slot_tr_sd = None  # Tr->SD 缓存属于哪个时隙 self.t
        self._snr_cache_tr_sd = None  # 形状 (K, )，本时隙每个 SD 一份 Tr->SD SNR(dB)

        # ==== NEW: BS 策略开关（默认自动轮询，不使用 BS Agent 的动作） ====
        self.use_bs_agent = False  # False=自动轮询；True=沿用旧的动作语义
        # 轮询指针（跨槽记忆，保证“公平性”）
        self._bs_rr_ptr_enc = 0  # 编码 eRU 轮询指针（0..M-1）
        self._bs_rr_ptr_trsd = 0  # Tr->SD 拉取轮询指针（0..M-1）
        self._apply_comm_mode()

    # ================================ API ================================ #
    def reset(self):
        self.RS_free = self.RS_tot
        self.finish_events.clear()
        self.Q_tr[:] = 0
        self.Q_sd[:, :] = 0
        self.SP[:] = 0
        self._place_sd_init_positions()
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
        self.bs_enc_last[:] = 0
        self.tr_tx_last[:, :] = 0
        self.sd2ds_sent_last[:, :] = 0
        self.ds_rx_last_from_sd[:, :] = 0

        # 1) 解包动作
        A_bs, sd_actions = self._unpack_actions(actions)
        enc_picks = A_bs[:self.L_BS_ENC]
        trans_picks = A_bs[self.L_BS_ENC:self.L_BS_ENC + self.L_BS_TRANS]

        if self.debug:
            self._p("STEP", slot=self.t)

        # 2) 槽边界释放：归还eRU并向Q_tr补货
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

        # 3) BS编码资源分配（登记返还事件）
        if self.RS_free > 0:
            if self.use_bs_agent:
                self._apply_bs_encoding(enc_picks.tolist(), int(self.RS_free))
                mode = "agent"
            else:
                self._apply_bs_encoding_auto_rr(int(self.RS_free))
                mode = "auto_rr"
            if self.debug:
                log = (self._last_enc_log.copy() if self._last_enc_log else {})
                log.update({
                    "mode": mode,
                    "enc_picks": enc_picks.tolist(),
                    "trans_picks": trans_picks.tolist(),
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

        # 4) 解析 SD 动作（名义飞/停），先在当前位置判定 valid_set 再修正实际飞/停
        dir_idx, fly_idx, ds_idx = self._parse_sd_actions(sd_actions)
        phis = self.dir_table[dir_idx]
        fly_times_nominal = (fly_idx / max(self.n_fly - 1, 1)) * self.delta_T
        hov_times_nominal = np.clip(self.delta_T - fly_times_nominal, 0.0, None)
        pre_pos = self.P_sd.copy()  # 悬停位置=当前位置

        # 冻结本槽 SD->DS 的 SNR(dB)
        snr_slot_pre = self._ensure_snr_cache_sd_ds().copy()

        # 链路地图日志（可选）
        linkmap_rows = []
        for k in range(self.K):
            L_all = np.linalg.norm(self.ds_pos - pre_pos[k], axis=1)
            S_all = snr_slot_pre[k]
            ds_d_all_str = self._as_idx_map_str(L_all, rounder=self._r1)
            ds_snr_all_str = self._as_idx_map_str(S_all, rounder=self._r1)
            conn_mask = (S_all >= self.snr_thr_sd_ds)
            conn_idx = np.where(conn_mask)[0]
            if conn_idx.size > 0:
                order = np.argsort(np.linalg.norm(self.ds_pos[conn_idx] - pre_pos[k], axis=1))
                head = [f"{int(m)}:{self._r1(np.linalg.norm(self.ds_pos[int(m)] - pre_pos[k]))}"
                        for m in conn_idx[order][:self._top_conn]]
                conn_head_str = "{ " + ", ".join(head) + (" ... }" if conn_idx.size > self._top_conn else " }")
            else:
                conn_head_str = "∅"
            linkmap_rows.append({
                "k": int(k),
                "ds_d_all": ds_d_all_str,
                "ds_snr_all": ds_snr_all_str,
                "conn_ds_d_head": conn_head_str,
            })
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

        dist_tr_pre = np.linalg.norm(pre_pos - self.tsd_pos[None, :], axis=1)

        # —— 最终采用：完全按动作 —— #
        fly_times = fly_times_nominal.astype(np.float32, copy=True)
        hov_times = hov_times_nominal.astype(np.float32, copy=False)
        for k in range(self.K):
            if valid_sets[k].size == 0:
                fly_times[k] = float(self.delta_T)
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

        # 5) 悬停阶段执行 SD→DS
        delivered_total = 0
        delivered_per_m = np.zeros(self.M, dtype=int)
        per_sd_brief: List[Dict[str, object]] = []
        T_sym = self.T_DFT_sd_ds + self.T_GI_sd_ds
        ds_exec = np.full(self.K, self.M, dtype=int)

        for k in range(self.K):
            valid_idx = valid_sets[k]

            # 供决策映射用的布尔掩码（长度 M）
            valid_mask = np.zeros(self.M, dtype=bool)
            valid_mask[valid_idx] = True

            # 供日志显示用的 (M+1) 掩码：前 M 为 valid，最后一格为 NOOP
            mask_valid = np.zeros(self.M + 1, dtype=int)
            mask_valid[self.M] = 1
            mask_valid[valid_idx] = 1

            raw_pick = int(ds_idx[k])
            pick = self._map_sd_ds_pick(raw_pick, valid_mask, mode=self.sd_ds_pick_semantics)

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
                                     "snr_db": round(snr_db, 1), "thr_db": float(self.snr_thr_sd_ds),
                                     "note": "not connectable"})
                ds_exec[k] = self.M
                continue

            bps, cr = self._lookup_mcs(snr_db)
            Rm = bps * cr * self.N_ss_tr_sd * float(self.N_sub_sd_ds) / T_sym  # bits/s
            rate_Mbps = float(Rm / self._MBIT)
            can_tx = int((Rm * float(hov_times[k])) // self.S_img)
            C_bits_sd_ds = Rm * float(hov_times[k])
            cap_orig_imgs_sd_ds = int(C_bits_sd_ds // self.size_orig_img_bits)
            cap_Mb_sd_ds = float(C_bits_sd_ds / self._MBIT)

            if can_tx <= 0:
                per_sd_brief.append({"k": int(k), "pick": int(pick),
                                     "snr_db": round(snr_db, 1),
                                     "rate_Mbps": self._r1(rate_Mbps),
                                     "cap_Mb": self._r2(cap_Mb_sd_ds),
                                     "hov_s": self._r1(hov_times[k]),
                                     "note": "rate too low"})
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

        # 6) 悬停后飞行：更新位置并越界检查
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

        self.P_sd[:, :] = post_pos

        # 7) 飞行结束后执行 Tr→SD（仅1架活跃，轮转）
        active = (self.t - 1) % self.K
        mid_pos = 0.5 * (pre_pos[active] + post_pos[active])
        L_tr_sd = float(np.linalg.norm(mid_pos - self.tsd_pos))
        snr_trsd = self._get_snr_tr_sd_slot(active, L_tr_sd)
        link_ok = (snr_trsd >= self.snr_thr_tr_sd) and (float(fly_times[active]) > 0.0)

        rx_total = 0
        if link_ok:
            enc_img_size_Mb = float(self.size_enc_img_bits / self._MBIT)

            if self.comm_mode == "SC":
                # ===== SC：符号计量 =====
                T_sym_tr = float(self.T_DFT_tr_sd + self.T_GI_tr_sd)
                sy_rate = (float(self.N_sub_tr_sd) * float(self.N_ss_tr_sd)) / max(T_sym_tr, 1e-12)  # symbols/s
                tx_time_s = float(fly_times[active])
                syms_avail = sy_rate * tx_time_s
                cap_imgs_theory = int(syms_avail // max(self.Upsilon_sym_per_img, 1e-9))

                if self.use_bs_agent:
                    tx_mask = (np.asarray(trans_picks[:self.M], dtype=int) > 0)
                    remain = int(cap_imgs_theory)
                    moved = np.zeros(self.M, dtype=int)
                    if remain > 0:
                        for m in range(self.M):
                            if not tx_mask[m]:
                                continue
                            take = min(int(self.Q_tr[m]), remain)
                            if take <= 0:
                                continue
                            self.Q_tr[m] -= take
                            self.Q_sd[active, m] += take
                            moved[m] = take
                            remain -= take
                            if remain <= 0:
                                break
                    rx_total = int(moved.sum())
                    backlog_blocks_selected = int(min(cap_imgs_theory, (self.Q_tr + moved).sum()))
                    self._last_pull_log = {
                        "mode": "agent_sc_symbol",
                        "quota": int(cap_imgs_theory),
                        "pulled": int(rx_total),
                        "used_first": moved.astype(int),
                        "used_rr": np.zeros(self.M, dtype=int),
                        "cap_enc_imgs_theory": int(cap_imgs_theory),
                        "cap_syms_theory": float(syms_avail),
                        "Upsilon_sym_per_img": float(self.Upsilon_sym_per_img),
                        "cap_Mb_theory": 0.0,
                        "enc_img_size_Mb": float(enc_img_size_Mb),
                        "backlog_enc_Mb": 0.0,
                        "backlog_blocks_selected": int(backlog_blocks_selected),
                    }
                else:
                    moved = self._auto_rr_tr_pull(active=active, cap_enc_imgs=int(cap_imgs_theory))
                    rx_total = int(moved.sum())
                    backlog_blocks_selected = int(min(cap_imgs_theory, self.Q_tr.sum() + rx_total))
                    self.tr_tx_last[:, :] = 0
                    # 关键：这里也把实际分配写入 used_first，便于 TR_PULL 打印
                    self._last_pull_log = {
                        "mode": "auto_rr_sc_symbol",
                        "quota": int(cap_imgs_theory),
                        "pulled": int(rx_total),
                        "used_first": moved.astype(int),
                        "used_rr": moved.astype(int),
                        "cap_enc_imgs_theory": int(cap_imgs_theory),
                        "cap_syms_theory": float(syms_avail),
                        "Upsilon_sym_per_img": float(self.Upsilon_sym_per_img),
                        "cap_Mb_theory": 0.0,
                        "enc_img_size_Mb": float(enc_img_size_Mb),
                        "backlog_enc_Mb": 0.0,
                        "backlog_blocks_selected": int(backlog_blocks_selected),
                    }

            else:
                # ===== TC：bit 速率 + MCS =====
                T_sym_tr = float(self.T_DFT_tr_sd + self.T_GI_tr_sd)
                bps_tr, cr_tr = self._lookup_mcs(float(snr_trsd))
                Rm_tr = float(bps_tr) * float(cr_tr) * float(self.N_ss_tr_sd) * float(self.N_sub_tr_sd) / max(T_sym_tr,
                                                                                                        1e-12)  # bits/s
                tx_time_s = float(fly_times[active])
                C_bits_tr_sd = Rm_tr * tx_time_s
                cap_imgs_theory = int(C_bits_tr_sd // max(self.size_enc_img_bits, 1.0))
                cap_Mb_tr_sd = float(C_bits_tr_sd / self._MBIT)

                if self.use_bs_agent:
                    tx_bits = np.asarray(trans_picks[:self.M], dtype=int) > 0
                    remain = int(cap_imgs_theory)
                    moved = np.zeros(self.M, dtype=int)
                    if remain > 0:
                        for m in range(self.M):
                            if not tx_bits[m]:
                                continue
                            take = min(int(self.Q_tr[m]), remain)
                            if take <= 0:
                                continue
                            self.Q_tr[m] -= take
                            self.Q_sd[active, m] += take
                            moved[m] = take
                            remain -= take
                            if remain <= 0:
                                break
                    rx_total = int(moved.sum())
                    backlog_blocks_selected = int(min(cap_imgs_theory, (self.Q_tr + moved).sum()))
                    self._last_pull_log = {
                        "mode": "agent",
                        "quota": int(cap_imgs_theory),
                        "pulled": int(rx_total),
                        "used_first": moved.astype(int),
                        "used_rr": np.zeros(self.M, dtype=int),
                        "cap_enc_imgs_theory": int(cap_imgs_theory),
                        "backlog_blocks_selected": int(backlog_blocks_selected),
                        "cap_Mb_theory": float(cap_Mb_tr_sd),
                        "enc_img_size_Mb": float(enc_img_size_Mb),
                        "backlog_enc_Mb": float(backlog_blocks_selected * self.size_enc_img_bits / self._MBIT),
                    }
                else:
                    moved = self._auto_rr_tr_pull(active=active, cap_enc_imgs=int(cap_imgs_theory))
                    rx_total = int(moved.sum())
                    backlog_blocks_selected = int(min(cap_imgs_theory, self.Q_tr.sum() + rx_total))
                    self.tr_tx_last[:, :] = 0
                    # 同样，把结果写到 used_first，便于打印
                    self._last_pull_log = {
                        "mode": "auto_rr",
                        "quota": int(cap_imgs_theory),
                        "pulled": int(rx_total),
                        "used_first": moved.astype(int),
                        "used_rr": moved.astype(int),
                        "cap_enc_imgs_theory": int(cap_imgs_theory),
                        "backlog_blocks_selected": int(backlog_blocks_selected),
                        "cap_Mb_theory": float(cap_Mb_tr_sd),
                        "enc_img_size_Mb": float(enc_img_size_Mb),
                        "backlog_enc_Mb": float(backlog_blocks_selected * self.size_enc_img_bits / self._MBIT),
                    }

            # 记录 Tr->SD（仅 active 行非零）+ 累计
            self.tr_tx_last[:, :] = 0
            self.tr_tx_last[active, :] = moved
            self.cum_tr_to_sd[active, :] += moved
            self.cum_tr_to_sd_total[active] += rx_total
        else:
            # === 修复点 #1：无链路占位日志，避免沿用上一槽 ===
            self._last_pull_log = {
                "mode": "no_link",
                "pulled": 0,
                "cap_enc_imgs_theory": 0,
                "backlog_blocks_selected": 0,
                "cap_Mb_theory": 0.0,
                "backlog_enc_Mb": 0.0,
                "used_first": np.zeros(self.M, dtype=int),
                "used_rr": np.zeros(self.M, dtype=int),
            }

        if self.debug:
            log = self._last_pull_log
            # === 修复点 #2：优先显示 used_first；若为 0 再回退 used_rr ===
            used_vec = np.asarray(log.get("used_first", np.zeros(self.M, dtype=int)))
            if used_vec.sum() == 0 and "used_rr" in log:
                used_vec = np.asarray(log.get("used_rr", np.zeros(self.M, dtype=int)))

            self._p("TR_PULL",
                    active=active,
                    link_ok=bool(link_ok),
                    pulled_blocks=int(log.get("pulled", 0)),
                    cap_enc_imgs=int(log.get("cap_enc_imgs_theory", 0)),
                    cap_Mb=self._r2(log.get("cap_Mb_theory", 0.0)),
                    backlog_blocks=int(log.get("backlog_blocks_selected", 0)),
                    backlog_Mb=self._r2(log.get("backlog_enc_Mb", 0.0)),
                    used_first=self._nz_summary(used_vec),
                    d_tr_active=self._r1(L_tr_sd),
                    tr_cum_active_nz=self._nz_summary(self.cum_tr_to_sd[active]),
                    raw_tr_pull=self._last_pull_log
                    )

        # 额外队列快照
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

        # 8) 更新SP与奖励
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
                    ds_streak_head=self._nz_summary(self.ds_streak),
                    decay_head=self._nz_summary((decay > 0).astype(int)),  # 或打印实际衰减值的头部
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

        # 9) 推进时间并返回
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
        改版：BS 侧观测（按用户要求，拼接以下 5 部分）
          1) Tr 队列 per-DS 的 [ds_id_norm, Q_tr[m]] 共 2M 维；
          2) 当前 Tr↔激活 SD 的“理论可传输编码图像数量”（以整图为单位；1 维）；
          3) 激活 SD 的 id_norm（1 维）；
          4) 激活 SD 的队列信息：[id_norm] + per-DS 的 [ds_id_norm, Q_sd[active,m]] 共 (1+2M) 维；
          5) 激活 SD 与全体 DS 的 SNR(dB) + SNR 阈值，共 (M+1) 维。
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
            # 物理层速率（bit/s）
            T_sym = float(self.T_DFT_tr_sd + self.T_GI_tr_sd)
            if snr_trsd >= float(self.snr_thr_tr_sd):
                if self.comm_mode == "SC":
                    # —— SC：符号计量 —— #
                    sy_rate = (float(self.N_sub_tr_sd) * float(self.N_ss_tr_sd)) / max(T_sym, 1e-9)  # symbols/s
                    syms_avail = sy_rate * float(self.delta_T)
                    cap_imgs = float(syms_avail // max(self.Upsilon_sym_per_img, 1e-9))
                else:
                    # —— TC：沿用 bit 速率 —— #
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

        # --------- 5) 激活 SD 的 SNR(dB) + 阈值：(M+1) 维 ---------
        if K > 0:
            snr_row = self._ensure_snr_cache_sd_ds()[active].astype(np.float32, copy=False)
        else:
            snr_row = np.zeros(M, dtype=np.float32)
        thr = np.array([float(self.snr_thr_sd_ds)], dtype=np.float32)
        snr_block = np.concatenate([snr_row, thr], axis=0)  # (M+1,)

        # ===== 拼接并返回 =====
        return np.concatenate([tr_pairs, cap_arr, active_id_arr, sd_block, snr_block]).astype(np.float32)

    def _build_masks_for_sd(self, k: int):
        """
        返回：
          mask_conn_global: (M+1,) float32，前 M 维对应全局 DS，最后一维为 NOOP。
                            规则：valid = (SNR>=thr) & (Q_sd[k,m]>0)。
          mask_fly:         (n_fly,) float32。若 valid=∅ => 仅最后一格为1（满飞），否则全1。
        """
        snr_row = self._ensure_snr_cache_sd_ds()[k]
        conn_mask = (snr_row >= float(self.snr_thr_sd_ds))
        data_mask = (self.Q_sd[k] > 0)
        valid_mask = conn_mask & data_mask

        mask_conn_global = np.zeros(self.M + 1, dtype=np.float32)
        mask_conn_global[:self.M] = valid_mask.astype(np.float32)
        mask_conn_global[self.M] = 1.0  # NOOP 永远允许

        mask_fly = np.ones(self.n_fly, dtype=np.float32)
        if not valid_mask.any():
            mask_fly[:] = 0.0
            mask_fly[self.n_fly - 1] = 1.0  # 仅允许满飞
        return mask_conn_global, mask_fly

    def _obs_sd(self, k: int):
        """
        SD 侧观测（原5块） + 两个 action mask：
          1) id_norm（1）
          2) 与其他 SD 的欧氏距离（K-1）
          3) 与所有 DS 的 [平面距离, SNR(dB)] 并追加 SNR 阈值 (2M+1)
          4) SD 内部队列：[ds_id_norm, Q_sd[k,m]] × M -> 2M
          5) 所有 DS 的 [ds_id_norm, SP[m]] × M -> 2M
          6) mask_conn_global: (M+1)
          7) mask_fly: (n_fly)
        返回长度：K + 7M + n_fly + 2
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
        ds_id_norm = (np.arange(M, dtype=np.float32) / float(M - 1)).astype(np.float32) if M > 1 else np.zeros(M,
                                                                                                               dtype=np.float32)

        # 3) [距离, SNR] + 阈值
        if M > 0:
            se_xy = self.P_sd[k, :2]
            ds_xy = self.ds_pos[:, :2]
            dist_sd_ds = np.linalg.norm(ds_xy - se_xy[None, :], axis=1).astype(np.float32)
            snr_row = self._ensure_snr_cache_sd_ds()[k].astype(np.float32, copy=False)
            pair_dist_snr = np.empty(2 * M, dtype=np.float32)
            pair_dist_snr[0::2] = dist_sd_ds
            pair_dist_snr[1::2] = snr_row
        else:
            pair_dist_snr = np.zeros(0, dtype=np.float32)
        thr_arr = np.array([float(self.snr_thr_sd_ds)], dtype=np.float32)

        # 4) [ds_id_norm, Q_sd[k,m]]
        if M > 0:
            qsd_k = self.Q_sd[k].astype(np.float32, copy=False)
            pair_id_q = np.empty(2 * M, dtype=np.float32)
            pair_id_q[0::2] = ds_id_norm
            pair_id_q[1::2] = qsd_k
        else:
            pair_id_q = np.zeros(0, dtype=np.float32)

        # 5) [ds_id_norm, SP[m]]
        if M > 0:
            SP = self.SP.astype(np.float32, copy=False)
            pair_id_sp = np.empty(2 * M, dtype=np.float32)
            pair_id_sp[0::2] = ds_id_norm
            pair_id_sp[1::2] = SP
        else:
            pair_id_sp = np.zeros(0, dtype=np.float32)

        # 基础观测
        base_obs = np.concatenate([id_arr, sd_dist_vec, pair_dist_snr, thr_arr, pair_id_q, pair_id_sp], axis=0).astype(
            np.float32)

        # 统一生成 mask
        mask_conn_global, mask_fly = self._build_masks_for_sd(k)

        return np.concatenate([base_obs, mask_conn_global.astype(np.float32), mask_fly.astype(np.float32)],
                              axis=0).astype(np.float32)

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
        # 允许 step() 事先存入，若没有则按 0.0 处理
        collision_penalty = float(getattr(self, "_last_collision_penalty", 0.0))
        move_bonus = float(getattr(self, "_last_move_bonus", 0.0))

        # 新增：细分字段（若没缓存则回退 0）
        move_pos = float(getattr(self, "_last_move_pos", 0.0))
        move_neg = float(getattr(self, "_last_move_neg", 0.0))
        move_bonus_pos = float(
            getattr(self, "_last_move_bonus_pos", move_pos * float(getattr(self, "w_move_pos", 1.0))))
        move_penalty_away = float(
            getattr(self, "_last_move_penalty_away", move_neg * float(getattr(self, "w_move_away", 1.0))))

        shared_reward = float(deliver_reward - sp_penalty - collision_penalty + move_bonus)

        info_bs["reward_components"] = {
            "deliver": float(deliver_reward),
            "sp": -float(sp_penalty),
            "collision": -float(collision_penalty),
            "move": float(move_bonus),  # 净移动项 = 加分 - 扣分

            # —— 可视化/诊断用的细分 ——
            "move_pos": float(move_pos),  # 未加权靠近量
            "move_neg": -float(move_neg),  # 未加权远离量（带负号表示惩罚向）
            "move_bonus_pos": float(move_bonus_pos),  # 已加权靠近加分
            "move_penalty_away": -float(move_penalty_away),  # 已加权远离扣分（负号）

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
    def _unpack_actions(self, actions: Union[List, Tuple, Dict]):
        if isinstance(actions, dict):
            A_bs = np.array(actions.get('bs', []), dtype=int)
            sd_raw = actions.get('sd', [])
            if not isinstance(sd_raw, (list, tuple)) or len(sd_raw) != self.K:
                raise ValueError("dict['sd'] 必须是长度 K 的列表")
            sd_actions = [np.array(a, dtype=int) for a in sd_raw]
        else:
            if not isinstance(actions, (list, tuple)) or len(actions) != self.K + 1:
                raise ValueError(f"list/tuple 形式下长度必须是 K+1={self.K + 1}")
            A_bs = np.array(actions[0], dtype=int)
            sd_actions = [np.array(actions[i + 1], dtype=int) for i in range(self.K)]

        # BS 长度对齐
        need_bs = self.L_BS_ENC + self.L_BS_TRANS
        if A_bs.size < need_bs:
            A_bs = np.pad(A_bs, (0, need_bs - A_bs.size), mode='constant')
        elif A_bs.size > need_bs:
            A_bs = A_bs[:need_bs]

        # SD 动作长度固定为3：[dir_idx, fly_idx, ds_idx]
        need_sd = 3
        for i in range(self.K):
            if sd_actions[i].size < need_sd:
                sd_actions[i] = np.pad(sd_actions[i], (0, need_sd - sd_actions[i].size), mode='constant')
            elif sd_actions[i].size > need_sd:
                sd_actions[i] = sd_actions[i][:need_sd]
        return A_bs, sd_actions

    def _map_sd_ds_pick(self, action_pick: int, valid_mask: np.ndarray, mode: str) -> int:
        """
        把策略给的 ds_idx 映射成【全局 DS id】。
        valid_mask: shape (M,), bool，表示对该 SD 当前槽下可连且有数据的 DS。
        返回 self.M 表示 NOOP。
        """
        if self.M <= 0:
            return self.M

        if mode == 'global':
            if 0 <= action_pick < self.M and bool(valid_mask[action_pick]):
                return int(action_pick)
            return self.M

        if mode == 'local+noop_last':
            valid_idx = np.flatnonzero(valid_mask)
            if action_pick == len(valid_idx):
                return self.M
            if 0 <= action_pick < len(valid_idx):
                return int(valid_idx[action_pick])
            return self.M

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
        """
        BS 编码资源分配：登记未来返还事件（整批同一返还时刻）。
        - 在 TC 模式下：编码时延固定为 self.tc_enc_delay_s（默认 1s）
        - 在 SC 模式下：维持原随机时延 tao ∈ [tao_min, tao_max]
        """
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

        # 填平四舍五入误差
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

        # === 时延改动点 ===
        if self.comm_mode == "TC":
            # 固定 1s 编码时延
            tao = float(self.tc_enc_delay_s)
        else:
            # SC 维持原随机时延
            tao = float(self.rng.uniform(self.tao_min, self.tao_max))
        tau_slot = int(math.ceil(tao / self.delta_T))
        t_rel = self.t + tau_slot

        # 登记返还事件
        for m in range(self.M):
            q = int(alloc[m])
            if q <= 0:
                continue
            for _ in range(q):
                self.finish_events.append((t_rel, int(m)))

        # 统计与日志
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

        # 一个批次共用一个延迟（与旧实现一致）
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
        TC 模式下的编码：编码时间=0，立即把本槽可用的编码资源 n_assign 转为 Q_tr 产能。
        - 不产生 finish_events，不等待返还；
        - 立刻释放 eRU（使 RS_free 在本槽不变动的等效行为）。
        - 如果 use_bs_agent=True，则参考 enc_picks 的“投票”来分配；否则按自动轮询平均分配。
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
            # 按需硬写：支持 [x,y] 或 [x,y,z]；不足 z 自动补 self.z_sd
            coords = np.array([
                [1000.0, 2000, 50.0],  # SD-0
                [1000.0, 1000.0, 50.0],  # SD-1
                [2000.0, 1000.0, 50.0],  # SD-2
            ], dtype=np.float32)

            # 若只给 (x,y)，自动补 z
            if coords.shape[1] == 2:
                zcol = np.full((coords.shape[0], 1), float(self.z_sd), dtype=np.float32)
                coords = np.concatenate([coords, zcol], axis=1)

            n = int(min(self.K, coords.shape[0]))
            # 限幅到地图范围
            coords[:n, 0] = np.clip(coords[:n, 0], 0.0, self.map_size)
            coords[:n, 1] = np.clip(coords[:n, 1], 0.0, self.map_size)
            self.P_sd[:n, :3] = coords[:n, :3]

            # 若 K 比硬写多，剩余的沿用原布局补齐
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

        # ===== 非 DEMO：保持你现有的默认布局（原代码）=====
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
