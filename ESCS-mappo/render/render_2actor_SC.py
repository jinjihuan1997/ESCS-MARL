# Filepath: render_2actor_SC.py
# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Render for multi-UAV env with SP-based DS coloring and modern Matplotlib APIs,
and (NEW) satellite/background image overlay.

新增功能（相对你上一版）：
1) 支持加载并叠加卫星背景图（--bg_image/--bg_alpha/--bg_origin）。
2) 背景图在三处均生效：GIF 每帧、总览 PNG、论文版单图。
3) 仍保持“仅点不连线”、SD 起终点标注、Tr-UAV 标注、右侧信息栏等原逻辑。

本次修改点：
A) 默认禁用 GIF 生成（新增 --make_gif 开关）。
B) 论文版图自动额外导出 PDF/SVG/EPS（extra_formats 默认含 "pdf"）。
C) 确保导出 PDF 时尽量为矢量（关闭 rasterized；如启用位图底图，PDF 仍会包含该位图）。

默认地图坐标为 0..3000 米（3km×3km）。若你的环境坐标不同，请调整 BG_EXTENT。
"""

import os
import sys
from pathlib import Path
import io
import re
import ast
import math
from typing import Optional

import numpy as np
import torch
from contextlib import redirect_stdout, redirect_stderr

# 避免 OpenMP 重复初始化
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# 无窗口后端，便于服务器/远程运行
os.environ.setdefault("MPLBACKEND", "Agg")

# 项目根目录
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(parent_dir)

from config_2actor import get_config
from envs.env_discrete_2actor import DiscreteActionEnv
from envs.env_wrappers_2actor import DummyVecEnv  # 你项目里已有

# ----------------------- 地图参数（固定 3km×3km） ----------------------- #
MAP_SIZE_METERS = 3000.0
GRID_MAJOR = 200.0
SP_VMAX = 400.0  # 颜色条最大值（0..400）

# 背景图默认参数
DEFAULT_BG_NAME = "figbackground.png"  # 你上传的图名
BG_EXTENT = [0.0, MAP_SIZE_METERS, 0.0, MAP_SIZE_METERS]  # [xmin, xmax, ymin, ymax]


def make_render_env(seed: int, debug: bool = True):
    """只起 1 个环境线程进行渲染；debug=True 以启用 EnvCore 的详细打印。"""

    def _init():
        env = DiscreteActionEnv(seed=seed, debug=debug)
        if hasattr(env, "seed"):
            env.seed(seed)
        return env

    return DummyVecEnv([_init])


def _pick_device(all_args):
    if getattr(all_args, "cuda", False) and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    torch.set_num_threads(max(1, int(getattr(all_args, "n_training_threads", 1))))
    return device


# -------------------------- 日志解析（兼容全量/精简） -------------------------- #
RESET_TR_RE = re.compile(r"\[RESET]\s*Tr-UAV\s*pos=\(([^)]+)\),\s*K=(\d+),\s*M=(\d+)")
RESET_SD_RE = re.compile(r"\[RESET]\s*SD\s*positions:\s*(.+)")
SD_ITEM_RE = re.compile(r"SD-(\d+)=\(([^)]+)\)")
RESET_DS_RE = re.compile(r"\[RESET]\s*DS\s*positions.*?:\s*(.+)")
DS_ITEM_RE = re.compile(r"DS-(\d+)=\(([^)]+)\)")

STEP_EQ_RE = re.compile(r"\[STEP]\s*slot=(\d+)")
STEP_DICT_RE = re.compile(r"\[STEP]\s*({.*})")  # log_full=True 时可命中

HOVER_EQ_RE = re.compile(r"\[SD_HOVER_THEN_FLY]\s*items=(\[.*?])", re.DOTALL)  # 精简/全量都可命中
HOVER_DICT_RE = re.compile(r"\[SD_HOVER_THEN_FLY]\s*({.*})", re.DOTALL)  # log_full=True 也可能是 dict，留兜底

QSTATE_DICT_RE = re.compile(r"\[Q_STATE]\s*({.*})", re.DOTALL)  # log_full=True：里面是可 literal_eval 的 dict

SDPOS_RE = re.compile(r"\[SD_POS]\s*slot=(\d+)\s*items=(\[.*\])", re.DOTALL)

SPREWARD_RE = re.compile(r"\[SP_REWARD]\s*({.*})", re.DOTALL)


def _parse_vec3(s: str):
    parts = [float(x.strip()) for x in s.split(",")]
    return tuple(parts)  # (x, y, z)


def _parse_sp_nz(sp_field):
    """
    EnvCore 在 log_full=True 时，SP_REWARD.sp_nz 是字符串摘要："{ 0:109, 1:17, ... }" 或 "∅"。
    也兼容传入 dict 的容错（虽然当前实现不会给 dict）。
    返回：dict[int->int]
    """
    if isinstance(sp_field, dict):
        return {int(k): int(v) for k, v in sp_field.items()}
    if not isinstance(sp_field, str):
        return {}
    s = sp_field.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return {}
    inner = s[1:-1].strip()
    if not inner:
        return {}
    out = {}
    for pair in inner.split(","):
        if ":" not in pair:
            continue
        k_str, v_str = pair.split(":", 1)
        try:
            k = int(k_str.strip())
            v = int(float(v_str.strip()))
            out[k] = v
        except Exception:
            pass
    return out


def parse_render_log(text: str):
    """
    输出：
      - tr_pos   (x,y,z)
      - K, M
      - sd_init  [(x,y,z), ...]
      - ds_pos   [(x,y,z), ...]
      - actions_by_slot: {slot: [{k,phi_deg,fly_t,hov_t,...}, ...]}
      - ds_rx_by_slot:  {slot: [M维 本槽接收量] }
      - sd_pos_by_slot: {slot: {k: (x,y)}}  # [SD_POS]
      - sp_by_slot:     {slot: dict{m->SP}} # 从 sp_nz 解析得到
      - sp_stats:       {slot: dict(avg_sp, max_sp)}
    """
    tr_pos = None
    K_val, M_val = None, None
    sd_init, ds_pos = [], []

    actions_by_slot = {}
    ds_rx_by_slot = {}
    sd_pos_by_slot = {}
    sp_by_slot = {}
    sp_stats = {}

    current_slot = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # RESET
        m = RESET_TR_RE.search(line)
        if m:
            tr_pos = _parse_vec3(m.group(1))
            K_val = int(m.group(2))
            M_val = int(m.group(3))
            continue
        m = RESET_SD_RE.search(line)
        if m:
            sd_items = SD_ITEM_RE.findall(m.group(1))
            sd_init = [_parse_vec3(v) for _, v in sorted(sd_items, key=lambda kv: int(kv[0]))]
            continue
        m = RESET_DS_RE.search(line)
        if m:
            ds_items = DS_ITEM_RE.findall(m.group(1))
            ds_pos = [_parse_vec3(v) for _, v in sorted(ds_items, key=lambda kv: int(kv[0]))]
            continue

        # STEP
        m = STEP_EQ_RE.search(line)
        if m:
            current_slot = int(m.group(1))
            continue
        m = STEP_DICT_RE.search(line)
        if m:
            try:
                obj = ast.literal_eval(m.group(1))
                if isinstance(obj, dict) and "slot" in obj:
                    current_slot = int(obj["slot"])
                    continue
            except Exception:
                pass

        # SD_HOVER_THEN_FLY
        m = HOVER_EQ_RE.search(line)
        if m:
            try:
                items = ast.literal_eval(m.group(1))
                if current_slot is None:
                    current_slot = (max(actions_by_slot) + 1) if actions_by_slot else 1
                actions_by_slot[current_slot] = items
                continue
            except Exception:
                pass
        m = HOVER_DICT_RE.search(line)
        if m:
            try:
                obj = ast.literal_eval(m.group(1))
                items = obj.get("items", None)
                if isinstance(items, list):
                    if current_slot is None:
                        current_slot = (max(actions_by_slot) + 1) if actions_by_slot else 1
                    actions_by_slot[current_slot] = items
                    continue
            except Exception:
                pass

        # Q_STATE（log_full=True 时为可 eval 的 dict）
        m = QSTATE_DICT_RE.search(line)
        if m and current_slot is not None:
            try:
                obj = ast.literal_eval(m.group(1))
                if isinstance(obj, dict) and "DS_rx" in obj:
                    arr = obj["DS_rx"]
                    if isinstance(arr, (list, tuple)):
                        ds_rx_by_slot[current_slot] = [int(x) for x in arr]
            except Exception:
                pass

        # SD_POS（最权威的 SD 坐标）
        m = SDPOS_RE.search(line)
        if m:
            try:
                s = int(m.group(1))
                items = ast.literal_eval(m.group(2))
                slot_dict = {}
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict) and "k" in it:
                            k = int(it["k"])
                            slot_dict[k] = (float(it.get("x", np.nan)),
                                            float(it.get("y", np.nan)))
                if slot_dict:
                    sd_pos_by_slot[s] = slot_dict
            except Exception:
                pass

        # SP_REWARD（sp_nz/avg_sp/max_sp）
        m = SPREWARD_RE.search(line)
        if m and current_slot is not None:
            try:
                obj = ast.literal_eval(m.group(1))
                sp_map = _parse_sp_nz(obj.get("sp_nz", "{}"))
                sp_by_slot[current_slot] = sp_map
                sp_stats[current_slot] = {
                    "avg_sp": float(obj.get("avg_sp", np.nan)),
                    "max_sp": float(obj.get("max_sp", np.nan)),
                }
            except Exception:
                pass

    return {
        "tr_pos": tr_pos, "K": K_val, "M": M_val,
        "sd_init": sd_init, "ds_pos": ds_pos,
        "actions_by_slot": actions_by_slot,
        "ds_rx_by_slot": ds_rx_by_slot,
        "sd_pos_by_slot": sd_pos_by_slot,
        "sp_by_slot": sp_by_slot, "sp_stats": sp_stats,
    }


def reconstruct_sd_paths(sd_init, actions_by_slot, fly_speed=25.0, sd_pos_by_slot=None):
    """
    返回:
      paths[k]:        [(x0,y0),...,(xT,yT)]
      slot_starts[k]:  [(x0,y0),...,(x{T-1},y{T-1})]
    变更：
      - 起点优先级：SD_POS@slot=0  > RESET  > 首个出现的 SD_POS 槽
      - k 集合来自日志里实际出现的 k
      - 有 SD_POS 则以其为准；缺失槽位时才按动作积分（保持连贯），否则原地
    """
    import numpy as _np
    sd_pos_by_slot = sd_pos_by_slot or {}
    all_slots = sorted(sd_pos_by_slot.keys()) if sd_pos_by_slot else []

    # ---- 确定 k 集合 & 起点来源 ----
    if 0 in sd_pos_by_slot and isinstance(sd_pos_by_slot[0], dict) and len(sd_pos_by_slot[0]) > 0:
        k_list = sorted(sd_pos_by_slot[0].keys())
        init_mode = "sdpos0"
    elif sd_init and len(sd_init) > 0:
        k_list = list(range(len(sd_init)))
        init_mode = "reset"
    elif all_slots:
        first_slot = all_slots[0]
        k_list = sorted(sd_pos_by_slot[first_slot].keys())
        init_mode = f"sdpos{first_slot}"
    else:
        return {}, {}

    # ---- 初始化路径 ----
    paths = {k: [] for k in k_list}
    slot_starts = {k: [] for k in k_list}

    for k in k_list:
        if init_mode == "sdpos0" and k in sd_pos_by_slot[0]:
            x, y = sd_pos_by_slot[0][k]
            paths[k].append((float(x), float(y)))
        elif init_mode == "reset" and k < len(sd_init):
            paths[k].append((float(sd_init[k][0]), float(sd_init[k][1])))
        else:
            first_slot = all_slots[0] if all_slots else None
            if first_slot is not None and k in sd_pos_by_slot[first_slot]:
                x, y = sd_pos_by_slot[first_slot][k]
                paths[k].append((float(x), float(y)))
            else:
                paths[k].append((_np.nan, _np.nan))

    # ---- 动作索引 & 槽推进 ----
    action_map = {s: {it.get("k"): it for it in items if isinstance(it, dict) and "k" in it}
                  for s, items in actions_by_slot.items()}
    max_slot = max(set(actions_by_slot.keys()) | set(sd_pos_by_slot.keys())) if (
            actions_by_slot or sd_pos_by_slot) else 0

    for slot in range(1, max_slot + 1):
        pos_dict = sd_pos_by_slot.get(slot, {})
        for k in k_list:
            last_x, last_y = paths[k][-1]

            if k in pos_dict:
                x, y = pos_dict[k]
                if _np.isfinite(x) and _np.isfinite(y):
                    slot_starts[k].append((last_x, last_y))
                    paths[k].append((float(x), float(y)))
                else:
                    slot_starts[k].append((_np.nan, _np.nan))
                    paths[k].append((_np.nan, _np.nan))
                continue

            # 无 SD_POS：按动作积分（若无动作则原地）
            it = action_map.get(slot, {}).get(k, None)
            slot_starts[k].append((last_x, last_y))
            if it is None:
                paths[k].append((last_x, last_y))
                continue

            fly_t = float(it.get("fly_t", 0.0))
            phi_deg = float(it.get("phi_deg", 0.0))
            if fly_t > 1e-12:
                dist = fly_t * float(fly_speed)
                rad = math.radians(phi_deg)
                dx = dist * math.cos(rad)  # 0°→ +x
                dy = dist * math.sin(rad)
                paths[k].append((last_x + dx, last_y + dy))
            else:
                paths[k].append((last_x, last_y))

    return paths, slot_starts


def _clamp_bounds_to_map():
    return 0.0, MAP_SIZE_METERS, 0.0, MAP_SIZE_METERS


# ------------------------------ 背景图工具 ------------------------------ #
def _maybe_load_bg(bg_path: Optional[str]):
    """若提供路径且可读，返回 (img_array, True)，否则 (None, False)。"""
    if not bg_path:
        return None, False
    p = Path(bg_path)
    if not p.exists():
        return None, False
    try:
        import matplotlib.image as mpimg
        img = mpimg.imread(str(p))
        return img, True
    except Exception:
        return None, False


def _draw_bg(ax, bg_img, bg_alpha: float = 1, origin: str = "upper"):
    """将背景图绘制在指定 axes 下方。"""
    if bg_img is None:
        return
    ax.imshow(bg_img, extent=BG_EXTENT, origin=origin, alpha=float(bg_alpha), zorder=0)


# ------------------------------ 绘图/导出 ------------------------------ #
def _setup_style():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    # 样式
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn-whitegrid')
        except Exception:
            plt.style.use('default')

    # ---- 关键：选一个支持中文的字体 ----
    cjk_candidates = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "Source Han Sans SC", "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    picked = None
    for name in cjk_candidates:
        if name in installed:
            picked = name
            break

    if picked is not None:
        mpl.rcParams["font.sans-serif"] = [picked] + mpl.rcParams.get("font.sans-serif", [])
    else:
        # 若无中文字体，不强制；避免崩溃
        pass

    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["axes.facecolor"] = "#FAFAFA"
    mpl.rcParams["figure.facecolor"] = "#FFFFFF"
    mpl.rcParams["grid.alpha"] = 0.15
    mpl.rcParams["grid.color"] = "#E0E0E0"
    mpl.rcParams["savefig.facecolor"] = "#FFFFFF"
    mpl.rcParams.update({
        "figure.dpi": 110,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def _nice_axes(ax):
    from matplotlib.patches import Rectangle
    xmin, xmax, ymin, ymax = _clamp_bounds_to_map()
    rect = Rectangle((0, 0), MAP_SIZE_METERS, MAP_SIZE_METERS,
                     fill=False, linewidth=2.0, linestyle="-", alpha=0.95)
    ax.add_patch(rect)
    xticks = np.arange(xmin, xmax + 1e-6, GRID_MAJOR)
    yticks = np.arange(ymin, ymax + 1e-6, GRID_MAJOR)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)


def _setup_paper_style():
    """
    论文版全局样式（高分辨率、统一字号/线宽；便于 PNG/SVG/EPS/PDF 直接投稿）。
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.style.use('default')
    mpl.rcParams.update({
        # 分辨率与导出
        "figure.dpi": 400,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # 颜色与底色
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",

        # 字体与字号
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei"],
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,

        # 轴外观
        "axes.linewidth": 0.9,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,

        # 默认图谱
        "image.cmap": "viridis",
        "legend.frameon": False,
    })


def draw_paper_figure(
        out_dir: Path,
        parsed,
        paths,
        paper_png: str = "trajectory_map_3.png",
        delta_T: float = 2.0,
        figsize=(5.0, 5.0),
        show_legend: bool = True,
        major_tick: float = 600.0,
        # 这里默认就会另外导出 pdf/svg/eps
        extra_formats=("pdf", "svg", "eps"),
        traj_pt_size: int = 8,
        sd_start_style: str = "star",
        # 图例大小控制
        legend_fontsize: float = 10,
        legend_markerscale: float = 1.4,
        legend_ncol: Optional[int] = None,
        # 非图例标记大小控制
        ds_marker_size="auto",
        sd_start_size: int = 24,
        sd_end_size: int = 26,
        tr_start_size: int = 120,
        size_scale: float = 1.0,
        # 背景图参数
        bg_img=None,
        bg_alpha: float = 0.85,
        bg_origin: str = "upper",
):
    """
    Paper-style overview (final frame) with tunable legend & non-legend marker sizes,
    and optional background overlay.
    """
    import numpy as _np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from pathlib import Path as _Path

    _setup_paper_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------- data -------
    ds_pos = parsed.get("ds_pos") or []
    sp_by_slot = parsed.get("sp_by_slot", {})
    K = parsed.get("K", len(parsed.get("sd_init") or []))
    M = len(ds_pos)
    tr = parsed.get("tr_pos") or (float("nan"), float("nan"), float("nan"))

    # final frame index
    T_from_paths = max((len(v) for v in paths.values()), default=1) - 1
    T_from_sdpos = max(parsed.get("sd_pos_by_slot", {}).keys()) if parsed.get("sd_pos_by_slot") else 0
    T_from_splog = max(sp_by_slot.keys()) if sp_by_slot else 0
    upto_slot = max(T_from_paths, T_from_sdpos, T_from_splog, 0)

    # colormap for DS (SP)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=float(SP_VMAX))
    try:
        cmap = mpl.colormaps["viridis"]
    except Exception:
        cmap = mpl.cm.get_cmap("viridis")

    # per-SD solid colors (no gradient)
    soft_colors = [
        "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
        "#EECA3B", "#B279A2", "#FF9DA6", "#9C755F", "#BAB0AC",
        "#8DA0CB", "#66C2A5", "#FC8D62", "#E78AC3", "#A6D854", "#FFD92F"
    ]

    _style_map = {
        "filled_circle": ("o", "traj"),
        "filled_square": ("s", "traj"),
        "filled_diamond": ("D", "traj"),
        "triangle_up": ("^", "traj"),
        "triangle_down": ("v", "traj"),
        "triangle_left": ("<", "traj"),
        "triangle_right": (">", "traj"),
        "star": ("*", "traj"),
        "pentagon": ("p", "traj"),
        "hexagon": ("h", "traj"),
        "plus": ("P", "traj"),
    }
    _marker, _fc_mode = _style_map.get(sd_start_style, ("D", "traj"))

    def _facecolor_for(col_hex: str):
        return col_hex if _fc_mode == "traj" else _fc_mode

    def _sp_vec_for_slot(slot_idx: int, M_show: int):
        vec = _np.zeros(M_show, dtype=float)
        if slot_idx in sp_by_slot:
            for m, v in sp_by_slot[slot_idx].items():
                if 0 <= m < M_show:
                    vec[m] = float(v)
        return vec

    # ------- draw -------
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111)

    # frame & axes
    xmin, xmax, ymin, ymax = 0.0, MAP_SIZE_METERS, 0.0, MAP_SIZE_METERS
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    # 叠加背景图（先画）
    _draw_bg(ax, bg_img, bg_alpha=bg_alpha, origin=bg_origin)

    # 边框与刻度
    ax.add_patch(Rectangle((0, 0), MAP_SIZE_METERS, MAP_SIZE_METERS,
                           fill=False, linewidth=1.0, linestyle="-", edgecolor="black"))
    xticks = _np.arange(xmin, xmax + 1e-6, float(major_tick))
    yticks = _np.arange(ymin, ymax + 1e-6, float(major_tick))
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)

    # === DS scatter + colorbar（大小可控） ===
    if ds_pos:
        spv = _sp_vec_for_slot(upto_slot, M)
        xs = [p[0] for p in ds_pos]
        ys = [p[1] for p in ds_pos]

        # 自适应或固定
        if ds_marker_size in (None, "auto"):
            _auto = int(_np.clip(22 - 0.10 * M, 7, 16))
            ds_size = max(1, int(_auto * float(size_scale)))
        else:
            ds_size = max(1, int(float(ds_marker_size) * float(size_scale)))

        edge_w = 0.35 if ds_size >= 12 else 0.25
        # PDF 矢量：关闭 rasterized
        ds_sc = ax.scatter(xs, ys, c=spv, cmap=cmap, norm=norm,
                           s=ds_size, linewidths=edge_w, edgecolors="#5C5C5C",
                           alpha=0.95, zorder=3, rasterized=False)
        cbar = fig.colorbar(ds_sc, ax=ax, fraction=0.048, pad=0.02)
        cbar.set_label("SP")

    # === SD trajectories + start/end（大小可控） ===
    SD_START_SIZE = max(1, int(sd_start_size * float(size_scale)))
    SD_END_SIZE = max(1, int(sd_end_size * float(size_scale)))
    TRAJ_PT_SIZE = max(1, int(traj_pt_size * float(size_scale)))

    for k, pts in sorted(paths.items(), key=lambda kv: kv[0]):
        col = soft_colors[k % len(soft_colors)]
        upto_slot_clamped = max(0, min(upto_slot, len(pts) - 1))
        pts_arr = _np.asarray(pts[:upto_slot_clamped + 1], dtype=float)
        xs = pts_arr[:, 0]
        ys = pts_arr[:, 1]
        finite = _np.isfinite(xs) & _np.isfinite(ys)
        if _np.any(finite):
            ax.scatter(xs[finite], ys[finite],
                       s=TRAJ_PT_SIZE, color=col,
                       edgecolors='white', linewidths=0.01,
                       alpha=0.95, zorder=4, rasterized=False)
            s_idx = int(_np.where(finite)[0][0])
            ax.scatter([xs[s_idx]], [ys[s_idx]],
                       s=SD_START_SIZE, marker=_marker,
                       facecolors=_facecolor_for(col), edgecolors="black",
                       linewidths=0.85, zorder=6)
            e_idx = int(_np.where(finite)[0][-1])
            ax.scatter([xs[e_idx]], [ys[e_idx]],
                       s=SD_END_SIZE, marker="X",
                       edgecolors="black", linewidths=0.9, c=col, zorder=7)

    # === Tr-UAV start（大小可控） ===
    has_tr = False
    try:
        tx, ty = float(tr[0]), float(tr[1])
        if _np.isfinite(tx) and _np.isfinite(ty):
            TR_SIZE = max(1, int(tr_start_size * float(size_scale)))
            ax.scatter([tx], [ty], marker="^", s=TR_SIZE,
                       edgecolors="white", linewidths=1.2,
                       c="#4682B4", alpha=0.98, zorder=8, rasterized=False)
            has_tr = True
    except Exception:
        has_tr = False

    # Legend
    if show_legend:
        from matplotlib.lines import Line2D
        start_handle = Line2D([0], [0], marker=_marker, linestyle='None',
                              markerfacecolor='#808080', markeredgecolor='black',
                              markeredgewidth=0.9, markersize=6, label='SD-UAV start')
        handles = [
            Line2D([0], [0], marker='o', linestyle='None',
                   markerfacecolor='#808080', markeredgecolor='#5C5C5C',
                   markeredgewidth=0.6, markersize=5, label='DS (SP-colored)'),
            start_handle,
            Line2D([0], [0], marker='X', linestyle='None',
                   markerfacecolor='black', markeredgecolor='black',
                   markersize=6, label='SD-UAV end'),
        ]
        if has_tr:
            handles.append(
                Line2D([0], [0], marker='^', linestyle='None',
                       markerfacecolor='#4682B4', markeredgecolor='white',
                       markeredgewidth=1.0, markersize=7, label='Tr-UAV location')
            )
        for i in range(int(K or 0)):
            handles.append(
                Line2D([0], [0], marker='o', linestyle='None',
                       markerfacecolor=soft_colors[i % len(soft_colors)],
                       markeredgecolor='white', markeredgewidth=0.4,
                       markersize=5, label=f'SD-UAV (k={i + 1}) trajectory')
            )
        ncol_eff = (legend_ncol if legend_ncol is not None else (2 if (K and K > 6) else 1))
        leg = ax.legend(handles=handles,
                        loc="upper left",
                        bbox_to_anchor=(0.015, 0.985),
                        bbox_transform=ax.transAxes,
                        ncol=ncol_eff,
                        frameon=True,
                        fontsize=legend_fontsize,
                        markerscale=legend_markerscale)
        leg.get_frame().set_facecolor('#F5F5F5')
        leg.get_frame().set_edgecolor('#C0C0C0')

    # 在保存前，确保没有集合被标记为 rasterized（保险）
    for coll in ax.collections:
        try:
            coll.set_rasterized(False)
        except Exception:
            pass

    # save
    out_path = out_dir / paper_png
    fig.savefig(out_path)
    for fmt in (extra_formats or []):
        try:
            fig.savefig(out_dir / (_Path(paper_png).with_suffix(f".{fmt}").name))
        except Exception:
            pass
    plt.close(fig)
    return out_path


def draw_overview_and_gif(
        out_dir: Path,
        parsed,
        paths,
        slot_starts,
        gif_out="uav_motion.gif",
        overview_png="system_overview.png",
        fps=6,
        annotate_every=0,
        delta_T=2.0,
        show_title=True,
        # 背景图参数
        bg_img=None,
        bg_alpha: float = 0.85,
        bg_origin: str = "upper",
        # 新增：是否生成 GIF（默认 False）
        make_gif: bool = False,
):
    """
    DS 着色：按“当槽位的 SP（0..300）”，并在本槽接收过的 DS 外圈高亮。
    支持底图叠加。
    """
    import matplotlib as mpl

    _setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = parsed.get("tr_pos") or (np.nan, np.nan, np.nan)
    K = parsed.get("K", None)
    M = parsed.get("M", None)
    K_infer = len(parsed.get("sd_init") or [])
    M_infer = len(parsed.get("ds_pos") or [])

    ds_pos = parsed.get("ds_pos") or []
    ds_rx_by_slot = parsed.get("ds_rx_by_slot", {})
    sp_by_slot = parsed.get("sp_by_slot", {})
    sp_stats = parsed.get("sp_stats", {})

    # 总时长（槽数）
    T_from_paths = max((len(v) for v in paths.values()), default=1) - 1
    T_from_rx = max(ds_rx_by_slot.keys()) if ds_rx_by_slot else 0
    T_from_sdpos = max(parsed.get("sd_pos_by_slot", {}).keys()) if parsed.get("sd_pos_by_slot") else 0
    T_from_splog = max(sp_by_slot.keys()) if sp_by_slot else 0
    T_slots = max(T_from_paths, T_from_rx, T_from_sdpos, T_from_splog, 0)

    norm = mpl.colors.Normalize(vmin=0.0, vmax=float(SP_VMAX))
    try:
        cmap = mpl.colormaps["viridis"]
    except Exception:
        cmap = mpl.cm.get_cmap("viridis")

    soft_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B", "#B279A2", "#FF9DA6"]

    def _sp_vec_for_slot(upto_slot: int, M_show: int):
        vec = np.zeros(M_show, dtype=float)
        if upto_slot in sp_by_slot:
            d = sp_by_slot[upto_slot]
            for m, v in d.items():
                if 0 <= m < M_show:
                    vec[m] = float(v)
        return vec

    def _draw_frame(fig, gs, upto_slot: int):
        from matplotlib.patches import Rectangle

        ax = fig.add_subplot(gs[0, 0])
        info_ax = fig.add_subplot(gs[0, 1])
        info_ax.set_axis_off()

        # 坐标轴与边框
        xmin, xmax, ymin, ymax = _clamp_bounds_to_map()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        # 先画背景
        _draw_bg(ax, bg_img, bg_alpha=bg_alpha, origin=bg_origin)

        # 再画边框与网格
        ax.add_patch(Rectangle((0, 0), MAP_SIZE_METERS, MAP_SIZE_METERS,
                               fill=False, linewidth=2.0, linestyle="-", edgecolor="black", alpha=0.95))
        xticks = np.arange(xmin, xmax + 1e-6, GRID_MAJOR)
        yticks = np.arange(ymin, ymax + 1e-6, GRID_MAJOR)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.set_xlabel("x (m)", fontsize=12)
        ax.set_ylabel("y (m)", fontsize=12)

        # ----- DS 散点 + 颜色条 -----
        if ds_pos:
            M_show = len(ds_pos)
            spv = _sp_vec_for_slot(upto_slot, M_show)
            xs = [p[0] for p in ds_pos]
            ys = [p[1] for p in ds_pos]
            ds_sc = ax.scatter(xs, ys, c=spv, cmap=cmap, norm=norm,
                               s=60, linewidths=1.0, edgecolors="#808080", alpha=0.92,
                               zorder=3)
            # 本槽收到数据的 DS（方形空心高亮）
            if upto_slot in ds_rx_by_slot:
                cur = np.array(ds_rx_by_slot[upto_slot], dtype=int)[:M_show]
                mask = cur > 0
                if np.any(mask):
                    ax.scatter(np.array(xs)[mask], np.array(ys)[mask],
                               facecolors="none", edgecolors="#FF8C69",
                               s=160, linewidths=2.6, marker='s', alpha=0.95,
                               zorder=4)

            cbar = fig.colorbar(ds_sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f'SP', rotation=90, fontsize=12)

        # ----- Tr-UAV 起始点（若提供） -----
        if not math.isnan(tr[0]):
            ax.scatter([tr[0]], [tr[1]], marker="^", s=220,
                       edgecolors='white', linewidths=2.0,
                       c='#4682B4', alpha=0.96, zorder=6)

        # ====== 仅“点”的轨迹绘制（不连线）+ 起点/终点标记 ======
        TRAJ_PT_SIZE = 12  # 轨迹点
        SD_START_SIZE = 36  # SD 起点
        SD_END_SIZE = 44  # SD 终点

        for k, pts in sorted(paths.items(), key=lambda kv: kv[0]):
            col = soft_colors[k % len(soft_colors)]
            upto_slot_clamped = max(0, min(upto_slot, len(pts) - 1))
            pts_arr = np.asarray(pts[:upto_slot_clamped + 1], dtype=float)
            xs = pts_arr[:, 0]
            ys = pts_arr[:, 1]
            finite = np.isfinite(xs) & np.isfinite(ys)

            if np.any(finite):
                # 轨迹点（不连线）
                ax.scatter(xs[finite], ys[finite],
                           s=TRAJ_PT_SIZE, color=col,
                           edgecolors='white', linewidths=0.01,
                           alpha=0.95, zorder=4, rasterized=False)
                # 起点 = 第一个有限值（空心圆）
                start_idx = int(np.where(finite)[0][0])
                ax.scatter([xs[start_idx]], [ys[start_idx]], s=SD_START_SIZE, color=col,
                           marker="o", facecolors="none", edgecolors="black",
                           linewidths=1.0, zorder=6)
                # 终点 = 当前帧最后一个有限值（X）
                end_idx = int(np.where(finite)[0][-1])
                ax.scatter([xs[end_idx]], [ys[end_idx]], s=SD_END_SIZE, color=col,
                           marker="X", edgecolors="black", linewidths=0.9, zorder=7)

        # —— 标题 —— #
        t_elapsed = upto_slot * float(delta_T)
        if show_title:
            ax.set_title(f"Multi-UAV Delivery | slot {upto_slot}  (t={t_elapsed:.1f}s)")

        # ===== 右栏信息 =====
        info_ax.set_axis_off()
        M_show = M if M is not None else M_infer
        spv = np.zeros(M_show, dtype=float)
        if upto_slot in sp_by_slot:
            for m, v in sp_by_slot[upto_slot].items():
                if 0 <= m < M_show:
                    spv[m] = float(v)
        avg_sp = sp_stats.get(upto_slot, {}).get("avg_sp", float(np.mean(spv)) if spv.size else np.nan)
        max_sp = sp_stats.get(upto_slot, {}).get("max_sp", float(np.max(spv)) if spv.size else np.nan)

        top_lines = ["  (no SP parsed)"]
        if spv.size > 0:
            order = np.argsort(-spv)[:min(10, spv.size)]
            top_lines = [f"  DS{int(i):02d}: {int(spv[i])}" for i in order]

        info_lines = []
        info_lines.append(f"[TIME]  Slot: {upto_slot}/{T_slots}   Δt={delta_T:.1f}s   t={t_elapsed:.1f}s")
        info_lines.append(f"[MAP]   0..{int(MAP_SIZE_METERS)} m (fixed)")
        info_lines.append(f"[AGENTS] SD-UAV: {K if K is not None else K_infer}   DS: {M_show}")
        if spv.size > 0:
            info_lines.append(f"[SP]    avg={avg_sp:.2f}   max={max_sp:.0f}   (range 0..{int(SP_VMAX)})")
        else:
            info_lines.append(f"[SP]    (no data)")
        info_lines.append("")
        info_lines.append("Top DS by SP:")
        info_lines.extend(top_lines)

        items = parsed.get("actions_by_slot", {}).get(upto_slot, [])
        if items:
            info_lines.append("")
            info_lines.append("SD-UAV Actions (phi, fly, hov):")
            for it in items:
                if not isinstance(it, dict):
                    continue
                kk = it.get("k", "?")
                ph = it.get("phi_deg", None)
                ft = it.get("fly_t", None)
                ht = it.get("hov_t", None)
                info_lines.append(f"  SD-UAV{kk}: φ={ph if ph is not None else '-'}°, fly={ft}, hov={ht}")

        full_text = "\n".join(info_lines)
        info_ax.text(0.02, 0.98, full_text,
                     ha="left", va="top", fontsize=9, family='monospace',
                     bbox=dict(boxstyle="round,pad=0.55",
                               facecolor="#F8F8FF", alpha=0.96, edgecolor="#D3D3D3"))

    # GIF（可选）
    gif_path = None
    if make_gif:
        import imageio.v2 as imageio
        gif_path = out_dir / gif_out
        with imageio.get_writer(gif_path, mode="I",
                                duration=1.0 / max(1, int(fps)), loop=0) as writer:
            for s in range(T_slots + 1):
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(12, 7), constrained_layout=True)
                gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[3, 1])
                _draw_frame(fig, gs, upto_slot=s)
                fig.canvas.draw()
                rgba = np.asarray(fig.canvas.buffer_rgba())
                frame = rgba[:, :, :3]
                writer.append_data(frame)
                plt.close(fig)

    # 总览 PNG（最后一帧）
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[3, 1])
    _draw_frame(fig, gs, upto_slot=T_slots)
    overview_path = out_dir / overview_png
    fig.savefig(overview_path, dpi=160)
    plt.close(fig)

    return gif_path, overview_path


# ==== NEW (slots version): time-series helpers (avg SP / avg RX) ====
def _save_timeseries_csv(out_dir: Path, base: str, slots, values):
    out_dir.mkdir(parents=True, exist_ok=True)
    import pandas as _pd
    df = _pd.DataFrame({"slot": slots, base: values})
    csv_path = out_dir / f"{base}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def _plot_timeseries_paper_slots(out_dir: Path, base: str, title: str,
                                 slots, values, ylabel: str):
    _setup_paper_style()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5.2, 3.2), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(slots, values, linewidth=1.6)
    ax.set_xlabel("Time slot (T)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=6)
    # 浅网格、去右上框
    ax.grid(True, linestyle="--", alpha=0.25)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{base}.png", dpi=300)
    fig.savefig(out_dir / f"{base}.pdf")
    fig.savefig(out_dir / f"{base}.svg")
    fig.savefig(out_dir / f"{base}.eps")
    plt.close(fig)


def build_avg_sp_series(parsed: dict):
    """
    返回：slots(1D int)、avg_sp(1D float)
    优先使用 parsed['sp_stats'][slot]['avg_sp']；若缺失，则由 sp_by_slot 求均值。
    """
    import numpy as _np
    sp_stats = parsed.get("sp_stats", {}) or {}
    sp_by_slot = parsed.get("sp_by_slot", {}) or {}

    all_slots = sorted(set(sp_stats.keys()) | set(sp_by_slot.keys()))
    if not all_slots:
        return _np.array([], dtype=int), _np.array([], dtype=float)

    avg_list = []
    for s in all_slots:
        if s in sp_stats and "avg_sp" in sp_stats[s] and _np.isfinite(sp_stats[s]["avg_sp"]):
            avg_list.append(float(sp_stats[s]["avg_sp"]))
        else:
            d = sp_by_slot.get(s, {})
            if d:
                vals = _np.array(list(d.values()), dtype=float)
                avg_list.append(float(_np.mean(vals)))
            else:
                avg_list.append(_np.nan)
    slots = _np.asarray(all_slots, dtype=int)
    avg_sp = _np.asarray(avg_list, dtype=float)
    return slots, avg_sp


def build_avg_rx_series(parsed: dict, reduce: str = "mean"):
    """
    返回：slots(1D int)、avg_rx(1D float)
    reduce 支持 "mean"（默认）或 "sum"（改为总接收量时用）。
    """
    import numpy as _np
    rx_map = parsed.get("ds_rx_by_slot", {}) or {}
    if not rx_map:
        return _np.array([], dtype=int), _np.array([], dtype=float)

    slots = sorted(rx_map.keys())
    vals = []
    for s in slots:
        arr = _np.asarray(rx_map.get(s, []), dtype=float)
        if arr.size == 0:
            vals.append(_np.nan)
        else:
            vals.append(float(_np.mean(arr) if reduce == "mean" else _np.sum(arr)))
    slots = _np.asarray(slots, dtype=int)
    avg_rx = _np.asarray(vals, dtype=float)
    return slots, avg_rx


# ==== END NEW ====


# ------------------------------- 主流程 ------------------------------- #
def main(argv):
    parser = get_config()
    # 出图相关参数（不影响训练/评估）
    parser.add_argument("--gif_out", type=str, default="uav_motion.gif",
                        help="渲染后输出的 GIF 文件名（run_dir/out 下）")
    parser.add_argument("--overview_png", type=str, default="system_overview.png",
                        help="渲染后输出的总览 PNG 文件名（run_dir/out 下）")
    parser.add_argument("--gif_fps", type=int, default=6, help="GIF 帧率（帧/秒）")
    parser.add_argument("--make_gif", action="store_true",
                        help="启用后才生成 GIF 动图（默认不生成）")
    parser.add_argument("--annotate_every", type=int, default=0, help="槽号标注步进（0 不标注）")
    parser.add_argument("--no_title", action="store_true", help="不在左图上方显示标题")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=r"../results/SCEnv/2actor/Discrete/r_mappo/check/run2/models",
        help="path to pretrained model directory.",
    )
    # ===== 新增：背景图参数 =====
    parser.add_argument("--bg_image", type=str, default=DEFAULT_BG_NAME,
                        help="背景图像文件路径（PNG/JPG/TIF 等），为空则不叠加")
    parser.add_argument("--bg_alpha", type=float, default=0.85,
                        help="背景图透明度（0~1）")
    parser.add_argument("--bg_origin", type=str, default="upper",
                        choices=["upper", "lower"],
                        help="背景图原点位置（upper=左上为(0,0)；lower=左下为(0,0)）")

    all_args = parser.parse_args(argv)
    assert all_args.model_dir, "请使用 --model_dir 指定包含 actor.pt/critic.pt 的目录或其父目录"

    # 渲染相关强制参数
    all_args.use_render = True
    all_args.n_rollout_threads = 1
    all_args.n_render_rollout_threads = 1
    all_args.save_gifs = False
    all_args.env_name = "Discrete"
    all_args.scenario_name = "Discrete"

    _ = _pick_device(all_args)
    torch.manual_seed(all_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # 创建单线程渲染环境（debug=True 以打印全量日志）
    envs = make_render_env(all_args.seed, debug=True)

    # 读取 agent 数
    env0 = envs.envs[0]
    num_agents = int(getattr(env0, "num_agent",
                             getattr(env0, "agent_num",
                                     getattr(env0, "num_agents", 0))))
    assert num_agents >= 1, f"expect num_agents>=1, got {num_agents}"
    all_args.num_agents = num_agents

    # === 环境优先：读取 delta_T 与 v_speed（若提供则覆盖 CLI） ===
    def _probe_attr(env, names):
        objs = [env]
        for attr in ("unwrapped", "env", "core"):
            o = getattr(objs[-1], attr, None)
            if o is not None and o is not objs[-1]:
                objs.append(o)
        for o in objs:
            for n in names:
                if hasattr(o, n):
                    v = getattr(o, n)
                    try:
                        v = float(v)
                        if np.isfinite(v) and v > 0:
                            return v
                    except Exception:
                        pass
        return None

    dt_env = _probe_attr(env0, ["delta_T", "Delta_T", "slot_dt", "slot_T"])
    vs_env = _probe_attr(env0, ["v_speed", "fly_speed", "sd_speed", "speed_mps", "speed"])

    if dt_env is not None:
        all_args.delta_T = dt_env
    if vs_env is not None:
        all_args.fly_speed = vs_env

    print(f"[render] using values -> delta_T={all_args.delta_T}s, fly_speed={all_args.fly_speed}m/s")

    # 运行目录（日志输出到模型目录上层，避免覆盖训练日志）
    model_dir = Path(all_args.model_dir)

    # 1) 严格校验 model_dir 必须存在
    if not model_dir.exists():
        raise FileNotFoundError(
            f"[render] 指定的 --model_dir 不存在：{model_dir}\n"
            f"请确认路径是否正确。"
        )

    # 2) 推导 run_dir（与原逻辑一致），但要求其必须存在
    if (model_dir / "actor.pt").exists() or (model_dir / "critic.pt").exists():
        run_dir = model_dir.parent
    elif (model_dir / "models").exists():
        run_dir = model_dir
    else:
        run_dir = model_dir.parent

    if not run_dir.exists():
        raise FileNotFoundError(
            f"[render] 推导出的运行目录 run_dir 不存在：{run_dir}\n"
            f"请检查 --model_dir 的层级是否正确。"
        )

    # 3) out 目录必须预先存在（若你仍想自动创建，可去掉此检查）
    out_dir = run_dir / "out"
    if not out_dir.exists():
        raise FileNotFoundError(
            f"[render] 输出目录不存在：{out_dir}\n"
            f"请手动创建该目录后重试。"
        )

    # 载入 Runner 并执行渲染
    from runner.env_runner_2actor import EnvRunner as Runner
    runner = Runner(dict(
        all_args=all_args,
        envs=envs,
        eval_envs=None,
        render_envs=envs,
        num_agents=num_agents,
        device=torch.device("cuda:0") if (
                getattr(all_args, "cuda", False) and torch.cuda.is_available()) else torch.device("cpu"),
        run_dir=run_dir,
    ))

    print("\n========== 渲染配置 ==========")
    print(f"模型目录       : {all_args.model_dir}")
    print(f"渲染回合数     : {getattr(all_args, 'render_episodes', 1)}")
    print(f"使用GPU        : {all_args.cuda and torch.cuda.is_available()}")
    print(f"随机种子(seed) : {all_args.seed}")
    if getattr(all_args, "make_gif", False):
        print(f"GIF 输出       : {all_args.gif_out}, fps={all_args.gif_fps}")
    else:
        print(f"GIF 输出       : disabled (use --make_gif to enable)")
    print(f"总览图输出     : {all_args.overview_png}")
    print(f"示意速度       : {all_args.fly_speed} m/s")
    print(f"起点标注间隔   : {all_args.annotate_every} 槽（0 不标注）")
    print(f"delta_T(显示)  : {all_args.delta_T} s")
    print(f"地图范围       : 0..{MAP_SIZE_METERS} m（固定）")
    print(f"背景图         : {all_args.bg_image or '(disabled)'}")
    print(f"背景透明度     : {all_args.bg_alpha}, origin={all_args.bg_origin}")
    print("================================\n")

    # 捕获渲染日志
    print(">>> 开始渲染 ...")
    buf_out, buf_err = io.StringIO(), io.StringIO()
    try:
        with torch.no_grad(), redirect_stdout(buf_out), redirect_stderr(buf_err):
            runner.render()
    finally:
        envs.close()

    # 保存/回显日志
    full_log = buf_out.getvalue() + "\n" + buf_err.getvalue()
    print(full_log)
    (out_dir / "render_log.txt").write_text(full_log, encoding="utf-8")

    print(">>> 解析日志并重建轨迹 ...")
    parsed = parse_render_log(full_log)
    # ==== NEW (slots version): 构建两条时间槽序列 ====
    sp_slots, sp_vals = build_avg_sp_series(parsed)
    rx_slots, rx_vals = build_avg_rx_series(parsed, reduce="mean")  # 若要总接收量改为 "sum"

    # 保存 CSV（列：slot, value）
    if sp_slots.size:
        _save_timeseries_csv(out_dir, "avg_sp_timeseries", sp_slots, sp_vals)
    if rx_slots.size:
        _save_timeseries_csv(out_dir, "avg_rx_timeseries", rx_slots, rx_vals)

    # 出图（PNG/PDF/SVG/EPS），横轴为“时间槽 T”
    if sp_slots.size:
        _plot_timeseries_paper_slots(out_dir, "avg_sp_vs_slot",
                                     "Average SP over Time Slots",
                                     sp_slots, sp_vals, ylabel="Average SP")
    if rx_slots.size:
        _plot_timeseries_paper_slots(out_dir, "avg_rx_vs_slot",
                                     "Average Received Data over Time Slots",
                                     rx_slots, rx_vals, ylabel="Average received per DS")

    paths, slot_starts = reconstruct_sd_paths(
        parsed.get("sd_init", []),
        parsed.get("actions_by_slot", {}),
        fly_speed=float(all_args.fly_speed),
        sd_pos_by_slot=parsed.get("sd_pos_by_slot", {})
    )

    # 加载背景图（一次）
    bg_img, ok = _maybe_load_bg(all_args.bg_image)
    if not ok and all_args.bg_image:
        print(f"[WARN] 背景图未找到或无法读取：{all_args.bg_image}，将忽略底图。")

    print(">>> 生成动图与总览图 ...")
    gif_path, png_path = draw_overview_and_gif(
        out_dir=out_dir,
        parsed=parsed,
        paths=paths,
        slot_starts=slot_starts,
        gif_out=str(all_args.gif_out),
        overview_png=str(all_args.overview_png),
        fps=int(all_args.gif_fps),
        annotate_every=int(all_args.annotate_every),
        delta_T=float(all_args.delta_T),
        show_title=(not getattr(all_args, "no_title", False)),
        # 背景图参数
        bg_img=bg_img,
        bg_alpha=float(all_args.bg_alpha),
        bg_origin=str(all_args.bg_origin),
        # 是否生成 GIF
        make_gif=bool(getattr(all_args, "make_gif", False)),
    )

    # 论文版单图
    paper_png_path = draw_paper_figure(
        out_dir=out_dir, parsed=parsed, paths=paths,
        figsize=(6, 6), show_legend=True,
        # — 非图例大小控制 —
        ds_marker_size=20,
        traj_pt_size=7,
        sd_start_size=60,
        sd_end_size=45,
        tr_start_size=90,
        size_scale=1.0,
        # — 图例大小 —
        legend_fontsize=7,
        legend_markerscale=1.0,
        legend_ncol=1,
        # 背景图参数
        bg_img=bg_img,
        bg_alpha=float(all_args.bg_alpha),
        bg_origin=str(all_args.bg_origin),
    )

    paper_pdf_path = paper_png_path.with_suffix(".pdf")

    print(f">>> 论文版总览图（PNG） ：{paper_png_path}")
    print(f">>> 论文版 PDF         ：{paper_pdf_path}")
    if gif_path is not None:
        print(f">>> GIF 动图            ：{gif_path}")
    else:
        print(f">>> GIF 动图            ：已禁用（使用 --make_gif 可启用）")
    print(f">>> 总览图（PNG）       ：{png_path}")
    print(f">>> 日志               ：{out_dir / 'render_log.txt'}")


if __name__ == "__main__":
    main(sys.argv[1:])
