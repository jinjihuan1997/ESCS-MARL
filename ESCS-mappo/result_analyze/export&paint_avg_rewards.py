#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Iterable, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.ticker import FuncFormatter, FixedLocator
from tensorboard.backend.event_processing import event_accumulator as ea

# ========== 示例“最终地址”（可为事件文件或其所在目录） ==========
EXAMPLE_PATH = r"/home/king/Downloads/Projects/TMC/light_mappo_new/results/SCEnv/1actor/Discrete/r_mappo/check/run1/logs/average_episode_rewards/average_episode_rewards/events.out.tfevents.1758219890.DESKTOP-6I941IT"
TAG = "average_episode_rewards"
OUTDIR = "plots_paper_raw"
# ================================================================

# —— 全局默认（未单独指定时使用） ——
FIG_WIDTH_IN = 8.0
FIG_ASPECT = 0.70
X_MIN, X_MAX = 0, 3_100_000
Y_MIN, Y_MAX = -2500, 4000
X_MAJOR_STEP = 500_000
X_MINOR_STEP = 100_000
Y_MAJOR_STEP = 1000
Y_MINOR_STEP = 250
FORMAT_X_IN_M = True

GRID_W = 2.0

RAW_LINE_W = 1
SMOOTH_LINE_W = 2
SMOOTH_EMA = 0.9
CAPSTYLE, JOINSTYLE = "round", "round"

RAW_LIGHT = 0.8
RAW_SAT_MUL = 0.8

# 标记（marker）
MARKER_STYLE_DEFAULT = "^"
MARKER_SIZE = 10
MARKER_EDGE_WIDTH = 1
MARK_EVERY_STEPS = 500_000  # 每 2w 步一个标记

# 自定义每条线的 marker（用于自定义对比图；键为图例文本）
CUSTOM_MARKERS: Dict[str, str] = {
    # "SC-1Actor (run2)": "o",
    # "SC-2Actor (run2)": "s",
    # "TC-1Actor (run2)": "P",
    # "TC-2Actor (run2)": "X",
}
# 轮换用的 marker 集合（不足会循环）
MARKER_CYCLE = ['o', 's', '^', 'D', 'X', 'P', '*', 'v', '>', '<', 'h', 'H', '+', 'x', '1', '2', '3', '4', 'p']

RASTER_DPI = 600

# ================= 修改区域：字体大小配置 =================
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    # --- 修改开始 ---
    "font.size": 18,  # 基础字体保持大号
    "axes.titlesize": 20,  # 标题保持大号
    "axes.labelsize": 18,  # 轴标签保持大号
    "legend.fontsize": 14,  # <--- [修改1] 图例字体调小 (原14 -> 12)
    "xtick.labelsize": 14,  # 刻度保持大号
    "ytick.labelsize": 14,  # 刻度保持大号
    "axes.linewidth": 4,  # 边框加粗
    # --- 修改结束 ---
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
# ========================================================

EVENT_PREFIXES = ("events.out.tfevents", "events.tfevents")

# ===================== 自定义对比（单组 & 多组） =====================
# ① 单组兼容
CUSTOM_COMPARE_LIST: Iterable[Tuple[str, str, str, str]] = [
    ("SCEnv", "1actor", "run2", "SC-1Actor (run2)"),
    ("SCEnv", "2actor", "run2", "SC-2Actor (run2)"),
    ("TCEnv", "1actor", "run2", "TC-1Actor (run2)"),
    ("TCEnv", "2actor", "run2", "TC-2Actor (run2)"),
]

CUSTOM_COMPARE_SETS: List[Dict] = [
    {
        "title": "",
        "slug": "Fig5",
        "items": [
            ("SCEnv", "1actor", "run1", "HO-MAPPO with SC (k=3,M=20)"),
            ("SCEnv", "2actor", "run1", "DH-MAPPO with SC (k=3,M=20)"),
            ("TCEnv", "1actor", "run1", "HO-MAPPO with TC (k=3,M=20)"),
            ("TCEnv", "2actor", "run1", "DH-MAPPO with TC (k=3,M=20)"),
        ],
        "axis": {  # 这张图的坐标独立
            "xlim": (0, 5_100_000),
            "ylim": (-1000, 1100),
            "x_major": 500_000, "x_minor": 100_000,
            "y_major": 250, "y_minor": 250,
            "format_x_in_m": True
        }
    },
    {
        "title": "",
        "slug": "Fig6",
        "items": [
            ("SCEnv", "1actor", "run5", "HO-MAPPO with SC (k=3,M=30)"),
            ("SCEnv", "2actor", "run5", "DH-MAPPO with SC (k=3,M=30)"),
            ("TCEnv", "1actor", "run5", "HO-MAPPO with TC (k=3,M=30)"),
            ("TCEnv", "2actor", "run5", "DH-MAPPO with TC (k=3,M=30)"),
        ],
        "axis": {  # 这张图的坐标独立
            "xlim": (0, 5_100_000),
            "ylim": (-1000, 750),
            "x_major": 500_000, "x_minor": 100_000,
            "y_major": 250, "y_minor": 250,
            "format_x_in_m": True
        }
    },
    {
        "title": "",
        "slug": "Fig7",
        "items": [
            ("SCEnv", "1actor", "run2", "HO-MAPPO with SC (k=2,M=20)"),
            ("SCEnv", "1actor", "run1", "HO-MAPPO with SC (k=3,M=20)"),
            ("SCEnv", "1actor", "run3", "HO-MAPPO with SC (k=4,M=20)"),
            ("SCEnv", "2actor", "run2", "DH-MAPPO with SC (k=2,M=20)"),
            ("SCEnv", "2actor", "run1", "DH-MAPPO with SC (k=3,M=20)"),
            ("SCEnv", "2actor", "run3", "DH-MAPPO with SC (k=4,M=20)"),
        ],
        "axis": {  # 这张图的坐标独立
            "xlim": (0, 5_100_000),
            "ylim": (-1000, 1100),
            "x_major": 500_000, "x_minor": 100_000,
            "y_major": 250, "y_minor": 250,
            "format_x_in_m": True
        }
    },
]

# ====================================================================

# -------------------- 平滑与配色工具 --------------------
def ema_smooth(y: np.ndarray, alpha: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if not (0 < alpha < 1): return y
    out = np.empty_like(y, dtype=float)
    s = np.nan
    for i, v in enumerate(y):
        if np.isnan(v):
            out[i] = s if i > 0 else np.nan
            continue
        s = v if (i == 0 or np.isnan(s)) else alpha * s + (1 - alpha) * v
        out[i] = s
    return out


def lighten_color(hex_color, light=0.55, sat_mul=0.85):
    r, g, b = mcolors.to_rgb(hex_color)
    import colorsys
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = 1 - (1 - l) * (1 - light)
    s = max(0.0, min(1.0, s * sat_mul))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return r2, g2, b2


OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
BASE_PALETTES = ["tab20", "tab20b", "tab20c", "tab10", "Set3", "Set2", "Accent", "Dark2", "Paired"]
MIN_BASE_COLORS = 30


def distinct_colors(n: int):
    colors = list(OKABE_ITO)
    for cmap_name in BASE_PALETTES:
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, "colors") and cmap.colors:
            colors += [mcolors.to_hex(c) for c in cmap.colors]
        else:
            k = max(n, MIN_BASE_COLORS)
            colors += [mcolors.to_hex(cmap(i / max(1, k - 1))) for i in range(k)]
    seen, uniq = set(), []
    for c in colors:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
        if len(uniq) >= max(n, MIN_BASE_COLORS): break
    while len(uniq) < max(n, MIN_BASE_COLORS):
        i, k = len(uniq), max(n, MIN_BASE_COLORS)
        h = (i + 1) / (k + 1)
        rgb = mcolors.hsv_to_rgb([h, 0.60, 0.85])
        hex_ = mcolors.to_hex(rgb)
        if hex_ not in seen:
            uniq.append(hex_)
            seen.add(hex_)
    return uniq[:n]


# -------------------- 基础工具 / 事件读取 --------------------
def is_event_file_name(name: str) -> bool:
    return os.path.basename(name).startswith(EVENT_PREFIXES)


def dir_has_event_files(d: str) -> bool:
    try:
        return any(is_event_file_name(f) for f in os.listdir(d))
    except FileNotFoundError:
        return False


def canonical_event_dir(path_like: str) -> str:
    p = Path(path_like)
    if p.is_file() and is_event_file_name(p.name):
        cand = p.parent
        if dir_has_event_files(str(cand)): return str(cand)
    if p.is_dir() and dir_has_event_files(str(p)): return str(p)
    for anc in [p] + list(p.parents):
        if anc.is_dir() and dir_has_event_files(str(anc)): return str(anc)
    raise FileNotFoundError(f"在 {path_like} 及其父目录未找到 events.* 文件")


def list_scalar_tags(event_dir: str) -> List[str]:
    try:
        acc = ea.EventAccumulator(event_dir, size_guidance={"scalars": 0})
        acc.Reload()
        return sorted(acc.Tags().get("scalars", []))
    except Exception:
        return []


def load_one_dir(event_dir: str, tag: str) -> pd.DataFrame:
    acc = ea.EventAccumulator(event_dir, size_guidance={"scalars": 0})
    acc.Reload()
    tags = set(acc.Tags().get("scalars", []))
    if tag not in tags: return pd.DataFrame()
    rows = [{"step": ev.step, "wall_time": ev.wall_time, "value": ev.value}
            for ev in acc.Scalars(tag)]
    return pd.DataFrame(rows)


# -------------------- 扫描根与元信息 --------------------
def find_results_root(start_dir: Path) -> Optional[Path]:
    for anc in [start_dir] + list(start_dir.parents):
        if anc.name.lower() == "results":
            return anc
    return None


def infer_roots_from_example(example_path: str) -> List[str]:
    ev_dir = Path(canonical_event_dir(example_path))
    res_root = find_results_root(ev_dir)
    if res_root is None:
        env_root = None
        for anc in list(ev_dir.parents):
            if anc.name.endswith("Env"):
                env_root = anc
                break
        return [str(env_root or ev_dir)]
    env_dirs = [d for d in res_root.iterdir() if d.is_dir() and d.name.endswith("Env")]
    return [str(d) for d in env_dirs] or [str(ev_dir)]


def find_event_dirs(roots: List[str]) -> List[str]:
    hits = []
    for root in roots:
        if not os.path.exists(root): continue
        for d, _, _ in os.walk(root):
            if dir_has_event_files(d):
                hits.append(os.path.normpath(d))
    return sorted(set(hits))


def parse_meta_from_path(event_dir: str, roots: List[str]) -> Dict[str, str]:
    rel = None
    for r in roots:
        try:
            cand = os.path.relpath(event_dir, r)
            if not cand.startswith(".."):
                if rel is None or len(cand) < len(rel): rel = cand
        except Exception:
            pass
    rel = rel or event_dir
    parts = [p for p in Path(rel).parts if p]

    meta = dict(env="", actors="", action="", algo="", run="", relpath=rel)
    for seg in parts:
        low = seg.lower()
        if seg.endswith("Env"):
            meta["env"] = seg
        elif "1actor" in low:
            meta["actors"] = "1actor"
        elif "2actor" in low:
            meta["actors"] = "2actor"
        elif seg.startswith("run"):
            meta["run"] = seg
        elif any(k in low for k in ("mappo", "ppo", "trpo", "happo", "hat")):
            meta["algo"] = seg
        elif low in ("discrete", "continuous", "box"):
            meta["action"] = seg
    if not meta["env"]:
        for anc in [Path(event_dir)] + list(Path(event_dir).parents):
            if anc.name.endswith("Env"): meta["env"] = anc.name; break
    if meta["actors"]: meta["actors"] = "1actor" if "1" in meta["actors"] else "2actor"
    meta["group_env_actors"] = "/".join([meta["env"], meta["actors"]]).strip("/")
    return meta


def make_run_display(meta: Dict[str, str], event_dir: str) -> str:
    base = meta.get("run") or "run?"
    h = hashlib.md5(event_dir.encode("utf-8")).hexdigest()[:4]
    extra = []
    if meta.get("algo"):   extra.append(meta["algo"])
    if meta.get("action"): extra.append(meta["action"])
    suffix = ("-" + "-".join(extra)) if extra else ""
    return f"{base}{suffix}#{h}"


# -------------------- 坐标轴/栅栏：支持传入 axis_cfg --------------------
def _resolve_axis_cfg(axis_cfg: Optional[Dict]) -> Dict:
    return {
        "xlim": (X_MIN, X_MAX) if not axis_cfg or "xlim" not in axis_cfg else tuple(axis_cfg["xlim"]),
        "ylim": (Y_MIN, Y_MAX) if not axis_cfg or "ylim" not in axis_cfg else tuple(axis_cfg["ylim"]),
        "x_major": X_MAJOR_STEP if not axis_cfg or "x_major" not in axis_cfg else int(axis_cfg["x_major"]),
        "x_minor": X_MINOR_STEP if not axis_cfg or "x_minor" not in axis_cfg else int(axis_cfg["x_minor"]),
        "y_major": Y_MAJOR_STEP if not axis_cfg or "y_major" not in axis_cfg else int(axis_cfg["y_major"]),
        "y_minor": Y_MINOR_STEP if not axis_cfg or "y_minor" not in axis_cfg else int(axis_cfg["y_minor"]),
        "format_x_in_m": FORMAT_X_IN_M if not axis_cfg or "format_x_in_m" not in axis_cfg else bool(
            axis_cfg["format_x_in_m"]),
    }


def setup_axes(ax, axis_cfg: Optional[Dict] = None) -> Dict:
    cfg = _resolve_axis_cfg(axis_cfg)
    (xmin, xmax), (ymin, ymax) = cfg["xlim"], cfg["ylim"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    def _ticks_inside(min_v, max_v, step):
        if step is None or step <= 0: return np.array([])
        start = np.ceil(min_v / step) * step
        end = np.floor(max_v / step) * step
        if end < start: return np.array([])
        n = int(round((end - start) / step)) + 1
        return start + np.arange(n) * step

    xmaj = _ticks_inside(xmin, xmax, cfg["x_major"])
    ymaj = _ticks_inside(ymin, ymax, cfg["y_major"])
    ax.xaxis.set_major_locator(FixedLocator(xmaj))
    ax.yaxis.set_major_locator(FixedLocator(ymaj))

    xminors = _ticks_inside(xmin, xmax, cfg["x_minor"])
    yminors = _ticks_inside(ymin, ymax, cfg["y_minor"])
    if xminors.size: ax.xaxis.set_minor_locator(FixedLocator(xminors))
    if yminors.size: ax.yaxis.set_minor_locator(FixedLocator(yminors))

    if cfg["format_x_in_m"]:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x / 1e6:.1f}M"))
    else:
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))

    ax.grid(False, which="both", axis="both")
    ax.grid(True, which="major", axis="both", linestyle="-", linewidth=GRID_W, color="white")
    return cfg  # 把使用的坐标配置返回给调用方（用于裁剪与 markevery）


# -------------------- 根据“步长”生成 markevery 索引（接受轴范围） --------------------
def markevery_by_steps(x: np.ndarray,
                       every: int = MARK_EVERY_STEPS,
                       xlim: Tuple[float, float] = (X_MIN, X_MAX)) -> Optional[np.ndarray]:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or every is None or every <= 0: return None
    xmin, xmax = xlim
    start = np.ceil(xmin / every) * every
    ticks = np.arange(start, xmax + 1e-9, every)
    idx = np.searchsorted(x, ticks, side="left")
    idx[idx >= x.size] = x.size - 1
    return np.unique(idx)


# -------------------- 导出工具 --------------------
def slugify(text: str) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^\w\-_.]+", "", text, flags=re.U)
    return text or "figure"


def save_figure(fig, outdir: Path, base: str):
    outdir = Path(outdir)
    for subdir in ["pdf", "eps", "svg", "png", "tiff"]:
        (outdir / subdir).mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "pdf" / f"{base}.pdf", format="pdf")
    fig.savefig(outdir / "eps" / f"{base}.eps", format="eps")
    fig.savefig(outdir / "svg" / f"{base}.svg", format="svg")
    fig.savefig(outdir / "png" / f"{base}.png", format="png", dpi=RASTER_DPI)
    try:
        fig.savefig(outdir / "tiff" / f"{base}.tiff", format="tiff", dpi=RASTER_DPI)
    except Exception:
        pass


# -------------------- 按组出图（支持每组独立坐标） --------------------
def plot_group_raw(df: pd.DataFrame, tag: str, group_name: str, outdir: Path, axis_cfg: Optional[Dict] = None):
    if df.empty: return
    col_name = "run_disp" if "run_disp" in df.columns else "run"
    pivot = df.pivot_table(index="step", columns=col_name, values="value", aggfunc="last").sort_index()
    steps = pivot.index.to_numpy()

    fig_h = FIG_WIDTH_IN * FIG_ASPECT
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#f3f4f6")
    ax.set_axisbelow(True)
    used_axis = setup_axes(ax, axis_cfg)

    run_names = list(pivot.columns)
    base_colors = distinct_colors(len(run_names))
    xlim_used = used_axis["xlim"]

    for i, rn in enumerate(run_names):
        y_raw_all = pivot[rn].to_numpy(dtype=float)
        y_smooth_all = ema_smooth(y_raw_all, SMOOTH_EMA)

        mask = (steps >= xlim_used[0]) & (steps <= xlim_used[1])
        x_plot = steps[mask]
        y_raw = y_raw_all[mask]
        y_smooth = y_smooth_all[mask]

        mark_idx = markevery_by_steps(x_plot, MARK_EVERY_STEPS, xlim_used)
        raw_color = lighten_color(base_colors[i], light=RAW_LIGHT, sat_mul=RAW_SAT_MUL)
        deep_color = mcolors.to_rgb(base_colors[i])

        ax.plot(x_plot, y_raw, linewidth=RAW_LINE_W, color=raw_color, linestyle="-",
                solid_capstyle=CAPSTYLE, solid_joinstyle=JOINSTYLE,
                antialiased=True, rasterized=False, zorder=1)

        ax.plot(x_plot, y_smooth, linewidth=SMOOTH_LINE_W, color=deep_color, linestyle="-",
                marker=CUSTOM_MARKERS.get(rn, MARKER_CYCLE[i % len(MARKER_CYCLE)]), markersize=MARKER_SIZE,
                markerfacecolor="none", markeredgecolor=deep_color,
                markeredgewidth=MARKER_EDGE_WIDTH, markevery=mark_idx,
                label=rn, solid_capstyle=CAPSTYLE, solid_joinstyle=JOINSTYLE,
                antialiased=True, rasterized=False, zorder=2)

    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_linewidth(1.0)

    ax.set_title(f"{tag} — {group_name}", pad=8)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Average episode reward")

    n_runs = len(run_names)
    leg = (ax.legend(loc="lower right", bbox_to_anchor=(0.5, 1.02), frameon=True, fancybox=False,
                     ncol=min(4, (n_runs + 1) // 2)) if n_runs > 14
           else ax.legend(loc="upper center", frameon=True, fancybox=False, ncol=2))
    frame = leg.get_frame()
    frame.set_facecolor("#f7f9fc")
    frame.set_edgecolor("#c7ced8")
    frame.set_linewidth(0.6)

    fig.tight_layout(pad=1.0)
    base = f"{tag}_{group_name.replace('/', '_')}"
    save_figure(fig, Path(outdir), base)
    plt.close(fig)


# -------------------- 自定义对比图（支持每图独立坐标） --------------------
def plot_compare_custom(data: pd.DataFrame,
                        compare_list: Iterable[Tuple[str, str, str, str]],
                        tag: str, outdir: Path,
                        title: Optional[str] = None,
                        file_slug: Optional[str] = None,
                        axis_cfg: Optional[Dict] = None):
    series, labels = [], []
    for env, actors, run_id, lab in compare_list:
        sub = data[(data["env"] == env) & (data["actors"] == actors) & (data["run"].str.lower() == run_id.lower())]
        if sub.empty:
            print(f"[INFO] 缺少 {env}/{actors}/{run_id}，跳过。")
            continue
        s = sub.sort_values("step").drop_duplicates("step", keep="last")
        series.append((s["step"].to_numpy(float), s["value"].to_numpy(float)))
        labels.append(lab)
    if not series:
        print("[WARN] 自定义对比没有任何可绘制序列。")
        return

    fig_h = FIG_WIDTH_IN * FIG_ASPECT
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#f3f4f6")
    ax.set_axisbelow(True)
    used_axis = setup_axes(ax, axis_cfg)
    xlim_used = used_axis["xlim"]

    colors = distinct_colors(len(series))
    for i, ((x, y), lab) in enumerate(zip(series, labels)):
        mask = (x >= xlim_used[0]) & (x <= xlim_used[1])
        x = x[mask]
        y = y[mask]
        mark_idx = markevery_by_steps(x, MARK_EVERY_STEPS, xlim_used)
        raw_c = lighten_color(colors[i], light=RAW_LIGHT, sat_mul=RAW_SAT_MUL)
        deep_c = mcolors.to_rgb(colors[i])

        ax.plot(x, y, linewidth=RAW_LINE_W, color=raw_c, linestyle="-",
                solid_capstyle=CAPSTYLE, solid_joinstyle=JOINSTYLE,
                antialiased=True, rasterized=False, zorder=1)

        mk = CUSTOM_MARKERS.get(lab, MARKER_CYCLE[i % len(MARKER_CYCLE)])
        ax.plot(x, ema_smooth(y, SMOOTH_EMA),
                linewidth=SMOOTH_LINE_W, color=deep_c, linestyle="-",
                marker=mk, markersize=MARKER_SIZE,
                markerfacecolor="none", markeredgecolor=deep_c,
                markeredgewidth=MARKER_EDGE_WIDTH, markevery=mark_idx,
                label=lab, solid_capstyle=CAPSTYLE, solid_joinstyle=JOINSTYLE,
                antialiased=True, rasterized=False, zorder=2)

    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_linewidth(1.0)

    title_text = title if title not in (None, "", False) else None
    if title_text:
        ax.set_title(title_text)

    # --- 保持轴标签大字体 ---
    ax.set_xlabel("Steps")
    ax.set_ylabel("Average episode reward")

    # --- 修改：缩小图例字体和边距，使其紧凑且靠边，避免遮挡 ---
    leg = ax.legend(
        loc="lower right",
        frameon=True, fancybox=False, ncol=1,
        borderaxespad=0.5,  # <--- [修改2] 减小与轴边界距离 (原4 -> 0.5)
        borderpad=0.5,  # <--- [修改3] 减小图例框内边距 (原1 -> 0.5)
    )
    frame = leg.get_frame()
    frame.set_facecolor("#f7f9fc")
    frame.set_edgecolor("#c7ced8")
    frame.set_linewidth(0.6)

    fig.tight_layout(pad=1.0)
    base = f"{(file_slug or slugify(title or 'custom_compare'))}_{tag}"
    save_figure(fig, Path(outdir), base)
    plt.close(fig)


# -------------------- 主流程 --------------------
def main():
    try:
        ev_dir = canonical_event_dir(EXAMPLE_PATH)
    except FileNotFoundError as e:
        print(f"[ERR] {e}")
        return

    Path(OUTDIR).mkdir(parents=True, exist_ok=True)

    tags = list_scalar_tags(ev_dir)
    print(f"[INFO] 示例事件目录：{ev_dir}")
    print(f"[INFO] 可用 scalar 标签：{tags}")
    if TAG not in tags:
        print(f"[WARN] 示例目录中不包含 TAG='{TAG}'，请将 TAG 改为上面列表中的实际名称。")

    roots = infer_roots_from_example(EXAMPLE_PATH)
    print("[INFO] 扫描根：")
    [print("  -", r) for r in roots]

    event_dirs = find_event_dirs(roots)
    if not event_dirs:
        print("[WARN] 未找到任何包含 events.* 的目录。")
        pd.DataFrame(columns=["env", "actors", "group_env_actors", "run", "run_disp", "step", "value"]).to_csv(
            Path(OUTDIR) / f"{TAG}_raw_all.csv", index=False, encoding="utf-8")
        return

    rows = []
    for d in event_dirs:
        try:
            df = load_one_dir(d, TAG)
            if df.empty: continue
            meta = parse_meta_from_path(d, roots)
            run_disp = make_run_display(meta, d)
            df["env"] = meta["env"]
            df["actors"] = meta["actors"]
            df["group_env_actors"] = meta["group_env_actors"] or "unknown"
            df["run"] = meta["run"] or os.path.basename(d)
            df["run_disp"] = run_disp
            rows.append(df[["env", "actors", "group_env_actors", "run", "run_disp", "step", "value"]])
            print(
                f"[OK] {d} → rows={len(df)} | group={df['group_env_actors'].iloc[0]} | run={df['run'].iloc[0]} | disp={run_disp}")
        except Exception as e:
            print(f"[ERR] {d}: {e}")

    if not rows:
        print(f"[WARN] 没有读到标签 '{TAG}' 的任何数据。")
        pd.DataFrame(columns=["env", "actors", "group_env_actors", "run", "run_disp", "step", "value"]).to_csv(
            Path(OUTDIR) / f"{TAG}_raw_all.csv", index=False, encoding="utf-8")
        return

    data = pd.concat(rows, ignore_index=True).sort_values(["group_env_actors", "run_disp", "step"])

    # 多组自定义对比：每张图可以带自己的 axis
    if CUSTOM_COMPARE_SETS:
        for cfg in CUSTOM_COMPARE_SETS:
            plot_compare_custom(
                data,
                cfg.get("items", []),
                TAG,
                Path(OUTDIR),
                title=cfg.get("title", "Custom Compare"),
                file_slug=cfg.get("slug"),
                axis_cfg=cfg.get("axis")
            )
    else:
        plot_compare_custom(
            data,
            CUSTOM_COMPARE_LIST,
            TAG,
            Path(OUTDIR),
            title="Average Episode Reward — Custom Compare",
            file_slug="custom_compare",
            axis_cfg=None
        )

    csv_path = Path(OUTDIR) / f"{TAG}_raw_all.csv"
    data.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[DONE] 输出到：{OUTDIR}/pdf, eps, svg, png, tiff；汇总 CSV → {csv_path}")


if __name__ == "__main__":
    main()
