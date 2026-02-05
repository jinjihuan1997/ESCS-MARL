#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot slot-based multi-series comparison on a single figure
for:
  1) avg_sp_vs_slot (multi runs, one figure)                 -> no smoothing
  2) avg_rx_vs_slot (non-cumulative; multi runs, one figure) -> EMA smoothing ONLY here
  3) avg_rx_vs_slot as cumulative RX (multi runs)            -> no smoothing
  4) avg_sp_vs_slot as cumulative SP (multi runs)            -> no smoothing
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.ticker import FixedLocator, FuncFormatter

# ========== Paths ==========
EXAMPLE_PATH = r"C:\Users\ENeS\Desktop\MARL\Code\light_mappo_new\results\SCEnv\1actor\Discrete\r_mappo\check\run5\logs"
OUTDIR = "plots_slot_series_compare"

# CSV names
CSV_SP = "avg_sp_timeseries.csv"
CSV_RX = "avg_rx_timeseries.csv"

# ========== Custom items ==========
CUSTOM_ITEMS_SP: List[Tuple[str, str, str, str]] = [
    ("SCEnv", "1actor", "run1", "HO-MAPPO with SC (k=3,M=20)"),
    ("TCEnv", "1actor", "run1", "HO-MAPPO with TC (k=3,M=20)"),
    ("SCEnv", "2actor", "run1", "DH-MAPPO with SC (k=3,M=20)"),
    ("TCEnv", "2actor", "run1", "DH-MAPPO with TC (k=3,M=20)"),
]

# RX (non-cumulative) — will be smoothed
CUSTOM_ITEMS_RX_AVG: List[Tuple[str, str, str, str]] = [
    ("SCEnv", "1actor", "run1", "HO-MAPPO with SC (k=3,M=20)"),
    ("TCEnv", "1actor", "run1", "HO-MAPPO with TC (k=3,M=20)"),
    ("SCEnv", "2actor", "run1", "DH-MAPPO with SC (k=3,M=20)"),
    ("TCEnv", "2actor", "run1", "DH-MAPPO with TC (k=3,M=20)"),
]

# RX cumulative
CUSTOM_ITEMS_RX_CUM: List[Tuple[str, str, str, str]] = [
    ("SCEnv", "1actor", "run1", "HO-MAPPO with SC (k=3,M=20)"),
    ("TCEnv", "1actor", "run1", "HO-MAPPO with TC (k=3,M=20)"),
    ("SCEnv", "2actor", "run1", "DH-MAPPO with SC (k=3,M=20)"),
    ("TCEnv", "2actor", "run1", "DH-MAPPO with TC (k=3,M=20)"),
]

# SP cumulative  ← 新增（默认与 SP 使用同一组实验）
CUSTOM_ITEMS_SP_CUM: List[Tuple[str, str, str, str]] = [
    ("SCEnv", "1actor", "run1", "HO-MAPPO with SC (k=3,M=20)"),
    ("TCEnv", "1actor", "run1", "HO-MAPPO with TC (k=3,M=20)"),
    ("SCEnv", "2actor", "run1", "DH-MAPPO with SC (k=3,M=20)"),
    ("TCEnv", "2actor", "run1", "DH-MAPPO with TC (k=3,M=20)"),
]

# ========== Style / defaults ==========
FIG_WIDTH_IN = 8.0
FIG_ASPECT = 0.62
GRID_W = 2.0
RASTER_DPI = 600

LINE_WIDTH = 1.3
MARKER_SIZE = 3.7
MARKER_EDGE_WIDTH = 0.7
MARK_EVERY_SLOTS = 10

# smoothing (ONLY for RX-AVG)
SMOOTH_EMA = 0.9
RAW_LINE_W = 0.8
SMOOTH_LINE_W = 1.2
RAW_LIGHT = 0.75
RAW_SAT_MUL = 0.80
CAPSTYLE = "round"
JOINSTYLE = "round"

# X axis
X_HARD_MAX = 400
X_PAD_EXTRA = 5
X_TICK_STEP = 50
X_MINOR_STEP = 25

# colors & markers
MARKER_CYCLE = ['o', 's', '^', 'D', 'X', 'P', '*', 'v', '>', '<', 'h', 'H', '+', 'x', '1', '2', '3', '4', 'p']
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
BASE_PALETTES = ["tab20", "tab20b", "tab20c", "tab10", "Set3", "Set2", "Accent", "Dark2", "Paired"]
MIN_BASE_COLORS = 30

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    # --- 修改以下数值 ---
    "font.size": 16,             # 原 12 -> 16 (全局基础大小)
    "axes.titlesize": 18,        # 原 12 -> 18 (标题大小)
    "axes.labelsize": 18,        # 原 13 -> 18 (轴标签大小)
    "legend.fontsize": 12,       # 原 11 -> 14 (图例大小)
    "xtick.labelsize": 14,       # 原 9  -> 14 (X轴刻度数值大小)
    "ytick.labelsize": 14,       # 原 9  -> 14 (Y轴刻度数值大小)
    # ------------------
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.linewidth": 1.5,       # 建议同时把坐标轴线宽从 1.0 调大到 1.5 以匹配大字体
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


# ========== Utils ==========
def distinct_colors(n: int):
    colors = list(OKABE_ITO)
    for cmap_name in BASE_PALETTES:
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, "colors") and cmap.colors:
            colors += [matplotlib.colors.to_hex(c) for c in cmap.colors]
        else:
            k = max(n, MIN_BASE_COLORS)
            colors += [matplotlib.colors.to_hex(cmap(i / max(1, k - 1))) for i in range(k)]
    seen, uniq = set(), []
    for c in colors:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
        if len(uniq) >= max(n, MIN_BASE_COLORS):
            break
    while len(uniq) < max(n, MIN_BASE_COLORS):
        i, k = len(uniq), max(n, MIN_BASE_COLORS)
        h = (i + 1) / (k + 1)
        rgb = matplotlib.colors.hsv_to_rgb([h, 0.60, 0.85])
        hex_ = matplotlib.colors.to_hex(rgb)
        if hex_ not in seen:
            uniq.append(hex_)
            seen.add(hex_)
    return uniq[:n]


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


def lighten_color(hex_color, light=RAW_LIGHT, sat_mul=RAW_SAT_MUL):
    r, g, b = mcolors.to_rgb(hex_color)
    import colorsys
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = 1 - (1 - l) * (1 - light)
    s = max(0.0, min(1.0, s * sat_mul))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return r2, g2, b2


def is_event_file_name(name: str) -> bool:
    return os.path.basename(name).startswith(("events.out.tfevents", "events.tfevents"))


def dir_has_event_files(d: str) -> bool:
    try:
        return any(is_event_file_name(f) for f in os.listdir(d))
    except FileNotFoundError:
        return False


def canonical_event_dir(path_like: str) -> str:
    p = Path(path_like)
    if p.is_file() and is_event_file_name(p.name):
        if dir_has_event_files(str(p.parent)):
            return str(p.parent)
    if p.is_dir() and dir_has_event_files(str(p)):
        return str(p)
    for anc in [p] + list(p.parents):
        if anc.is_dir() and dir_has_event_files(str(anc)):
            return str(anc)
    return str(p if p.is_dir() else p.parent)


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


def find_candidate_out_dirs(roots: List[str], env: str, actors: str, run_id: str) -> List[Path]:
    hits: List[Path] = []
    for root in roots:
        base = Path(root)
        if not base.exists():
            continue
        for d, subdirs, files in os.walk(base):
            p = Path(d)
            parts = [seg for seg in p.parts if seg]
            if (env in parts) and (actors in parts) and any(seg.lower() == run_id.lower() for seg in parts):
                if p.name == "out":
                    hits.append(p.resolve())
    return sorted(set(hits))


def normalize_series_csv(csv_path: Path, value_col_fallback: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "slot" not in df.columns:
        sc = "slot" if "slot" in df.columns else ("step" if "step" in df.columns else None)
        if sc is None:
            raise ValueError(f"{csv_path} 缺少 'slot' / 'step' 列。")
        df.rename(columns={sc: "slot"}, inplace=True)
    if "value" not in df.columns:
        candidates = [c for c in df.columns if c != "slot"]
        if len(candidates) == 1:
            df.rename(columns={candidates[0]: "value"}, inplace=True)
        else:
            if value_col_fallback in df.columns:
                df.rename(columns={value_col_fallback: "value"}, inplace=True)
            else:
                df.rename(columns={candidates[0]: "value"}, inplace=True)
    df = df[["slot", "value"]].dropna().sort_values("slot").drop_duplicates("slot", keep="last")
    df["slot"] = df["slot"].round().astype(int)
    return df


def setup_axes_slots(ax,
                     xlim: Optional[Tuple[float, float]] = None,
                     ylim: Optional[Tuple[float, float]] = None) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xlim = (0, X_HARD_MAX + X_PAD_EXTRA)
    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    xmaj = np.arange(0, X_HARD_MAX + 1, X_TICK_STEP, dtype=int)
    ax.xaxis.set_major_locator(FixedLocator(xmaj))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v)}"))

    if X_MINOR_STEP and X_MINOR_STEP > 0:
        xminors = np.arange(0, X_HARD_MAX + 1, X_MINOR_STEP, dtype=int)
        xminors = np.array([t for t in xminors if t not in set(xmaj)], dtype=int)
        if xminors.size:
            ax.xaxis.set_minor_locator(FixedLocator(xminors))

    ax.set_facecolor("#f3f4f6")
    ax.set_axisbelow(True)
    ax.grid(False, which="both", axis="both")
    ax.grid(True, which="major", axis="both", linestyle="-", linewidth=GRID_W, color="white")
    return ax.get_xlim(), ax.get_ylim()


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


def markevery_by_stride(x: np.ndarray, stride_slots: int, xlim: Tuple[float, float]) -> Optional[np.ndarray]:
    if x.size == 0 or not stride_slots or stride_slots <= 0:
        return None
    xmin, xmax = xlim
    start = int(np.ceil(xmin / stride_slots) * stride_slots)
    ticks = np.arange(start, xmax + 1e-9, stride_slots)
    idx = np.searchsorted(x, ticks, side="left")
    idx[idx >= x.size] = x.size - 1
    return np.unique(idx)


# ========== Plotting ==========
def plot_multi_series(series_list: List[Tuple[np.ndarray, np.ndarray, str]],
                      ylabel: str, title: str, base_name: str, outdir: Path,
                      smooth: bool = False):
    """
    If smooth=True (used ONLY for RX-AVG):
      - plot raw line in light color
      - overlay EMA-smoothed line in deep color (with markers)
    Else:
      - plot single deep line with markers (original behavior).
    """
    if not series_list:
        print(f"[WARN] 没有可绘制的数据：{base_name}")
        return

    fig_h = FIG_WIDTH_IN * FIG_ASPECT
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)

    # auto y
    all_y = np.concatenate([y for _, y, _ in series_list]) if series_list else np.array([])
    ylim = None
    if np.isfinite(all_y).any():
        yv = all_y[np.isfinite(all_y)]
        ypad = 0.05 * (float(yv.max()) - float(yv.min()) + 1.0)
        ylim = (float(yv.min()) - ypad, float(yv.max()) + ypad)

    xlim_used, _ = setup_axes_slots(ax, xlim=None, ylim=ylim)

    colors = distinct_colors(len(series_list))
    for i, (x, y, lab) in enumerate(series_list):
        order = np.argsort(x)
        x = np.asarray(x, dtype=float)[order]
        y = np.asarray(y, dtype=float)[order]

        deep_c = mcolors.to_rgb(colors[i])
        mk = MARKER_CYCLE[i % len(MARKER_CYCLE)]
        idx = markevery_by_stride(x, MARK_EVERY_SLOTS, xlim_used)

        if smooth:
            # raw (light)
            raw_c = lighten_color(colors[i], light=RAW_LIGHT, sat_mul=RAW_SAT_MUL)
            ax.plot(x, y, linewidth=RAW_LINE_W, color=raw_c, linestyle="-",
                    solid_capstyle="round", solid_joinstyle="round",
                    antialiased=True, rasterized=False, zorder=1)

            # smooth (deep) + markers
            y_s = ema_smooth(y, SMOOTH_EMA)
            ax.plot(x, y_s, linewidth=SMOOTH_LINE_W, color=deep_c, linestyle="-",
                    marker=mk, markersize=MARKER_SIZE,
                    markerfacecolor="white", markeredgecolor=deep_c,
                    markeredgewidth=MARKER_EDGE_WIDTH, markevery=idx,
                    label=lab, solid_capstyle="round", solid_joinstyle="round",
                    antialiased=True, rasterized=False, zorder=2)
        else:
            # original style: deep line + markers, no smoothing
            ax.plot(x, y, linewidth=LINE_WIDTH, color=deep_c, linestyle="-",
                    marker=mk, markersize=MARKER_SIZE,
                    markerfacecolor="white", markeredgecolor=deep_c,
                    markeredgewidth=MARKER_EDGE_WIDTH, markevery=idx,
                    label=lab, solid_capstyle="round", solid_joinstyle="round",
                    antialiased=True, rasterized=False, zorder=2)

    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_linewidth(1.0)

    ax.set_xlabel("Time slot (T)", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    if title:
        ax.set_title(title, pad=6)

    ncol = 1 if len(series_list) <= 8 else 2
    leg = ax.legend(loc="best", frameon=True, fancybox=False, ncol=ncol)
    frame = leg.get_frame()
    frame.set_facecolor("#f7f9fc")
    frame.set_edgecolor("#c7ced8")
    frame.set_linewidth(0.6)

    fig.tight_layout(pad=1.0)
    save_figure(fig, Path(outdir), base_name)
    plt.close(fig)


# ========== Main ==========
def main():
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    roots = infer_roots_from_example(EXAMPLE_PATH)
    print("[INFO] Roots:")
    for r in roots:
        print("  -", r)

    sp_series: List[Tuple[np.ndarray, np.ndarray, str]] = []
    rx_avg_series: List[Tuple[np.ndarray, np.ndarray, str]] = []
    rx_cum_series: List[Tuple[np.ndarray, np.ndarray, str]] = []
    sp_cum_series: List[Tuple[np.ndarray, np.ndarray, str]] = []  # ← 新增

    # SP (non-cumulative)
    for env, actors, run_id, label in CUSTOM_ITEMS_SP:
        outs = find_candidate_out_dirs(roots, env, actors, run_id)
        if not outs:
            print(f"[WARN] 找不到 {env}/{actors}/{run_id} 的 out 目录，跳过：{label}")
            continue
        sp_csv = Path(outs[0]) / CSV_SP
        if not sp_csv.exists():
            print(f"[WARN] {outs[0]} 缺少 {CSV_SP}，跳过：{label}")
            continue
        try:
            df = normalize_series_csv(sp_csv, "avg_sp_timeseries")
            if df.empty:
                print(f"[INFO] {label} SP CSV 为空，跳过。")
                continue
            sp_series.append((df["slot"].to_numpy(float), df["value"].to_numpy(float), label))
        except Exception as e:
            print(f"[ERR] 读取 SP {label} 失败：{e}")

    # RX-AVG (smoothed)
    rx_avg_items = CUSTOM_ITEMS_RX_AVG if len(CUSTOM_ITEMS_RX_AVG) > 0 else CUSTOM_ITEMS_RX_CUM
    for env, actors, run_id, label in rx_avg_items:
        outs = find_candidate_out_dirs(roots, env, actors, run_id)
        if not outs:
            print(f"[WARN] 找不到 {env}/{actors}/{run_id} 的 out 目录，跳过：{label}")
            continue
        rx_csv = Path(outs[0]) / CSV_RX
        if not rx_csv.exists():
            print(f"[WARN] {outs[0]} 缺少 {CSV_RX}，跳过：{label}")
            continue
        try:
            df = normalize_series_csv(rx_csv, "avg_rx_timeseries")
            if df.empty:
                print(f"[INFO] {label} RX CSV 为空，跳过。")
                continue
            rx_avg_series.append((df["slot"].to_numpy(float), df["value"].to_numpy(float), label))
        except Exception as e:
            print(f"[ERR] 读取 RX(非累计) {label} 失败：{e}")

    # RX-CUM (cumulative sum over slots)
    for env, actors, run_id, label in CUSTOM_ITEMS_RX_CUM:
        outs = find_candidate_out_dirs(roots, env, actors, run_id)
        if not outs:
            print(f"[WARN] 找不到 {env}/{actors}/{run_id} 的 out 目录，跳过：{label}")
            continue
        rx_csv = Path(outs[0]) / CSV_RX
        if not rx_csv.exists():
            print(f"[WARN] {outs[0]} 缺少 {CSV_RX}，跳过：{label}")
            continue
        try:
            df = normalize_series_csv(rx_csv, "avg_rx_timeseries")
            if df.empty:
                print(f"[INFO] {label} RX CSV 为空，跳过。")
                continue
            y_cum = np.cumsum(df["value"].to_numpy(float))
            rx_cum_series.append((df["slot"].to_numpy(float), y_cum, label))
        except Exception as e:
            print(f"[ERR] 读取 RX(累计) {label} 失败：{e}")

    # SP-CUM (cumulative sum over slots) ← 新增
    for env, actors, run_id, label in CUSTOM_ITEMS_SP_CUM:
        outs = find_candidate_out_dirs(roots, env, actors, run_id)
        if not outs:
            print(f"[WARN] 找不到 {env}/{actors}/{run_id} 的 out 目录，跳过：{label}")
            continue
        sp_csv = Path(outs[0]) / CSV_SP
        if not sp_csv.exists():
            print(f"[WARN] {outs[0]} 缺少 {CSV_SP}，跳过：{label}")
            continue
        try:
            df = normalize_series_csv(sp_csv, "avg_sp_timeseries")
            if df.empty:
                print(f"[INFO] {label} SP CSV 为空，跳过。")
                continue
            y_cum = np.cumsum(df["value"].to_numpy(float))
            sp_cum_series.append((df["slot"].to_numpy(float), y_cum, label))
        except Exception as e:
            print(f"[ERR] 读取 SP(累计) {label} 失败：{e}")

    # === Plots ===
    # 1) SP (no smoothing)
    plot_multi_series(
        sp_series,
        ylabel="Average SP",
        title="",
        base_name="avg_sp_vs_slot_compare",
        outdir=Path(OUTDIR),
        smooth=False,
    )

    # 2) RX-AVG (SMOOTHED)
    plot_multi_series(
        rx_avg_series,
        ylabel="Average received images",
        title="",
        base_name="avg_rx_vs_slot_compare",
        outdir=Path(OUTDIR),
        smooth=True,  # <---- only here
    )

    # 3) RX-CUM (no smoothing)
    plot_multi_series(
        rx_cum_series,
        ylabel="Average cumulative received images",
        title="",
        base_name="avg_rx_cumulative_vs_slot_compare",
        outdir=Path(OUTDIR),
        smooth=False,
    )

    # 4) SP-CUM (no smoothing) ← 新增
    plot_multi_series(
        sp_cum_series,
        ylabel="Average cumulative SP",
        title="",
        base_name="avg_sp_cumulative_vs_slot_compare",
        outdir=Path(OUTDIR),
        smooth=False,
    )

    print(f"[DONE] 图已输出到 {OUTDIR}/pdf, eps, svg, png, tiff")


if __name__ == "__main__":
    main()
