#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-bin (SCEnv, TCEnv) bar charts (four separate figures):
  - X axis has exactly two categories: SCEnv and TCEnv.
  - Within each category, we draw bars for every (actors, run) combination present in CUSTOM_ITEMS_BAR.
  - Color encodes the (actors, run) variant (e.g., HO-MAPPO-run1, DH-MAPPO-run2...), and the legend explains colors.

Figures:
  1) Average SP over full period
  2) Average received images (RX) over full period
  3) Average cumulative received images over full period
  4) Average cumulative SP over full period
"""

import os
import re
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ========== 起点路径（用于推断 results 根） ==========
EXAMPLE_PATH = r"C:\Users\ENeS\Desktop\MARL\Code\light_mappo_new\results\SCEnv\1actor\Discrete\r_mappo\check\run5\logs"
OUTDIR = "plots_slot_bars_twoenv"

# ========== 需要汇总为柱状图的自定义项 ==========
# (env_name, actors, run_id, display_label)
CUSTOM_ITEMS_BAR: List[tuple[str, str, str, str]] = [
    ("SCEnv", "2actor", "run2", "SC-2Actor (run2)"),
    ("TCEnv", "2actor", "run2", "TC-2Actor (run2)"),
    ("SCEnv", "2actor", "run1", "SC-2Actor (run1)"),
    ("TCEnv", "2actor", "run1", "TC-2Actor (run1)"),
    ("SCEnv", "2actor", "run3", "SC-2Actor (run3)"),
    ("TCEnv", "2actor", "run3", "TC-2Actor (run3)"),
    ("SCEnv", "1actor", "run2", "SC-HO-MAPPO (run2)"),
    ("TCEnv", "1actor", "run2", "TC-HO-MAPPO (run2)"),
    ("SCEnv", "1actor", "run1", "SC-HO-MAPPO (run1)"),
    ("TCEnv", "1actor", "run1", "TC-HO-MAPPO (run1)"),
    ("SCEnv", "1actor", "run3", "SC-HO-MAPPO (run3)"),
    ("TCEnv", "1actor", "run3", "TC-HO-MAPPO (run3)"),
]

# 源 CSV 文件名
CSV_SP = "avg_sp_timeseries.csv"  # 需含：slot, avg_sp_timeseries（或自动探测）
CSV_RX = "avg_rx_timeseries.csv"  # 需含：slot, avg_rx_timeseries（或自动探测）

# ---------- 画图风格（对齐“论文风格”） ----------
FIG_WIDTH_IN = 9.5
FIG_ASPECT = 0.60
GRID_W = 2.0
RASTER_DPI = 600

# 两个大类：SCEnv、TCEnv
ENV_CATS = ["SCEnv", "TCEnv"]
ENV_LABELS = ["SC", "TC"]

# 两个大类之间的距离（类中心）
ENV_GAP = 1.6
BAR_WIDTH = 0.16
VALUE_LABELS = True

# ================= 修改区域：字体大小配置 =================
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    # --- 修改开始 ---
    "font.size": 16,  # 原 12 -> 16
    "axes.titlesize": 18,  # 原 12 -> 18
    "axes.labelsize": 18,  # 原 13 -> 18
    "legend.fontsize": 12,  # 原 10 -> 14
    "xtick.labelsize": 14,  # 原 10 -> 14
    "ytick.labelsize": 14,  # 原 10 -> 14
    "axes.linewidth": 1.5,  # 原 1.0 -> 1.5 (加粗边框)
    # --- 修改结束 ---
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
# ========================================================

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
BASE_PALETTES = ["tab20", "tab20b", "tab20c", "tab10", "Set3", "Set2", "Accent", "Dark2", "Paired"]
MIN_BASE_COLORS = 30

# -------------------- 显示名称映射（仅影响图例） --------------------
RUN_LABEL = {
    "run1": "(K=3,M=20)",
    "run2": "(K=2,M=20)",
    "run3": "(K=4,M=20)",
}
_SERIES_RE = re.compile(r"^(?P<base>.+?)-(?P<run>run[0-9]+)$")


def pretty_series_label(series: str) -> str:
    """'HO-MAPPO-run1' -> 'HO-MAPPO (K=3,M=20)'。匹配失败则原样返回。"""
    m = _SERIES_RE.match(series)
    if not m:
        return series
    base = m.group("base")
    run = m.group("run")
    return f"{base} {RUN_LABEL.get(run, run)}"


# -------------------- 工具：颜色 --------------------
def distinct_colors(n: int):
    colors = list(OKABE_ITO)
    for cmap_name in BASE_PALETTES:
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, "colors") and getattr(cmap, "colors"):
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


# -------------------- 系列名生成（全局统一） --------------------
def SERIES_KEY(actors: str, run: str) -> str:
    """把 (actors, run) 映射为 'HO-MAPPO-run1' / 'DH-MAPPO-run2'。"""
    short_a = "HO-MAPPO" if actors == "1actor" else "DH-MAPPO"
    return f"{short_a}-{run}"


# -------------------- 文件/路径工具 --------------------
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


# -------------------- 读取 & 规格化 --------------------
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


# -------------------- 指标计算（新增 avg_cum_sp） --------------------
def compute_metrics(sp_df: pd.DataFrame, rx_df: pd.DataFrame) -> tuple[float, float, float, float]:
    # 平均 SP
    avg_sp = float(sp_df["value"].mean()) if not sp_df.empty else np.nan
    # 平均 RX
    avg_rx = float(rx_df["value"].mean()) if not rx_df.empty else np.nan

    # 累计平均 RX（对累计序列再取均值；如需最终累计总量，用 cum[-1]）
    if not rx_df.empty:
        cum_rx = np.cumsum(rx_df["value"].to_numpy(dtype=float))
        avg_cum_rx = float(np.nanmean(cum_rx))
    else:
        avg_cum_rx = np.nan

    # 累计平均 SP（新增）
    if not sp_df.empty:
        cum_sp = np.cumsum(sp_df["value"].to_numpy(dtype=float))
        avg_cum_sp = float(np.nanmean(cum_sp))
    else:
        avg_cum_sp = np.nan

    return avg_sp, avg_rx, avg_cum_rx, avg_cum_sp


# -------------------- 绘图 --------------------
def plot_two_env_multiseries(
        df: pd.DataFrame,
        metric_col: str,
        ylabel: str,
        title: str,
        base_name: str,
        *,
        series_levels_all: Optional[List[str]] = None,
        series_palette: Optional[dict] = None,
        legend_spec: Optional[dict] = None,
        value_labels: Optional[bool] = None
):
    """
    df 需含列：['env','actors','run','value']。
    legend_spec 支持键：show/include/order/rename/title/ncol/place(loc inside)/loc
    """
    if df.empty:
        print(f"[WARN] 没有可绘制的数据：{base_name}")
        return

    df = df.copy()
    df["series"] = df.apply(lambda r: SERIES_KEY(r["actors"], r["run"]), axis=1)

    if series_levels_all is None:
        series_levels = list(dict.fromkeys(df["series"].tolist()))
    else:
        exists = set(df["series"].tolist())
        series_levels = [s for s in series_levels_all if s in exists]

    if series_palette is None:
        colors = distinct_colors(len(series_levels))
        color_map = {s: c for s, c in zip(series_levels, colors)}
    else:
        color_map = dict(series_palette)

    env_centers = np.array([0.0, ENV_GAP])
    n_series = len(series_levels)
    spacing = BAR_WIDTH * 1.25
    offsets = ((np.arange(n_series) - (n_series // 2)) * spacing
               if n_series % 2 == 1
               else (np.arange(n_series) - (n_series / 2 - 0.5)) * spacing)

    fig_h = FIG_WIDTH_IN * FIG_ASPECT
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h))
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f3f4f6")
    ax.set_axisbelow(True)
    ax.grid(False, which="both", axis="both")
    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=GRID_W, color="white")

    bars_for_labels = []
    for ei, env in enumerate(ENV_CATS):
        cx = env_centers[ei]
        sub_env = df[df["env"] == env]
        for si, sname in enumerate(series_levels):
            sub = sub_env[sub_env["series"] == sname]
            if sub.empty:
                continue
            val = float(sub.iloc[0][metric_col])
            x = cx + offsets[si]
            rect = ax.bar(
                x, val, width=BAR_WIDTH,
                color=color_map.get(sname, "#999999"),
                edgecolor=color_map.get(sname, "#999999"),
                linewidth=0.6, zorder=3
            )
            bars_for_labels.append(rect[0])

    if value_labels if value_labels is not None else VALUE_LABELS:
        for rect in bars_for_labels:
            h = rect.get_height()
            if np.isfinite(h):
                # --- 修改：数值标签字体调大到 12 (原9) ---
                ax.annotate(f"{h:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2.0, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=12)

    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_linewidth(1.0)

    ax.set_xticks(env_centers)
    # --- 修改：显式指定大字体 ---
    ax.set_xticklabels(ENV_LABELS, rotation=0, ha="center", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)

    if title:
        ax.set_title(title, pad=6)

    # ===== 图例（每图可独立配置） =====
    L = dict(legend_spec or {})
    if L.get("show", True):
        if "include" in L and L["include"]:
            include_set = set(L["include"])
            legend_series = [s for s in series_levels if s in include_set]
        else:
            legend_series = list(series_levels)

        if "order" in L and L["order"]:
            order_map = {s: i for i, s in enumerate(L["order"])}
            legend_series.sort(key=lambda s: order_map.get(s, 10_000))
        elif series_levels_all:
            order_map = {s: i for i, s in enumerate(series_levels_all)}
            legend_series.sort(key=lambda s: order_map.get(s, 10_000))

        rename = L.get("rename", {}) or {}

        handles = [
            Patch(facecolor=color_map.get(s, "#999999"),
                  edgecolor=color_map.get(s, "#999999"),
                  label=rename.get(s, s))
            for s in legend_series
        ]

        place = L.get("place", "inside")
        ncol = int(L.get("ncol", 1))
        title_ = L.get("title", "")

        if place == "right":
            leg = ax.legend(handles=handles, title=title_, loc="center left",
                            bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=False, ncol=ncol)
            fig.subplots_adjust(right=0.80)
        elif place == "bottom":
            leg = ax.legend(handles=handles, title=title_, loc="upper center",
                            bbox_to_anchor=(0.5, -0.12), frameon=True, fancybox=False, ncol=max(ncol, len(handles)))
            fig.subplots_adjust(bottom=0.24)
        else:
            leg = ax.legend(handles=handles, title=title_,
                            loc=L.get("loc", "upper right"),
                            frameon=True, fancybox=False, ncol=ncol)

        frame = leg.get_frame()
        frame.set_facecolor("#f7f9fc")
        frame.set_edgecolor("#c7ced8")
        frame.set_linewidth(0.6)

    fig.tight_layout(pad=1.0)
    _save_figure(fig, Path(OUTDIR), base_name)
    plt.close(fig)


# -------------------- 导出工具 --------------------
def _save_figure(fig, outdir: Path, base: str):
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


# -------------------- 主流程 --------------------
def main():
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    roots = infer_roots_from_example(EXAMPLE_PATH)
    print("[INFO] Roots:")
    for r in roots:
        print("  -", r)

    rows = []
    for env, actors, run_id, label in CUSTOM_ITEMS_BAR:
        out_dirs = find_candidate_out_dirs(roots, env, actors, run_id)
        if not out_dirs:
            print(f"[WARN] 找不到 {env}/{actors}/{run_id} 的 out 目录，跳过：{label}")
            continue
        out_dir = out_dirs[0]
        sp_csv = Path(out_dir) / CSV_SP
        rx_csv = Path(out_dir) / CSV_RX
        if not sp_csv.exists() or not rx_csv.exists():
            print(f"[WARN] {out_dir} 缺少 {CSV_SP} 或 {CSV_RX}，跳过：{label}")
            continue

        try:
            sp_df = normalize_series_csv(sp_csv, "avg_sp_timeseries")
            rx_df = normalize_series_csv(rx_csv, "avg_rx_timeseries")
            if sp_df.empty or rx_df.empty:
                print(f"[INFO] {label} CSV 为空，跳过。")
                continue
            avg_sp, avg_rx, avg_cum_rx, avg_cum_sp = compute_metrics(sp_df, rx_df)
            rows.append({
                "env": env,
                "actors": actors,
                "run": run_id,
                "display": label,
                "avg_sp": avg_sp,
                "avg_rx": avg_rx,
                "avg_cum_rx": avg_cum_rx,
                "avg_cum_sp": avg_cum_sp,  # 新增列
            })
            print(f"[OK] 指标完成：{label} → {out_dir}")
        except Exception as e:
            print(f"[ERR] 处理 {label} 失败：{e}")

    if not rows:
        print("[WARN] 没有任何有效数据，结束。")
        return

    df_all = pd.DataFrame(rows)
    csv_out = Path(OUTDIR) / "bar_values_summary_twoenv.csv"
    df_all.to_csv(csv_out, index=False, encoding="utf-8")
    print(f"[DONE] 数值表已导出：{csv_out}")

    # —— 统一系列顺序 & 颜色：确保四张图颜色一致 —— #
    all_series_levels = list(dict.fromkeys([SERIES_KEY(a, r) for a, r in df_all[["actors", "run"]].to_numpy()]))
    series_palette = {s: c for s, c in zip(all_series_levels, distinct_colors(len(all_series_levels)))}

    # —— 图例重命名（把 run1/2/3 显示为 (K=?,M=20)） —— #
    legend_rename_map = {s: pretty_series_label(s) for s in all_series_levels}

    # —— 为每张图单独设置图例 —— #
    legend_sp = dict(show=True, place="inside", loc="upper left", ncol=2, rename=legend_rename_map)
    legend_rx = dict(show=True, place="inside", loc="upper right", ncol=2, rename=legend_rename_map)
    legend_cumrx = dict(show=True, place="inside", loc="upper right", ncol=2, rename=legend_rename_map)
    legend_cumsp = dict(show=True, place="inside", loc="upper left", ncol=2, rename=legend_rename_map)

    # —— 四张图分别绘制 —— #
    for metric_col, ylabel, title, base, L in [
        ("avg_sp", "Average SP", "", "twoenv_avg_sp", legend_sp),
        ("avg_rx", "Average received images", "", "twoenv_avg_rx", legend_rx),
        ("avg_cum_rx", "Average cumulative received images", "", "twoenv_avg_cum_rx", legend_cumrx),
        ("avg_cum_sp", "Average cumulative SP", "", "twoenv_avg_cum_sp", legend_cumsp),  # 新增
    ]:
        df_plot = df_all[["env", "actors", "run", metric_col]].rename(columns={metric_col: "value"}).copy()
        plot_two_env_multiseries(
            df_plot,
            metric_col="value",
            ylabel=ylabel,
            title=title,
            base_name=base,
            series_levels_all=all_series_levels,
            series_palette=series_palette,
            legend_spec=L,
            value_labels=True,
        )

    print(f"[DONE] 四张柱状图已输出到 {OUTDIR}/pdf, eps, svg, png, tiff")


if __name__ == "__main__":
    main()