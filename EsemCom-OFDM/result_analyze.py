#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Static 3-D surface plots for PSNR / SSIM / LPIPS
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from pathlib import Path

# ---------- 全局配置 ----------
CSV_PATH = r"experiments_csv/experiments_20250618_144300.csv"
OUT_DIR = Path("result_analyze")
ORTHO = True
RASTER_DPI = 600
SAVE_FORMATS = ("pdf", "png", "svg", "eps")

# 矢量输出字体
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["pdf.use14corefonts"] = False
matplotlib.rcParams["patch.force_edgecolor"] = False

# 🔧 可调参数：画布尺寸（英寸）与“绝对英寸”页边距
FIGSIZE_INCH = (8.0, 6.0)  # (宽, 高) 英寸
MARGINS_INCH = dict(
    left=0.01,  # 左边距（英寸）
    right=0.01,  # 右边距（英寸，建议稍大给 colorbar 留地）
    top=0.1,  # 上边距（英寸）
    bottom=0.1  # 下边距（英寸）
)
# 如果你还想让保存时自动压缩边缘，请设为 True（会覆盖手动边距）
USE_TIGHT_SAVE = False


def find_col(df: pd.DataFrame, key: str) -> str:
    key_low = key.lower()
    for c in df.columns:
        if key_low in c.lower():
            return c
    raise KeyError(f"找不到包含关键字 '{key}' 的列，请检查 CSV 列名。")


def build_mesh(df, snr_c, step_c, metric_c, delay_c, snrs, steps):
    Z = np.zeros((len(steps), len(snrs)), dtype=float)
    D = np.zeros_like(Z)
    for i, s in enumerate(steps):
        for j, snr in enumerate(snrs):
            rows = df[(df[step_c] == s) & (df[snr_c] == snr)]
            if rows.empty:
                raise ValueError(f"找不到 step={s}, snr={snr} 的数据行")
            row = rows.iloc[0]
            Z[i, j] = float(row[metric_c])
            D[i, j] = float(row[delay_c])
    return Z, D


def _apply_absolute_margins(fig: plt.Figure, margins_inch: dict):
    """
    用英寸边距换算成 subplots_adjust 需要的 0~1 归一化值。
    """
    fw, fh = fig.get_size_inches()  # 画布宽高（英寸）
    l = margins_inch.get("left", 0.0) / fw
    r = 1.0 - margins_inch.get("right", 0.0) / fw
    b = margins_inch.get("bottom", 0.0) / fh
    t = 1.0 - margins_inch.get("top", 0.0) / fh
    # 保护：若边距过大导致区间反转，适度回退
    eps = 0.02
    if r - l < eps:
        mid = (r + l) / 2.0
        l, r = mid - eps / 2, mid + eps / 2
    if t - b < eps:
        mid = (t + b) / 2.0
        b, t = mid - eps / 2, mid + eps / 2
    fig.subplots_adjust(left=l, right=r, bottom=b, top=t)


def _save_multi_formats(fig: plt.Figure, base_path: Path, cbar=None):
    base = base_path.with_suffix("")
    extras = list(fig.texts)
    if cbar is not None:
        extras.append(cbar.ax)

    def save(path, **kw):
        if USE_TIGHT_SAVE:
            # 注意：tight 会覆盖手动边距；只在你想“尽可能贴边”时启用
            fig.savefig(path, bbox_inches="tight", pad_inches=0.08,
                        bbox_extra_artists=extras, **kw)
        else:
            fig.savefig(path, **kw)  # 保留我们手动设置的边距

    if "pdf" in SAVE_FORMATS:
        save(base.with_suffix(".pdf"))
    if "png" in SAVE_FORMATS:
        save(base.with_suffix(".png"), dpi=RASTER_DPI)
    if "svg" in SAVE_FORMATS:
        save(base.with_suffix(".svg"))
    if "eps" in SAVE_FORMATS:
        save(base.with_suffix(".eps"), format="eps")


def draw(title, Z, D, snrs, steps, out_path, invert_z=False,
         cbar_label="Average encoding delay (s/img)"):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 关闭 constrained_layout，改用我们自己的边距
    fig = plt.figure(figsize=FIGSIZE_INCH, constrained_layout=False)
    ax = fig.add_subplot(111, projection="3d")

    if ORTHO and hasattr(ax, "set_proj_type"):
        ax.set_proj_type("ortho")

    X, Y = np.meshgrid(snrs, steps)

    norm = colors.Normalize(vmin=np.nanmin(D), vmax=np.nanmax(D))
    facecolors = cm.viridis(norm(D))

    surf = ax.plot_surface(
        X, Y, Z,
        rstride=1, cstride=1,
        facecolors=facecolors,
        linewidth=0,
        antialiased=True,
        shade=True,
    )
    try:
        surf.set_rasterized(False)
    except Exception:
        pass

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Steps", fontsize=12)
    ax.set_zlabel(title, fontsize=12)
    ax.invert_yaxis()
    ax.view_init(elev=30, azim=45)
    ax.grid(True)

    if invert_z:
        ax.invert_zaxis()

    # 颜色条（默认在右侧；pad 是相对轴宽的比例）
    mappable = cm.ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array(D)
    cbar = fig.colorbar(mappable, ax=ax, orientation="vertical", pad=0.05, shrink=0.6)

    # 【修改点】：设置 fontsize=16，调大标题
    cbar.set_label(cbar_label, fontsize=14)

    try:
        cbar.solids.set_edgecolor("face")
        cbar.solids.set_rasterized(False)
    except Exception:
        pass

    # —— 关键：按“英寸”设置四周留白（确保不被吃掉）——
    _apply_absolute_margins(fig, MARGINS_INCH)

    _save_multi_formats(fig, out_path, cbar=cbar)
    plt.close(fig)


def main():
    if not Path(CSV_PATH).exists():
        print(f"Error: CSV file not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    snr_c = find_col(df, "snr")
    step_c = find_col(df, "step")
    psnr_c = find_col(df, "psnr")
    ssim_c = find_col(df, "ssim")
    lpips_c = find_col(df, "lpips")
    total_c = find_col(df, "total")

    count_c = df.columns[0]
    df[count_c] = pd.to_numeric(df[count_c], errors="coerce")
    if df[count_c].isna().any():
        raise ValueError(f"第一列 '{count_c}' 中存在非数值，无法用于除法计算。")
    if (df[count_c] == 0).any():
        raise ValueError(f"第一列 '{count_c}' 中存在 0，无法计算平均时延。")

    df["avg_delay_s_per_img"] = df[total_c] / df[count_c]

    snrs = np.sort(df[snr_c].unique())
    steps = np.sort(df[step_c].unique())[::-1]

    metrics = [
        ("PSNR (dB)", psnr_c, "psnr_surface", False),
        ("MS-SSIM", ssim_c, "ssim_surface", False),
        ("LPIPS", lpips_c, "lpips_surface", True),
    ]

    for title, col, basename, inv in metrics:
        Z, D = build_mesh(df, snr_c, step_c, col, "avg_delay_s_per_img", snrs, steps)
        out_base = OUT_DIR / basename
        draw(title, Z, D, snrs, steps, out_base, invert_z=inv,
             cbar_label="Average encoding delay (s/img)")


if __name__ == "__main__":
    main()
