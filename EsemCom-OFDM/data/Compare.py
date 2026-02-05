#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量：JPEG vs BPG 压缩 + 质量(LPIPS/SSIM/PSNR) + 计时
并加入信道编码(POLAR/LDPC-like)后的时延
—— 显示口径（按你的要求）——
- 原图：MB（十进制）
- 仅信源编码后的数据量：KB（十进制） + 相对原图压缩率(%)
- 信源+信道编码后的数据量：KB（十进制） + 相对原图压缩率(%)

其它：
- 时间：秒（均值±标准差，4 位有效数字）
- 质量：LPIPS（越小越好）、SSIM（越大越好）、PSNR dB（越大越好）
- 计时策略：JPEG 记录编码+解码；BPG 记录编码+解码；FEC 记录编码
- 控制台：逐图详细打印；结束时打印数据集“平均口径”和“总体(加总)口径”
- 导出：CSV / JSON / TXT

依赖：
  pip install pillow numpy torch lpips scikit-image
"""

import csv
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import random
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image

# ======= 可按需修改的默认参数 =======
JPEG_QUALITY = 20  # 1~95
JPEG_SUBSAMPLING = 2  # 0=4:4:4, 1=4:2:2, 2=4:2:0
BPG_QP = 35  # bpgenc -q（越小越高质）
BPG_CHROMA = "420"  # '444'/'422'/'420'/'400'；None=默认
LPIPS_NET = "alex"  # 'alex'|'vgg'|'squeeze'
RUNS = 5  # 图像编解码计时重复次数（>=3 更稳）
OUTDIR = "outputs"  # 输出目录
RECURSIVE = True  # 是否递归扫描子目录
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
MAX_IMAGES = 10  # 设为 None 表示不限制
RANDOM_SEED = 42

# 信道编码（FEC）设置（教学型编码器，便于计时与数据量统计）
POLAR_BLOCK_N = 1024                 # 极化码块长 N=2^n
POLAR_RATES = [1/3, 1/2]             # 1/3, 1/2
LDPC_BLOCK_K = 1200                  # 教学型 LDPC-like 每块信息位
LDPC_COL_WEIGHT = 3                  # 奇偶方程列重
LDPC_RATES = [2/3, 3/4]              # 2/3, 3/4
RUNS_FEC = 3                         # FEC 编码计时重复次数（减少整体耗时）

# Windows 示例：把下面改成你的 BPG 文件夹（不是 .exe）
DEFAULT_BPG_DIR: str = r"C:\Users\ENeS\Desktop\BPG"
EXE = ".exe" if os.name == "nt" else ""

# ======= GUI（可选）=======
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False


def ask_open_dir(title="选择原图文件夹") -> Optional[str]:
    if not TK_AVAILABLE: return None
    root = tk.Tk(); root.withdraw()
    d = filedialog.askdirectory(title=title)
    return d or None


def ask_bpg_dir() -> Optional[str]:
    if not TK_AVAILABLE: return None
    root = tk.Tk(); root.withdraw()
    d = filedialog.askdirectory(title="选择包含 bpgenc/bpgdec 的目录")
    return d or None


def info_box(msg: str):
    print(msg)
    if TK_AVAILABLE:
        try:
            root = tk.Tk(); root.withdraw()
            messagebox.showinfo("信息", msg)
        except Exception:
            pass


def warn_box(msg: str):
    print("[WARN]", msg, file=sys.stderr)
    if TK_AVAILABLE:
        try:
            root = tk.Tk(); root.withdraw()
            messagebox.showwarning("警告", msg)
        except Exception:
            pass


def error_box(msg: str):
    print("[ERROR]", msg, file=sys.stderr)
    if TK_AVAILABLE:
        try:
            root = tk.Tk(); root.withdraw()
            messagebox.showerror("错误", msg)
        except Exception:
            pass


# ======= 工具函数 =======
def is_image_file(path: str) -> bool:
    return pathlib.Path(path).suffix.lower() in IMAGE_EXTS


def list_images(root_dir: str, recursive: bool = True) -> List[str]:
    imgs: List[str] = []
    p = pathlib.Path(root_dir)
    it = p.rglob("*") if recursive else p.glob("*")
    for fp in it:
        if fp.is_file() and is_image_file(str(fp)):
            imgs.append(str(fp))
    imgs.sort()
    return imgs


def ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def load_image_rgb(path: str) -> Image.Image:
    return ensure_rgb(Image.open(path))


def pil_to_lpips_tensor(img: Image.Image):
    """
    LPIPS 输入：
      - RGB
      - [H,W,3] -> [1,3,H,W]
      - 归一化到 [-1,1]
    """
    import torch
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    ten = ten * 2 - 1
    return ten


def fmt_sig(x: Optional[float], sig: int = 4, unit: str = "s") -> str:
    """4 位有效数字格式化（秒）；x=None 返回 'n/a'。"""
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x:.{sig}g} {unit}"


# 十进制单位显示（按你的要求）
def fmt_kB_dec(nbytes: int) -> str:
    return f"{nbytes/1_000:.2f} kB"

def fmt_MB_dec(nbytes: int) -> str:
    return f"{nbytes/1_000_000:.2f} MB"


def mean_std(times: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not times: return None, None
    if len(times) == 1: return times[0], 0.0
    return statistics.mean(times), statistics.stdev(times)


def build_env_with_dir(dir_path: Optional[str]) -> dict:
    env = os.environ.copy()
    if dir_path:
        env["PATH"] = dir_path + os.pathsep + env.get("PATH", "")
    return env


# ======= 比特 <-> 字节 =======
def bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits.astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.size == 0:
        return b""
    pad = (-bits.size) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    arr = np.packbits(bits.astype(np.uint8))
    return arr.tobytes()


# ======= BPG 可执行文件解析 =======
def try_abs_bins(bpg_dir: Optional[str]) -> Optional[Tuple[str, str]]:
    if not bpg_dir: return None
    enc = os.path.join(bpg_dir, f"bpgenc{EXE}")
    dec = os.path.join(bpg_dir, f"bpgdec{EXE}")
    if os.path.isfile(enc) and os.path.isfile(dec):
        return enc, dec
    return None


def try_bins_in_path(env: Optional[dict]) -> Optional[Tuple[str, str]]:
    try:
        if os.name == "nt":
            r1 = subprocess.run(["where", "bpgenc"], capture_output=True, env=env)
            r2 = subprocess.run(["where", "bpgdec"], capture_output=True, env=env)
        else:
            r1 = subprocess.run(["which", "bpgenc"], capture_output=True, env=env)
            r2 = subprocess.run(["which", "bpgdec"], capture_output=True, env=env)
        if r1.returncode == 0 and r2.returncode == 0:
            return ("bpgenc", "bpgdec")
    except Exception:
        pass
    return None


def resolve_bpg_bins(initial_dir: Optional[str]) -> Tuple[Optional[Tuple[str, str]], dict]:
    """
    返回 (bins, env)：(enc_bin, dec_bin) 或 None（不可用），以及包含PATH注入的env
    解析顺序：绝对路径优先 → PATH → 弹窗 → 不可用(None)
    """
    env = build_env_with_dir(initial_dir)
    bins = try_abs_bins(initial_dir)
    if bins: return bins, env
    bins = try_bins_in_path(env)
    if bins: return bins, env
    chosen = ask_bpg_dir()
    env2 = build_env_with_dir(chosen)
    bins = try_abs_bins(chosen) or try_bins_in_path(env2)
    if bins:
        info_box(f"已使用 BPG 目录：{chosen or '(PATH)'}")
        return bins, env2
    warn_box("未找到 bpgenc/bpgdec，将仅输出 JPEG 结果。")
    return None, env


# ======= SSIM / PSNR =======
def to_float01_rgb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def compute_ssim_and_psnr(ref_img: Image.Image, test_img: Image.Image) -> Tuple[float, float]:
    ref = to_float01_rgb(ref_img)
    tst = to_float01_rgb(test_img)
    if ref.shape != tst.shape:
        h, w = ref.shape[:2]
        test_img = test_img.resize((w, h), Image.BICUBIC)
        tst = to_float01_rgb(test_img)
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        try:
            ssim_val = float(ssim(ref, tst, channel_axis=-1, data_range=1.0))
        except TypeError:
            ssim_val = float(ssim(ref, tst, multichannel=True, data_range=1.0))
        psnr_val = float(psnr(ref, tst, data_range=1.0))
    except Exception:
        mse = float(np.mean((ref - tst) ** 2))
        psnr_val = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
        ssim_val = float("nan")
    return ssim_val, psnr_val


# ======= Polar 编码（参考实现，Arikan 变换）=======
def _polar_butterfly_inplace(x: np.ndarray):
    """就地执行 F^{⊗n} over GF(2)，x 为 uint8(0/1)，长度 N=2^n"""
    N = x.size
    s = 1
    while s < N:
        step = s * 2
        for i in range(0, N, step):
            x[i:i+s] ^= x[i+s:i+step]
        s *= 2


def polar_encode_block(msg_bits: np.ndarray, N: int, K: int) -> np.ndarray:
    """
    简化极化编码：信息位放在前 K 个位置，剩余冻结为 0。
    实际工程会依据可靠性序列选择信息位，这里为可运行基准。
    """
    assert N & (N-1) == 0, "N 必须为 2^n"
    assert 0 < K <= N
    u = np.zeros(N, dtype=np.uint8)
    m = msg_bits
    if m.size < K:
        tmp = np.zeros(K, dtype=np.uint8); tmp[:m.size] = m; m = tmp
    u[:K] = m[:K]
    x = u.copy()
    _polar_butterfly_inplace(x)
    return x


def measure_polar_on_bytes(data: bytes, rate: float, N: int, runs: int) -> Dict[str, Any]:
    """
    对字节流逐块极化编码，统计平均编码时延与编码后总字节数。
    """
    bits = bytes_to_bits(data)
    K = int(round(rate * N))
    K = max(1, min(K, N))
    blocks = (bits.size + K - 1) // K
    enc_times: List[float] = []
    enc_out_bits = None

    for _ in range(runs):
        t0 = time.perf_counter()
        out_list = []
        ptr = 0
        for _b in range(blocks):
            m = bits[ptr:ptr+K]
            ptr += K
            c = polar_encode_block(m, N=N, K=K)
            out_list.append(c)
        out = np.concatenate(out_list) if out_list else np.zeros(0, dtype=np.uint8)
        t1 = time.perf_counter()
        enc_times.append(t1 - t0)
        enc_out_bits = out

    enc_mean, enc_std = mean_std(enc_times)
    enc_bytes = len(bits_to_bytes(enc_out_bits))
    return {
        "fec_bytes": enc_bytes,
        "enc_mean_s": enc_mean,
        "enc_std_s": enc_std,
        "eff_rate": (K / N)
    }


# ======= LDPC-like 编码（教学型稀疏系统码，非标准）=======
class LDPCLikeEncoder:
    """
    玩具型“LDPC-like”系统码：
      - 每块信息长 k，码率 R => 总长 n = ceil(k/R)，奇偶位数 p = n - k ≈ k*(1/R - 1)
      - 构造 p 个奇偶方程，每个方程对若干个信息位做 XOR（列重可调）
      - 码字为 [info_bits, parity_bits]
    仅用于：统计编码时间与扩展后数据量。非标准 LDPC。
    """
    def __init__(self, k_block: int, rate: float, col_weight: int = 3, seed: int = 0):
        assert 0 < rate < 1
        self.k = int(k_block)
        self.n = int(math.ceil(self.k / rate))
        self.p = self.n - self.k
        self.col_w = int(max(1, col_weight))
        random.seed(seed)
        self.rows: List[List[int]] = []
        for _ in range(self.p):
            idxs = random.sample(range(self.k), self.col_w)
            self.rows.append(sorted(idxs))

    def encode_block(self, m: np.ndarray) -> np.ndarray:
        if m.size < self.k:
            tmp = np.zeros(self.k, dtype=np.uint8); tmp[:m.size] = m; m = tmp
        parity = np.zeros(self.p, dtype=np.uint8)
        for j, idxs in enumerate(self.rows):
            parity[j] = np.bitwise_xor.reduce(m[idxs])
        c = np.concatenate([m[:self.k], parity])
        return c


def measure_ldpc_on_bytes(data: bytes, rate: float, k_block: int, col_w: int, runs: int, seed: int) -> Dict[str, Any]:
    bits = bytes_to_bits(data)
    enc = LDPCLikeEncoder(k_block=k_block, rate=rate, col_weight=col_w, seed=seed)
    blocks = (bits.size + enc.k - 1) // enc.k
    enc_times: List[float] = []
    enc_out_bits = None

    for _ in range(runs):
        t0 = time.perf_counter()
        out_list = []
        ptr = 0
        for _b in range(blocks):
            m = bits[ptr:ptr+enc.k]
            ptr += enc.k
            c = enc.encode_block(m)
            out_list.append(c)
        out = np.concatenate(out_list) if out_list else np.zeros(0, dtype=np.uint8)
        t1 = time.perf_counter()
        enc_times.append(t1 - t0)
        enc_out_bits = out

    enc_mean, enc_std = mean_std(enc_times)
    enc_bytes = len(bits_to_bytes(enc_out_bits))
    return {
        "fec_bytes": enc_bytes,
        "enc_mean_s": enc_mean,
        "enc_std_s": enc_std,
        "eff_rate": (enc.k / enc.n)
    }


# ======= 单图处理（含“KB/MB + 压缩率”显示所需字段）=======
def process_one_image(
        in_path: str,
        outdir: str,
        out_prefix: str,
        jpeg_quality: int,
        jpeg_subsampling: int,
        bpg_qp: int,
        bpg_chroma: Optional[str],
        runs: int,
        bpg_bins: Optional[Tuple[str, str]],
        env_bpg: dict
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "path": in_path, "w": None, "h": None,
        "input_bytes": None,
        "jpeg": None, "bpg": None
    }

    # 原图
    img = load_image_rgb(in_path)
    w, h = img.size
    rec["w"], rec["h"] = w, h
    size_input_file = os.path.getsize(in_path)
    rec["input_bytes"] = size_input_file

    # LPIPS
    import torch, lpips
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net=LPIPS_NET).to(device).eval()
    ref_ten = pil_to_lpips_tensor(img).to(device)

    # ---------- JPEG ----------
    jpeg_enc_times: List[float] = []
    jpeg_dec_times: List[float] = []
    jpeg_last_path: Optional[str] = None

    for run in range(runs):
        jpeg_path = os.path.join(outdir, f"{out_prefix}_jpeg_q{jpeg_quality}_ss{jpeg_subsampling}_r{run+1}.jpg")
        # 编码
        t0 = time.perf_counter()
        img.save(
            jpeg_path,
            format="JPEG",
            quality=int(jpeg_quality),
            subsampling=int(jpeg_subsampling),
            optimize=True,
            progressive=False
        )
        t1 = time.perf_counter()
        jpeg_enc_times.append(t1 - t0)
        # 解码
        t2 = time.perf_counter()
        dimg = Image.open(jpeg_path).convert("RGB"); dimg.load()
        t3 = time.perf_counter()
        jpeg_dec_times.append(t3 - t2)
        jpeg_last_path = jpeg_path

    assert jpeg_last_path is not None
    size_jpeg = os.path.getsize(jpeg_last_path)

    # 质量
    jpeg_img = load_image_rgb(jpeg_last_path)
    if jpeg_img.size != (w, h):
        jpeg_img = jpeg_img.resize((w, h), Image.LANCZOS)
    ten_jpeg = pil_to_lpips_tensor(jpeg_img).to(device)
    with torch.no_grad():
        lpips_jpeg = float(loss_fn(ref_ten, ten_jpeg).item())
    ssim_jpeg, psnr_jpeg = compute_ssim_and_psnr(img, jpeg_img)

    jm, js = mean_std(jpeg_enc_times)
    jdm, jds = mean_std(jpeg_dec_times)

    # —— 仅信源编码相对原图压缩率（显示用） ——
    jpeg_saving_percent = 100.0 * (1.0 - size_jpeg / size_input_file) if size_input_file > 0 else float("nan")

    # —— JPEG 码流做 FEC（Polar/LDPC）并计算“含FEC压缩率与 KB 大小” ——
    with open(jpeg_last_path, "rb") as f:
        jpeg_bytes = f.read()
    fec_results_jpeg: Dict[str, Any] = {}
    # Polar
    for r in POLAR_RATES:
        res = measure_polar_on_bytes(jpeg_bytes, rate=r, N=POLAR_BLOCK_N, runs=RUNS_FEC)
        key = f"polar_r{int(round(r*100)):02d}"  # r33, r50
        res["saving_percent_with_fec"] = 100.0 * (1.0 - res["fec_bytes"]/size_input_file) if size_input_file>0 else float("nan")
        res["fec_size_kB_dec"] = res["fec_bytes"] / 1_000.0
        res["pipe_enc_mean_s"] = (jm or 0.0) + (res["enc_mean_s"] or 0.0)   # 编码总时延
        res["e2e_mean_s"] = res["pipe_enc_mean_s"] + (jdm or 0.0)           # 端到端(不含FEC解码)
        fec_results_jpeg[key] = res
    # LDPC-like
    for r in LDPC_RATES:
        res = measure_ldpc_on_bytes(jpeg_bytes, rate=r, k_block=LDPC_BLOCK_K,
                                    col_w=LDPC_COL_WEIGHT, runs=RUNS_FEC, seed=RANDOM_SEED)
        key = f"ldpc_r{int(round(r*100)):02d}"   # r67, r75
        res["saving_percent_with_fec"] = 100.0 * (1.0 - res["fec_bytes"]/size_input_file) if size_input_file>0 else float("nan")
        res["fec_size_kB_dec"] = res["fec_bytes"] / 1_000.0
        res["pipe_enc_mean_s"] = (jm or 0.0) + (res["enc_mean_s"] or 0.0)
        res["e2e_mean_s"] = res["pipe_enc_mean_s"] + (jdm or 0.0)
        fec_results_jpeg[key] = res

    rec["jpeg"] = {
        "size_bytes": size_jpeg,
        "size_kB_dec": size_jpeg / 1_000.0,
        "saving_percent": jpeg_saving_percent,
        "enc_mean_s": jm, "enc_std_s": js,
        "dec_mean_s": jdm, "dec_std_s": jds,
        "LPIPS": lpips_jpeg,
        "SSIM": ssim_jpeg,
        "PSNR_dB": psnr_jpeg,
        "out_path": jpeg_last_path,
        "fec": fec_results_jpeg
    }

    # ---------- BPG ----------
    if bpg_bins is not None:
        enc_bin, dec_bin = bpg_bins
        with tempfile.TemporaryDirectory(prefix="bpgtmp_") as td:
            tmp_png_in = os.path.join(td, "in.png")
            ensure_rgb(img).save(tmp_png_in, format="PNG")

            bpg_enc_times: List[float] = []
            bpg_dec_times: List[float] = []
            bpg_last_path: Optional[str] = None
            bpg_last_decoded_png: Optional[str] = None

            for run in range(runs):
                bpg_path = os.path.join(outdir, f"{out_prefix}_bpg_qp{bpg_qp}_r{run+1}.bpg")
                bpg_png_out = os.path.join(outdir, f"{out_prefix}_bpg_qp{bpg_qp}_r{run+1}_decoded.png")

                # 编码
                enc_cmd = [enc_bin, "-q", str(int(bpg_qp))]
                if BPG_CHROMA in {"444", "422", "420", "400"}:
                    enc_cmd += ["-f", BPG_CHROMA]
                enc_cmd += ["-o", bpg_path, tmp_png_in]
                t0 = time.perf_counter()
                r = subprocess.run(enc_cmd, capture_output=True, env=env_bpg)
                t1 = time.perf_counter()
                if r.returncode != 0 or (not os.path.exists(bpg_path)):
                    sys.stderr.write(r.stderr.decode(errors="ignore"))
                    raise RuntimeError("bpgenc 编码失败")
                bpg_enc_times.append(t1 - t0)

                # 解码
                dec_cmd = [dec_bin, "-o", bpg_png_out, bpg_path]
                t2 = time.perf_counter()
                r2 = subprocess.run(dec_cmd, capture_output=True, env=env_bpg)
                t3 = time.perf_counter()
                if r2.returncode != 0 or (not os.path.exists(bpg_png_out)):
                    sys.stderr.write(r2.stderr.decode(errors="ignore"))
                    raise RuntimeError("bpgdec 解码失败")
                bpg_dec_times.append(t3 - t2)

                bpg_last_path = bpg_path
                bpg_last_decoded_png = bpg_png_out

        assert bpg_last_path is not None and bpg_last_decoded_png is not None
        size_bpg = os.path.getsize(bpg_last_path)

        # 质量
        bpg_img = load_image_rgb(bpg_last_decoded_png)
        if bpg_img.size != (w, h):
            bpg_img = bpg_img.resize((w, h), Image.LANCZOS)
        ten_bpg = pil_to_lpips_tensor(bpg_img).to(device)
        with torch.no_grad():
            lpips_bpg = float(loss_fn(ref_ten, ten_bpg).item())
        ssim_bpg, psnr_bpg = compute_ssim_and_psnr(img, bpg_img)

        bm, bs = mean_std(bpg_enc_times)
        bdm, bds = mean_std(bpg_dec_times)

        # —— 仅信源编码相对原图压缩率（显示用） ——
        bpg_saving_percent = 100.0 * (1.0 - size_bpg / size_input_file) if size_input_file > 0 else float("nan")

        # —— BPG 码流做 FEC 并计算“含FEC压缩率与 KB 大小” ——
        with open(bpg_last_path, "rb") as f:
            bpg_bytes = f.read()
        fec_results_bpg: Dict[str, Any] = {}
        for r in POLAR_RATES:
            res = measure_polar_on_bytes(bpg_bytes, rate=r, N=POLAR_BLOCK_N, runs=RUNS_FEC)
            key = f"polar_r{int(round(r*100)):02d}"
            res["saving_percent_with_fec"] = 100.0 * (1.0 - res["fec_bytes"]/size_input_file) if size_input_file>0 else float("nan")
            res["fec_size_kB_dec"] = res["fec_bytes"] / 1_000.0
            res["pipe_enc_mean_s"] = (bm or 0.0) + (res["enc_mean_s"] or 0.0)
            res["e2e_mean_s"] = res["pipe_enc_mean_s"] + (bdm or 0.0)
            fec_results_bpg[key] = res
        for r in LDPC_RATES:
            res = measure_ldpc_on_bytes(bpg_bytes, rate=r, k_block=LDPC_BLOCK_K,
                                        col_w=LDPC_COL_WEIGHT, runs=RUNS_FEC, seed=RANDOM_SEED)
            key = f"ldpc_r{int(round(r*100)):02d}"
            res["saving_percent_with_fec"] = 100.0 * (1.0 - res["fec_bytes"]/size_input_file) if size_input_file>0 else float("nan")
            res["fec_size_kB_dec"] = res["fec_bytes"] / 1_000.0
            res["pipe_enc_mean_s"] = (bm or 0.0) + (res["enc_mean_s"] or 0.0)
            res["e2e_mean_s"] = res["pipe_enc_mean_s"] + (bdm or 0.0)
            fec_results_bpg[key] = res

        rec["bpg"] = {
            "size_bytes": size_bpg,
            "size_kB_dec": size_bpg / 1_000.0,
            "saving_percent": bpg_saving_percent,
            "enc_mean_s": bm, "enc_std_s": bs,
            "dec_mean_s": bdm, "dec_std_s": bds,
            "LPIPS": lpips_bpg,
            "SSIM": ssim_bpg,
            "PSNR_dB": psnr_bpg,
            "out_path": bpg_last_path,
            "decoded_png": bpg_last_decoded_png,
            "fec": fec_results_bpg
        }
    else:
        rec["bpg"] = None

    return rec


# ======= 汇总与导出（含“MB/KB + 压缩率”字段）=======
def safe_mean(vals: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None and math.isfinite(v)]
    return statistics.mean(xs) if xs else None


def export_reports(outdir: str, records: List[Dict[str, Any]], root_dir: str):
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "dataset_report.csv")
    json_path = os.path.join(outdir, "dataset_report.json")
    txt_path  = os.path.join(outdir, "dataset_report.txt")

    # FEC 键与标签
    fec_keys = [
        ("polar_r33", "Polar(1/3)"),
        ("polar_r50", "Polar(1/2)"),
        ("ldpc_r67",  "LDPC(2/3)"),
        ("ldpc_r75",  "LDPC(3/4)"),
    ]

    def fec_cols(prefix: str):
        cols = []
        for key, _ in fec_keys:
            cols += [
                f"{prefix}_{key}_bytes",                      # FEC 编码后字节
                f"{prefix}_{key}_size_kB_dec",                # FEC 后大小（kB 十进制）
                f"{prefix}_{key}_saving_percent_with_fec",    # 加FEC后的压缩率(%)
                f"{prefix}_{key}_fec_enc_mean_s",
                f"{prefix}_{key}_fec_enc_std_s",
                f"{prefix}_{key}_pipe_enc_mean_s",            # 图像编码 + FEC编码
                f"{prefix}_{key}_e2e_mean_s",                 # pipe编码 + 图像解码
                f"{prefix}_{key}_eff_rate"
            ]
        return cols

    headers = [
        "path","w","h","input_bytes","input_size_MB_dec",
        # JPEG（含解码时间）
        "jpeg_size_bytes","jpeg_size_kB_dec","jpeg_saving_percent",
        "jpeg_LPIPS","jpeg_SSIM","jpeg_PSNR_dB",
        "jpeg_enc_mean_s","jpeg_enc_std_s","jpeg_dec_mean_s","jpeg_dec_std_s",
    ] + fec_cols("jpeg") + [
        # BPG（含解码时间）
        "bpg_size_bytes","bpg_size_kB_dec","bpg_saving_percent",
        "bpg_LPIPS","bpg_SSIM","bpg_PSNR_dB",
        "bpg_enc_mean_s","bpg_enc_std_s","bpg_dec_mean_s","bpg_dec_std_s",
    ] + fec_cols("bpg")

    rows: List[Dict[str, Any]] = []
    for r in records:
        jr = r["jpeg"] or {}
        br = r["bpg"] or {}
        row = {
            "path": r["path"],
            "w": r["w"], "h": r["h"],
            "input_bytes": r["input_bytes"],
            "input_size_MB_dec": (r["input_bytes"] or 0) / 1_000_000.0,
            # JPEG
            "jpeg_size_bytes": jr.get("size_bytes"),
            "jpeg_size_kB_dec": jr.get("size_kB_dec"),
            "jpeg_saving_percent": jr.get("saving_percent"),
            "jpeg_LPIPS": jr.get("LPIPS"),
            "jpeg_SSIM": jr.get("SSIM"),
            "jpeg_PSNR_dB": jr.get("PSNR_dB"),
            "jpeg_enc_mean_s": jr.get("enc_mean_s"),
            "jpeg_enc_std_s": jr.get("enc_std_s"),
            "jpeg_dec_mean_s": jr.get("dec_mean_s"),
            "jpeg_dec_std_s": jr.get("dec_std_s"),
            # BPG
            "bpg_size_bytes": br.get("size_bytes"),
            "bpg_size_kB_dec": br.get("size_kB_dec"),
            "bpg_saving_percent": br.get("saving_percent"),
            "bpg_LPIPS": br.get("LPIPS"),
            "bpg_SSIM": br.get("SSIM"),
            "bpg_PSNR_dB": br.get("PSNR_dB"),
            "bpg_enc_mean_s": br.get("enc_mean_s"),
            "bpg_enc_std_s": br.get("enc_std_s"),
            "bpg_dec_mean_s": br.get("dec_mean_s"),
            "bpg_dec_std_s": br.get("dec_std_s"),
        }

        def fill_fec(prefix: str, sub: Dict[str, Any]):
            for key, _ in fec_keys:
                d = (sub or {}).get(key) or {}
                row[f"{prefix}_{key}_bytes"] = d.get("fec_bytes")
                row[f"{prefix}_{key}_size_kB_dec"] = d.get("fec_size_kB_dec")
                row[f"{prefix}_{key}_saving_percent_with_fec"] = d.get("saving_percent_with_fec")
                row[f"{prefix}_{key}_fec_enc_mean_s"] = d.get("enc_mean_s")
                row[f"{prefix}_{key}_fec_enc_std_s"]  = d.get("enc_std_s")
                row[f"{prefix}_{key}_pipe_enc_mean_s"] = d.get("pipe_enc_mean_s")
                row[f"{prefix}_{key}_e2e_mean_s"]      = d.get("e2e_mean_s")
                row[f"{prefix}_{key}_eff_rate"]        = d.get("eff_rate")

        fill_fec("jpeg", jr.get("fec") or {})
        fill_fec("bpg",  br.get("fec") or {})

        rows.append(row)

    # ===== 数据集“平均口径”与“总体(加总)口径” =====
    def mean_field(records, codec: str, key: str) -> Optional[float]:
        vals = []
        for r in records:
            v = (r.get(codec) or {}).get(key)
            if v is not None and math.isfinite(v):
                vals.append(v)
        return statistics.mean(vals) if vals else None

    def mean_fec(records, codec: str, fec_key: str, subkey: str) -> Optional[float]:
        vals = []
        for r in records:
            c = r.get(codec) or {}
            f = (c.get("fec") or {}).get(fec_key) or {}
            v = f.get(subkey)
            if v is not None and math.isfinite(v):
                vals.append(v)
        return statistics.mean(vals) if vals else None

    # 总体(加总)口径：以全部图片的总输入字节为基准
    def overall_aggregate(records, codec: str, fec_key: Optional[str] = None):
        total_in = 0
        total_out = 0
        for r in records:
            total_in += int(r.get("input_bytes") or 0)
            if fec_key is None:
                c = r.get(codec) or {}
                total_out += int(c.get("size_bytes") or 0)
            else:
                c = r.get(codec) or {}
                f = (c.get("fec") or {}).get(fec_key) or {}
                total_out += int(f.get("fec_bytes") or 0)
        overall_percent = (100.0 * (1 - total_out / total_in)) if total_in > 0 else float("nan")
        return {
            "total_input_MB_dec": total_in / 1_000_000.0,
            "total_output_kB_dec": total_out / 1_000.0,
            "overall_saving_percent": overall_percent
        }

    fec_means = {}
    fec_overall = {}
    for fec_key, label in [
        ("polar_r33","Polar(1/3)"),
        ("polar_r50","Polar(1/2)"),
        ("ldpc_r67","LDPC(2/3)"),
        ("ldpc_r75","LDPC(3/4)")
    ]:
        fec_means[f"jpeg_{fec_key}"] = {
            "fec_size_kB_dec_mean":        mean_fec(records, "jpeg", fec_key, "fec_size_kB_dec"),
            "saving_percent_with_fec_mean": mean_fec(records, "jpeg", fec_key, "saving_percent_with_fec"),
            "fec_enc_mean_s":               mean_fec(records, "jpeg", fec_key, "enc_mean_s"),
            "pipe_enc_mean_s":              mean_fec(records, "jpeg", fec_key, "pipe_enc_mean_s"),
            "e2e_mean_s":                   mean_fec(records, "jpeg", fec_key, "e2e_mean_s"),
        }
        fec_means[f"bpg_{fec_key}"] = {
            "fec_size_kB_dec_mean":        mean_fec(records, "bpg", fec_key, "fec_size_kB_dec"),
            "saving_percent_with_fec_mean": mean_fec(records, "bpg", fec_key, "saving_percent_with_fec"),
            "fec_enc_mean_s":               mean_fec(records, "bpg", fec_key, "enc_mean_s"),
            "pipe_enc_mean_s":              mean_fec(records, "bpg", fec_key, "pipe_enc_mean_s"),
            "e2e_mean_s":                   mean_fec(records, "bpg", fec_key, "e2e_mean_s"),
        }
        fec_overall[f"jpeg_{fec_key}"] = overall_aggregate(records, "jpeg", fec_key)
        fec_overall[f"bpg_{fec_key}"]  = overall_aggregate(records, "bpg",  fec_key)

    summary = {
        "root_dir": os.path.abspath(root_dir),
        "num_images": len(records),
        "params": {
            "JPEG_QUALITY": JPEG_QUALITY,
            "JPEG_SUBSAMPLING": JPEG_SUBSAMPLING,
            "BPG_QP": BPG_QP,
            "BPG_CHROMA": BPG_CHROMA,
            "LPIPS_NET": LPIPS_NET,
            "RUNS": RUNS,
            "RUNS_FEC": RUNS_FEC,
            "RECURSIVE": RECURSIVE,
            "POLAR_BLOCK_N": POLAR_BLOCK_N,
            "POLAR_RATES": POLAR_RATES,
            "LDPC_BLOCK_K": LDPC_BLOCK_K,
            "LDPC_COL_WEIGHT": LDPC_COL_WEIGHT,
            "LDPC_RATES": LDPC_RATES,
        },
        "dataset_means": {
            "jpeg": {
                "size_kB_dec_mean": mean_field(records, "jpeg", "size_kB_dec"),
                "saving_percent_mean": mean_field(records, "jpeg", "saving_percent"),
                "LPIPS": mean_field(records, "jpeg", "LPIPS"),
                "SSIM":  mean_field(records, "jpeg", "SSIM"),
                "PSNR_dB": mean_field(records, "jpeg", "PSNR_dB"),
                "enc_mean_s": mean_field(records, "jpeg", "enc_mean_s"),
                "dec_mean_s": mean_field(records, "jpeg", "dec_mean_s"),
            },
            "bpg": {
                "size_kB_dec_mean": mean_field(records, "bpg", "size_kB_dec"),
                "saving_percent_mean": mean_field(records, "bpg", "saving_percent"),
                "LPIPS": mean_field(records, "bpg", "LPIPS"),
                "SSIM":  mean_field(records, "bpg", "SSIM"),
                "PSNR_dB": mean_field(records, "bpg", "PSNR_dB"),
                "enc_mean_s": mean_field(records, "bpg", "enc_mean_s"),
                "dec_mean_s": mean_field(records, "bpg", "dec_mean_s"),
            },
            "fec_after_codec_means": fec_means
        },
        "overall": {
            "jpeg": overall_aggregate(records, "jpeg", None),
            "bpg":  overall_aggregate(records, "bpg",  None),
            "fec_after_codec_overall": fec_overall
        }
    }

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # JSON
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({"summary": summary, "records": rows}, jf, ensure_ascii=False, indent=2)

    # TXT（简洁均值与总体摘要）
    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(f"Root: {summary['root_dir']}\n")
        tf.write(f"Images: {summary['num_images']}\n")
        tf.write(f"Params: {json.dumps(summary['params'], ensure_ascii=False)}\n\n")
        tf.write("== Dataset Mean (平均口径) ==\n")
        tf.write(json.dumps(summary["dataset_means"], ensure_ascii=False, indent=2))
        tf.write("\n\n== Overall (总体加总口径) ==\n")
        tf.write(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
        tf.write("\n")

    print(f"[OK] 数据集报告已保存：\n- CSV:  {csv_path}\n- JSON: {json_path}\n- TXT:  {txt_path}")


# ======= 主流程 =======
def main():
    # 选择原图文件夹
    root_dir = ask_open_dir("选择原图文件夹")
    if not root_dir:
        error_box("未选择文件夹，已退出。"); return
    root_dir = os.path.abspath(root_dir)
    imgs = list_images(root_dir, RECURSIVE)
    if not imgs:
        error_box("该文件夹中未找到可识别的图片。"); return

    os.makedirs(OUTDIR, exist_ok=True)

    # 解析 BPG 可执行文件（绝对路径优先 → PATH → 弹窗）
    bpg_bins, env_bpg = resolve_bpg_bins(DEFAULT_BPG_DIR)

    # 抽样数量
    if MAX_IMAGES is not None and len(imgs) > MAX_IMAGES:
        random.seed(RANDOM_SEED)
        imgs = random.sample(imgs, MAX_IMAGES)
    n = len(imgs)

    print(f"[INFO] 共发现 {n} 张图片，开始处理...\n")

    records: List[Dict[str, Any]] = []
    for idx, path in enumerate(imgs, 1):
        rel = os.path.relpath(path, root_dir)
        stem = pathlib.Path(path).stem
        out_prefix = f"{idx:05d}_{stem}"
        print(f"[{idx}/{n}] {rel}")
        try:
            rec = process_one_image(
                in_path=path,
                outdir=OUTDIR,
                out_prefix=out_prefix,
                jpeg_quality=JPEG_QUALITY,
                jpeg_subsampling=JPEG_SUBSAMPLING,
                bpg_qp=BPG_QP,
                bpg_chroma=BPG_CHROMA,
                runs=RUNS,
                bpg_bins=bpg_bins,
                env_bpg=env_bpg
            )
            records.append(rec)

            # ===== 控制台摘要（逐图，按“MB/KB + 压缩率”展示） =====
            input_b = rec["input_bytes"]
            print(f"  原图: {fmt_MB_dec(input_b)}")

            def print_codec_line(tag: str, r: Dict[str, Any]):
                print(f"  {tag}: {fmt_kB_dec(int(r['size_bytes']))} | 压缩率={r['saving_percent']:.2f}% "
                      f"| LPIPS={r['LPIPS']:.4f} SSIM={r['SSIM']:.4f} PSNR={r['PSNR_dB']:.2f}dB "
                      f"| enc={fmt_sig(r['enc_mean_s'])}±{fmt_sig(r['enc_std_s'])} "
                      f"| dec={fmt_sig(r['dec_mean_s'])}±{fmt_sig(r['dec_std_s'])}")

            def print_fec_block(tag: str, fec_map: Dict[str, Any]):
                if not fec_map:
                    print(f"    {tag}+FEC: (无)"); return
                for key,label in [("polar_r33","Polar(1/3)"), ("polar_r50","Polar(1/2)"),
                                  ("ldpc_r67","LDPC(2/3)"), ("ldpc_r75","LDPC(3/4)")]:
                    d = fec_map.get(key)
                    if not d: continue
                    print(f"    {label}: FEC={d['fec_size_kB_dec']:.2f} kB | 压缩率(含FEC)={d['saving_percent_with_fec']:.2f}% "
                          f"| FECenc={fmt_sig(d['enc_mean_s'])}±{fmt_sig(d['enc_std_s'])} "
                          f"| PIPEenc={fmt_sig(d['pipe_enc_mean_s'])} | E2E(enc+dec)={fmt_sig(d['e2e_mean_s'])}")

            jr = rec["jpeg"]; print_codec_line("JPEG", jr); print_fec_block("JPEG", jr.get("fec") or {})
            if rec["bpg"]:
                br = rec["bpg"]; print_codec_line("BPG", br); print_fec_block("BPG", br.get("fec") or {})
            else:
                print("  BPG : (不可用)")

        except Exception as e:
            traceback.print_exc()
            warn_box(f"处理失败：{path}\n{e}")

    print("\n[INFO] 处理完成，开始生成汇总报告...")
    export_reports(OUTDIR, records, root_dir)

    # 控制台打印数据集均值与总体（简版）
    def dataset_mean(records, codec, key):
        vals = []
        for r in records:
            c = r.get(codec) or {}
            v = c.get(key)
            if v is not None and math.isfinite(v):
                vals.append(v)
        return statistics.mean(vals) if vals else None

    def overall_summary(records, codec, fec_key=None):
        total_in = 0; total_out = 0
        for r in records:
            total_in += int(r.get("input_bytes") or 0)
            if fec_key is None:
                c = r.get(codec) or {}; total_out += int(c.get("size_bytes") or 0)
            else:
                c = r.get(codec) or {}; f = (c.get("fec") or {}).get(fec_key) or {}
                total_out += int(f.get("fec_bytes") or 0)
        pct = (100.0 * (1 - total_out / total_in)) if total_in>0 else float("nan")
        return total_in, total_out, pct

    for codec in ["jpeg", "bpg"]:
        if all((r.get(codec) is None) for r in records):  # 例如无BPG
            continue
        print(f"\n== 数据集均值 ({codec.upper()}) ==")
        print(f"  平均仅信源输出 ≈ {dataset_mean(records, codec, 'size_kB_dec') or float('nan'):.2f} kB "
              f"| 平均压缩率 ≈ {(dataset_mean(records, codec, 'saving_percent') or float('nan')):.2f}% "
              f"| enc_mean: {fmt_sig(dataset_mean(records, codec, 'enc_mean_s'))} "
              f"| dec_mean: {fmt_sig(dataset_mean(records, codec, 'dec_mean_s'))}")
        tin, tout, tpct = overall_summary(records, codec, None)
        print(f"  总体(加总)仅信源输出 = {tout/1_000:.2f} kB / 输入 {tin/1_000_000:.2f} MB "
              f"| 总体压缩率 = {tpct:.2f}%")

        for fec_key, label in [("polar_r33","Polar(1/3)"), ("polar_r50","Polar(1/2)"),
                               ("ldpc_r67","LDPC(2/3)"), ("ldpc_r75","LDPC(3/4)")]:
            # 平均口径
            def mean_fec(records, subkey):
                vals = []
                for r in records:
                    c = r.get(codec) or {}; f = (c.get("fec") or {}).get(fec_key) or {}
                    v = f.get(subkey)
                    if v is not None and math.isfinite(v):
                        vals.append(v)
                return statistics.mean(vals) if vals else None
            m_size_kB = mean_fec(records, "fec_size_kB_dec")
            m_pct = mean_fec(records, "saving_percent_with_fec")
            m_fe = mean_fec(records, "enc_mean_s")
            m_pe = mean_fec(records, "pipe_enc_mean_s")
            m_ee = mean_fec(records, "e2e_mean_s")
            if m_size_kB is None:  # 说明该编码器可能不存在（比如没跑到BPG）
                continue
            # 总体口径
            tin, tout, tpct = overall_summary(records, codec, fec_key)
            print(f"  {label}: 平均含FEC输出 ≈ {m_size_kB:.2f} kB | 平均压缩率 ≈ {m_pct or float('nan'):.2f}% "
                  f"| FECenc_mean={fmt_sig(m_fe)} | PIPEenc_mean={fmt_sig(m_pe)} | E2E_mean={fmt_sig(m_ee)}")
            print(f"           总体(加总)含FEC输出 = {tout/1_000:.2f} kB / 输入 {tin/1_000_000:.2f} MB "
                  f"| 总体压缩率 = {tpct:.2f}%")

    info_box(f"完成！所有结果已保存到：{os.path.abspath(OUTDIR)}\n"
             f"• 控制台按“原图MB / 编码KB + 压缩率(%)”展示逐图与汇总。\n"
             f"• 查看 BPG：打开 *_decoded.png；或 bpgdec -o out.png file.bpg。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_box(str(e))
        raise
