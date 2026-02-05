#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_ofdm.py (Slim + Consistent + Active Subcarriers + BMP Random Sampling + Semantic Encode Timing + Focused Logs)
=================================================================================================================
Evaluate:
  1) Semantic OFDM-JSCC (StyleGAN inversion over AWGN/OFDM channel)
  2) Digital baseline (JPEG roundtrip metrics + SNR->MCS -> required OFDM symbols)

Key points (IMPORTANT)
----------------------
- In this codebase, n_pilot means the number of PILOT OFDM SYMBOLS (time-domain overhead),
  NOT "pilot subcarriers".
- n_fft is the FFT size (total bins).
- n_data is the number of ACTIVE DATA subcarriers used to carry payload (<= n_fft).
  If n_data < n_fft, the remaining bins are treated as null/guard/DC (implemented in ofdm_wrapper.py).

Timing (IMPORTANT)
------------------
This version records ONLY semantic "encoding" time:
  - We define semantic encoding time as the inversion optimization time in optimize_latent().
  - It outputs avg_encode_ms_per_image to summary_all.csv (average per image for each run).
  - Digital baseline is NOT timed (left blank), as requested.
  - In debug mode, it also prints per-batch ms/img and running average.

Focused Logging (NEW)
---------------------
When --debug True:
  - optimize_latent() prints loss terms every --debug_every optimization steps (existing behavior).
  - run_one_semantic() prints timing/progress every --log_every_batch batches:
      [DBG][ENC] snr=.. step=.. batch=i/N bsz=.. ms/img=.. avg_ms/img=.. psnr=.. ms-ssim=.. lpips=..

Image IO
--------
- Dataset supports .jpg/.jpeg/.png/.bmp/.webp
- It RANDOMLY SAMPLES max_images from the folder (reproducible via --seed)
- DataLoader shuffle controls per-epoch ordering of the sampled set.

Example command
---------------
python -u main_ofdm.py \
  --dataset ./data/examples \
  --ckpt pretrained/CelebAMask-HQ-512x512.pt \
  --outdir results/ESemCom-OFDM \
  --exp_name ofdm_nfft32_ndata24_cp8_bw1mhz \
  --channel_mode ofdm --snr_list 0-20 \
  --steps 100,200,300,400 --run_semantic True \
  --run_digital True --jpeg_q_list 10,15,20 \
  --debug True --debug_every 20
"""

import os
import io
import csv
import time
import math
import argparse
import random
from copy import deepcopy
from typing import Dict, List, Any, Tuple

import numpy as np
from PIL import Image
from imageio import imwrite

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import piq
from pytorch_msssim import ms_ssim

# --- project imports ---
from models import make_model
from criteria.lpips import lpips

# --- OFDM imports ---
from ofdm_channel import OFDMParams
from ofdm_wrapper import OFDMChannelWrapper


# ---------------------------
# Utilities
# ---------------------------
def parse_boolean(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip()
    return not (s in ["False", "false", "0", "No", "no", "N", "n"])


def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_snr_list(s: str) -> List[int]:
    """
    Accept:
      - "15"
      - "0,1,2,...,20"
      - "0-20"  (inclusive)
    """
    s = (s or "").strip()
    if not s:
        return []
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        a, b = int(a.strip()), int(b.strip())
        if a <= b:
            return list(range(a, b + 1))
        return list(range(a, b - 1, -1))
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def tensor2image(t: torch.Tensor) -> np.ndarray:
    """(B,3,H,W) in [-1,1] -> (B,H,W,3) uint8"""
    imgs = t.detach().cpu().clamp(-1, 1).permute(0, 2, 3, 1).numpy()
    imgs = imgs * 127.5 + 127.5
    return imgs.astype(np.uint8)


def get_transformation(args):
    return transforms.Compose(
        [
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def calc_lpips_loss(im1, im2, percept):
    img1 = F.adaptive_avg_pool2d(im1, (256, 256))
    img2 = F.adaptive_avg_pool2d(im2, (256, 256))
    return percept(img1, img2).mean()


def ofdm_symbol_duration_s(n_fft: int, n_cp: int, bw_hz: float) -> float:
    """Simple duration model: Ts = (n_fft + n_cp) / Fs, with Fs approximated by bw_hz."""
    return (int(n_fft) + int(n_cp)) / max(1.0, float(bw_hz))


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def get_n_data(args) -> int:
    """Active payload subcarriers used per DATA OFDM symbol."""
    nd = int(getattr(args, "n_data", 0))
    if nd <= 0:
        return int(args.n_fft)
    return min(int(args.n_fft), nd)


def cuda_sync_if_needed(device: str):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------
# Incremental summary writer
# ---------------------------
SUMMARY_FIELDS = [
    "timestamp",
    "system",          # semantic / digital
    "codec",           # "" / jpeg
    "channel_mode",    # awgn / ofdm (semantic); ofdm (digital estimation)
    "snr_db",
    "step",            # inversion steps (semantic) / kept for grouping (digital)
    "jpeg_q",

    # distortion
    "avg_psnr",
    "avg_ms_ssim",
    "avg_lpips",

    # OFDM estimate
    "pilot_symbols",                 # params.n_pilot (pilot OFDM symbols, time overhead)
    "data_symbols_avg",              # avg payload symbols (digital avg, semantic avg)
    "total_symbols_avg",             # pilot + data
    "bits_per_data_symbol",          # digital only (payload bits per DATA OFDM symbol)
    "throughput_kbps_payload",       # digital only: payload bits per data symbol / Ts
    "throughput_kbps_effective",     # digital only: payload bits / (total symbols) / Ts

    # timing (semantic only)
    "avg_encode_ms_per_image",       # average semantic encoding time per image (optimize_latent only)

    # bookkeeping
    "batch_size",
    "max_images",
    "seed",
    "run_name",
    "run_dir",

    # OFDM params
    "n_fft",
    "n_data",
    "n_cp",
    "n_pilot",
    "taps",
    "decay",
    "eq",
    "snr_mode",
    "ofdm_tx_pwr",
    "ofdm_symbols",
    "ofdm_fix_channel",
    "ofdm_clip",
    "ofdm_cr",
    "ofdm_pilot_seed",
    "bw_hz",
    "ofdm_symbol_duration_ms",
]


def append_summary_row(csv_path: str, row: Dict[str, Any]):
    ensure_dir(os.path.dirname(csv_path) or ".")
    write_header = not os.path.exists(csv_path)

    out = {k: "" for k in SUMMARY_FIELDS}
    out.update(row)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(out)
        f.flush()


# ---------------------------
# Core modules
# ---------------------------
class PowerNormalize(nn.Module):
    def __init__(self, t_pow=1.0):
        super().__init__()
        self.t_pow = float(t_pow)

    def forward(self, x, dim=(1, 2)):
        pwr = torch.mean(x ** 2, dim=dim, keepdim=True) + 1e-12
        return math.sqrt(self.t_pow) * x / torch.sqrt(pwr)


class AWGN_Channel(nn.Module):
    def __init__(self, snr_db):
        super().__init__()
        self.change_snr(snr_db)

    def change_snr(self, snr_db):
        self.std = 10 ** (-0.05 * float(snr_db))

    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(1, self.count)

    def update(self, val, n=1):
        self.sum += float(val) * int(n)
        self.count += int(n)


class ImageDataset:
    def __init__(self, data_dir, transform=None, max_images=100, seed=0, recursive=False):
        """
        data_dir: 图片根目录
        max_images: 最多抽多少张图片来做测试（会随机抽样）
        seed: 控制“抽样”的随机性（保证可复现）
        recursive: 是否递归搜索子目录
        """
        self.data_dir = data_dir
        self.transform = transform

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        def collect_files(root: str):
            paths = []
            if recursive:
                for r, _, fs in os.walk(root):
                    for f in fs:
                        if f.lower().endswith(exts):
                            paths.append(os.path.join(r, f))
            else:
                fs = os.listdir(root)
                for f in fs:
                    if f.lower().endswith(exts):
                        paths.append(os.path.join(root, f))
            return paths

        files = collect_files(data_dir)

        # Random sampling (reproducible)
        rng = random.Random(int(seed))
        rng.shuffle(files)
        if max_images is not None and int(max_images) > 0:
            files = files[: int(max_images)]

        self.image_paths = files

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, p


def make_channel(args, device):
    if args.channel_mode == "awgn":
        return AWGN_Channel(snr_db=args.snr_db).to(device)

    p = OFDMParams(
        n_fft=args.n_fft,
        n_cp=args.n_cp,
        n_pilot=args.n_pilot,       # pilot OFDM symbols (time overhead)
        taps=args.taps,
        decay=args.decay,
        eq=args.eq,
        snr_mode=args.snr_mode,
        tx_pwr=args.ofdm_tx_pwr,
        clip=args.ofdm_clip,
        cr=args.ofdm_cr,
        pilot_seed=args.ofdm_pilot_seed,
    )

    n_data = get_n_data(args)
    ch = OFDMChannelWrapper(
        params=p,
        snr_db=float(args.snr_db),
        ofdm_symbols=int(args.ofdm_symbols),
        fix_channel=bool(args.ofdm_fix_channel),
        n_active=int(n_data),  # payload mapped onto active bins only
    ).to(device)

    if hasattr(ch, "set_debug"):
        ch.set_debug(bool(args.debug), int(args.debug_every))
    return ch


# ---------------------------
# Digital baseline (JPEG) with cache
# (kept intact; timing NOT recorded as requested)
# ---------------------------
MCS_TABLE = [
    (0, "BPSK",    1/2,  2),
    (1, "QPSK",    1/2,  5),
    (2, "QPSK",    3/4,  9),
    (3, "16QAM",   1/2, 11),
    (4, "16QAM",   3/4, 15),
    (5, "64QAM",   2/3, 18),
    (6, "64QAM",   3/4, 20),
    (7, "64QAM",   5/6, 25),
    (8, "256QAM",  3/4, 29),
    (9, "256QAM",  5/6, 31),
]

MOD_BITS = {"BPSK": 1, "QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8}


def select_mcs_by_snr(snr_db: float) -> Tuple[int, str, float, float]:
    cand = [row for row in MCS_TABLE if snr_db >= row[3]]
    if not cand:
        return MCS_TABLE[0]
    return cand[-1]


def jpeg_roundtrip_tensor(
    x_norm_m11: torch.Tensor,
    jpeg_q: int,
    size: int,
    subsampling: int = 2,
    optimize: bool = False,
    progressive: bool = False,
):
    b = x_norm_m11.shape[0]
    imgs_u8 = tensor2image(x_norm_m11)
    outs = []
    byte_lens = []

    for i in range(b):
        pil = Image.fromarray(imgs_u8[i]).convert("RGB").resize((size, size))
        buf = io.BytesIO()
        save_kwargs = dict(format="JPEG", quality=int(jpeg_q),
                           optimize=bool(optimize), progressive=bool(progressive))
        try:
            save_kwargs["subsampling"] = int(subsampling)
            pil.save(buf, **save_kwargs)
        except TypeError:
            save_kwargs.pop("subsampling", None)
            save_kwargs.pop("optimize", None)
            save_kwargs.pop("progressive", None)
            pil.save(buf, **save_kwargs)

        data = buf.getvalue()
        byte_lens.append(len(data))

        buf2 = io.BytesIO(data)
        pil2 = Image.open(buf2).convert("RGB").resize((size, size))
        t = transforms.ToTensor()(pil2)
        t = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(t)
        outs.append(t)

    return torch.stack(outs, dim=0), byte_lens


def build_jpeg_cache(args, device, percept) -> Dict[int, Dict[str, Any]]:
    qs = parse_int_list(args.jpeg_q_list)
    if not qs:
        qs = [10, 40, 70]

    transform = get_transformation(args)
    ds = ImageDataset(args.dataset, transform=transform, max_images=args.max_images, seed=args.seed, recursive=False)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    shuffle=bool(args.shuffle), drop_last=False)

    cache: Dict[int, Dict[str, Any]] = {}

    for q in qs:
        meter_psnr = AverageMeter()
        meter_ssim = AverageMeter()
        meter_lpips = AverageMeter()
        all_bytes: List[int] = []

        for images, _paths in dl:
            images = images.to(device)
            with torch.no_grad():
                rec, byte_lens = jpeg_roundtrip_tensor(
                    images, q, args.size,
                    subsampling=args.jpeg_subsampling,
                    optimize=args.jpeg_optimize,
                    progressive=args.jpeg_progressive,
                )
                rec = rec.to(device)

                lp = calc_lpips_loss(rec, images, percept)
                y = rec.clamp(-1, 1) * 0.5 + 0.5
                x = images * 0.5 + 0.5
                ps = piq.psnr(x, y)
                ss = ms_ssim(x, y, data_range=1)

                meter_lpips.update(lp.item(), images.size(0))
                meter_psnr.update(ps.item(), images.size(0))
                meter_ssim.update(ss.item(), images.size(0))

                all_bytes.extend([int(b) for b in byte_lens])

        cache[int(q)] = {
            "avg_psnr": float(meter_psnr.avg),
            "avg_ms_ssim": float(meter_ssim.avg),
            "avg_lpips": float(meter_lpips.avg),
            "bytes_list": all_bytes,
            "bytes_avg": float(np.mean(all_bytes)) if all_bytes else 0.0,
        }

    return cache


def estimate_digital_symbols_total(bits: int, n_data: int, modulation: str, code_rate: float, pilot_symbols: int) -> Tuple[int, float]:
    payload_bits_per_data_symbol = float(int(n_data) * MOD_BITS[modulation] * float(code_rate))
    data_symbols = int(math.ceil(bits / max(1e-9, payload_bits_per_data_symbol)))
    total_symbols = int(pilot_symbols) + data_symbols
    return total_symbols, payload_bits_per_data_symbol


def digital_rows_for_snr(args, jpeg_cache: Dict[int, Dict[str, Any]], snr_db: int, step_for_grouping: int, exp_root: str) -> List[Dict[str, Any]]:
    mcs, mod, cr, _min_snr = select_mcs_by_snr(float(snr_db))

    ts = ofdm_symbol_duration_s(args.n_fft, args.n_cp, args.bw_hz)
    ts_ms = ts * 1e3

    n_data = get_n_data(args)
    pilot_symbols = int(args.n_pilot)

    rows = []
    for q, info in jpeg_cache.items():
        bytes_list = info["bytes_list"]
        if not bytes_list:
            avg_total_symbols = 0.0
            payload_bps = 0.0
        else:
            total_syms = []
            payload_bits_per_data_symbol = float(n_data * MOD_BITS[mod] * float(cr))
            for bl in bytes_list:
                bits = int(bl) * 8
                total_sym, _pbs = estimate_digital_symbols_total(bits, n_data, mod, cr, pilot_symbols)
                total_syms.append(total_sym)
            avg_total_symbols = float(np.mean(total_syms))
            payload_bps = payload_bits_per_data_symbol

        avg_bits = float(info["bytes_avg"]) * 8.0
        eff_bits_per_symbol = (avg_bits / max(1e-9, avg_total_symbols)) if avg_total_symbols > 0 else 0.0
        throughput_payload_kbps = (payload_bps / ts) / 1e3 if ts > 0 else 0.0
        throughput_effective_kbps = (eff_bits_per_symbol / ts) / 1e3 if ts > 0 else 0.0

        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system": "digital",
            "codec": "jpeg",
            "channel_mode": "ofdm",
            "snr_db": int(snr_db),
            "step": int(step_for_grouping),
            "jpeg_q": int(q),

            "avg_psnr": float(info["avg_psnr"]),
            "avg_ms_ssim": float(info["avg_ms_ssim"]),
            "avg_lpips": float(info["avg_lpips"]),

            "pilot_symbols": int(pilot_symbols),
            "data_symbols_avg": float(max(0.0, avg_total_symbols - pilot_symbols)),
            "total_symbols_avg": float(avg_total_symbols),

            "bits_per_data_symbol": float(payload_bps),
            "throughput_kbps_payload": float(throughput_payload_kbps),
            "throughput_kbps_effective": float(throughput_effective_kbps),

            # timing field left blank for digital (as requested)
            "avg_encode_ms_per_image": "",

            "batch_size": int(args.batch_size),
            "max_images": int(args.max_images),
            "seed": int(args.seed),
            "run_name": f"digital_ofdm_snr{snr_db}dB_jpegq{q}_mcs{mcs}",
            "run_dir": "",

            "n_fft": int(args.n_fft),
            "n_data": int(n_data),
            "n_cp": int(args.n_cp),
            "n_pilot": int(args.n_pilot),
            "taps": int(args.taps),
            "decay": float(args.decay),
            "eq": str(args.eq),
            "snr_mode": str(args.snr_mode),
            "ofdm_tx_pwr": float(args.ofdm_tx_pwr),
            "ofdm_symbols": int(args.ofdm_symbols),
            "ofdm_fix_channel": int(bool(args.ofdm_fix_channel)),
            "ofdm_clip": int(bool(args.ofdm_clip)),
            "ofdm_cr": float(args.ofdm_cr),
            "ofdm_pilot_seed": int(args.ofdm_pilot_seed),
            "bw_hz": float(args.bw_hz),
            "ofdm_symbol_duration_ms": float(ts_ms),
        }
        rows.append(row)

    return rows


# ---------------------------
# Semantic (StyleGAN inversion)
# ---------------------------
def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1.0, (1.0 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1.0, t / rampup)
    return initial_lr * lr_ramp


def prepare_latent_init(g_ema, batch_size: int, device: str, w_plus: bool, mean_samples: int):
    with torch.no_grad():
        n = int(mean_samples)
        n = max(256, min(20000, n))
        noise_sample = torch.randn(n, 512, device=device)
        latent_mean = g_ema.style(noise_sample).mean(0)  # (512,)
        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(batch_size, 1)  # (B,512)
        if w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)  # (B,n_latent,512)
    return latent_in, latent_mean


def estimate_semantic_symbols_total(
    latent_tx: torch.Tensor,
    n_data: int,
    pilot_symbols: int,
    forced_data_symbols: int = 0
) -> int:
    b = int(latent_tx.shape[0])
    real_per_sample = int(latent_tx.numel() // max(1, b))
    n_cplx = int(math.ceil(real_per_sample / 2.0))

    if int(forced_data_symbols) > 0:
        data_symbols = int(forced_data_symbols)
    else:
        data_symbols = int(math.ceil(n_cplx / max(1, int(n_data))))

    return int(pilot_symbols) + data_symbols


def optimize_latent(args, g_ema, target_img, latent_in, latent_mean, p_norm, channel, percept, device):
    noises = g_ema.render_net.get_noise(noise=None, randomize_noise=False)
    for nz in noises:
        nz.requires_grad = False

    latent = latent_in.detach().clone()
    latent.requires_grad = True

    if args.no_noises:
        optimizer = optim.Adam([latent], lr=args.lr)
    else:
        optimizer = optim.Adam([latent] + list(noises), lr=args.lr)

    for i in range(int(args.step)):
        optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]["lr"] = get_lr(float(i) / max(1, int(args.step)), args.lr)

        latent_tx = p_norm(latent, dim=(1, 2)) if args.w_plus else p_norm(latent, dim=(1,))
        latent_rx = channel(latent_tx)

        img_gen, _ = g_ema([latent_rx], input_is_latent=True, randomize_noise=False, noise=None)

        p_loss = calc_lpips_loss(img_gen, target_img, percept)
        l2_loss = F.mse_loss(img_gen, target_img)
        ssim_loss = 1 - ms_ssim(
            img_gen.clamp(-1, 1) * 0.5 + 0.5,
            target_img * 0.5 + 0.5,
            data_range=1,
            size_average=True,
        )

        if args.w_plus:
            latent_mean_loss = F.mse_loss(
                latent, latent_mean.unsqueeze(0).repeat(latent.size(0), g_ema.n_latent, 1)
            )
        else:
            latent_mean_loss = F.mse_loss(latent, latent_mean.repeat(latent.size(0), 1))

        loss = (
            p_loss * args.lambda_lpips
            + ssim_loss * args.lambda_ssim
            + l2_loss * args.lambda_l1
            + latent_mean_loss * args.lambda_mean
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Loss became NaN/Inf at step {i}.")

        loss.backward()
        optimizer.step()

        # existing debug (loss terms)
        if args.debug and (i % int(args.debug_every) == 0):
            print(
                f"[DBG][LOSS] step={i}/{args.step} loss={loss.item():.4f} "
                f"lpips={p_loss.item():.4f} l2={l2_loss.item():.4f} ssimloss={ssim_loss.item():.4f}"
            )

    return latent.detach()


def build_run_name(args):
    n_data = get_n_data(args)
    if args.channel_mode == "awgn":
        return f"semantic_awgn_snr{args.snr_db}dB_step{args.step}"
    return (
        f"semantic_ofdm_snr{args.snr_db}dB_step{args.step}"
        f"_nfft{args.n_fft}_ndata{n_data}_cp{args.n_cp}_taps{args.taps}"
        f"_eq{args.eq}_snrmode{args.snr_mode}"
        f"_fixch{int(bool(args.ofdm_fix_channel))}"
        f"_pil{int(args.n_pilot)}"
        f"_S{int(args.ofdm_symbols)}"
    )


def run_one_semantic(args, device, g_ema, percept, exp_root: str) -> Dict[str, Any]:
    set_all_seeds(int(args.seed))

    run_name = build_run_name(args)
    run_dir = os.path.join(exp_root, run_name)

    if args.overwrite and os.path.exists(run_dir):
        import shutil
        shutil.rmtree(run_dir)

    ensure_dir(run_dir)
    if args.save_recon:
        ensure_dir(os.path.join(run_dir, "recon"))

    transform = get_transformation(args)
    ds = ImageDataset(args.dataset, transform=transform, max_images=args.max_images, seed=args.seed, recursive=False)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    shuffle=bool(args.shuffle), drop_last=False)

    if args.debug:
        print(
            f"[RUN][SEMANTIC] snr={int(args.snr_db)} step={int(args.step)} "
            f"images={len(ds)} batch={int(args.batch_size)} iters={len(dl)} "
            f"n_fft={int(args.n_fft)} n_data={get_n_data(args)}"
        )

    p_norm = PowerNormalize(t_pow=1.0).to(device)
    channel = make_channel(args, device)

    meter_psnr = AverageMeter()
    meter_ssim = AverageMeter()
    meter_lpips = AverageMeter()

    # semantic encoding time meter (optimize_latent only)
    meter_encode_ms = AverageMeter()

    total_symbols_list: List[int] = []
    pilot_symbols = int(args.n_pilot)
    n_data = get_n_data(args)

    ts = ofdm_symbol_duration_s(args.n_fft, args.n_cp, args.bw_hz)
    ts_ms = ts * 1e3

    num_batches = len(dl)
    batch_idx = 0
    log_every = max(1, int(getattr(args, "log_every_batch", 1)))
    log_quality = bool(getattr(args, "log_quality", True))

    for images, paths in dl:
        batch_idx += 1

        images = images.to(device)
        target = images
        bsz = int(images.size(0))

        latent_in, latent_mean = prepare_latent_init(
            g_ema,
            batch_size=bsz,
            device=device,
            w_plus=bool(args.w_plus),
            mean_samples=int(args.latent_mean_samples),
        )

        # --------- measure semantic encoding time (optimization only) ---------
        cuda_sync_if_needed(device)
        t0 = time.perf_counter()
        latent = optimize_latent(args, g_ema, target, latent_in, latent_mean, p_norm, channel, percept, device)
        cuda_sync_if_needed(device)
        t1 = time.perf_counter()
        encode_ms_per_img = (t1 - t0) * 1e3 / max(1, bsz)
        meter_encode_ms.update(encode_ms_per_img, n=bsz)
        # ---------------------------------------------------------------------

        with torch.no_grad():
            latent_tx = p_norm(latent, dim=(1, 2)) if args.w_plus else p_norm(latent, dim=(1,))

            if args.channel_mode == "ofdm":
                tot_sym = estimate_semantic_symbols_total(
                    latent_tx,
                    n_data=n_data,
                    pilot_symbols=pilot_symbols,
                    forced_data_symbols=int(args.ofdm_symbols),
                )
                total_symbols_list.append(int(tot_sym))

            latent_rx = channel(latent_tx)
            img_gen, _ = g_ema([latent_rx], input_is_latent=True, randomize_noise=False, noise=None)

            lp = calc_lpips_loss(img_gen, target, percept)
            y = img_gen.clamp(-1, 1) * 0.5 + 0.5
            x = target * 0.5 + 0.5
            ps = piq.psnr(x, y)
            ss = ms_ssim(x, y, data_range=1)

            meter_lpips.update(lp.item(), bsz)
            meter_psnr.update(ps.item(), bsz)
            meter_ssim.update(ss.item(), bsz)

            if args.save_recon:
                imgs = tensor2image(img_gen)
                for i in range(imgs.shape[0]):
                    imwrite(os.path.join(run_dir, "recon", os.path.basename(paths[i])), imgs[i])

        # --------- NEW: focused debug print for timing/progress/avg ---------
        if args.debug and ((batch_idx % log_every == 0) or (batch_idx == num_batches)):
            if log_quality:
                q_str = f" psnr={meter_psnr.avg:.2f} ms-ssim={meter_ssim.avg:.4f} lpips={meter_lpips.avg:.4f}"
            else:
                q_str = ""
            print(
                f"[DBG][ENC] snr={int(args.snr_db)} step={int(args.step)} "
                f"batch={batch_idx}/{num_batches} bsz={bsz} "
                f"ms/img={encode_ms_per_img:.2f} avg_ms/img={meter_encode_ms.avg:.2f}"
                f"{q_str}"
            )
        # -------------------------------------------------------------------

    avg_total_symbols = float(np.mean(total_symbols_list)) if total_symbols_list else 0.0

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": "semantic",
        "codec": "",
        "channel_mode": str(args.channel_mode),
        "snr_db": int(args.snr_db),
        "step": int(args.step),
        "jpeg_q": "",

        "avg_psnr": float(meter_psnr.avg),
        "avg_ms_ssim": float(meter_ssim.avg),
        "avg_lpips": float(meter_lpips.avg),

        "pilot_symbols": int(pilot_symbols),
        "data_symbols_avg": float(max(0.0, avg_total_symbols - pilot_symbols)),
        "total_symbols_avg": float(avg_total_symbols),

        "bits_per_data_symbol": "",
        "throughput_kbps_payload": "",
        "throughput_kbps_effective": "",

        # semantic encoding time (ms/image)
        "avg_encode_ms_per_image": float(meter_encode_ms.avg),

        "batch_size": int(args.batch_size),
        "max_images": int(args.max_images),
        "seed": int(args.seed),
        "run_name": run_name,
        "run_dir": run_dir,

        "n_fft": int(args.n_fft),
        "n_data": int(n_data),
        "n_cp": int(args.n_cp),
        "n_pilot": int(args.n_pilot),
        "taps": int(args.taps),
        "decay": float(args.decay),
        "eq": str(args.eq),
        "snr_mode": str(args.snr_mode),
        "ofdm_tx_pwr": float(args.ofdm_tx_pwr),
        "ofdm_symbols": int(args.ofdm_symbols),
        "ofdm_fix_channel": int(bool(args.ofdm_fix_channel)),
        "ofdm_clip": int(bool(args.ofdm_clip)),
        "ofdm_cr": float(args.ofdm_cr),
        "ofdm_pilot_seed": int(args.ofdm_pilot_seed),
        "bw_hz": float(args.bw_hz),
        "ofdm_symbol_duration_ms": float(ts_ms),
    }
    return row


# ---------------------------
# Main
# ---------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    p = argparse.ArgumentParser()

    # paths
    p.add_argument("--ckpt", type=str, default="pretrained/CelebAMask-HQ-512x512.pt")
    p.add_argument("--outdir", type=str, default="results/inversion_compare")
    p.add_argument("--exp_name", type=str, default="", help="Experiment folder name under outdir")
    p.add_argument("--dataset", type=str, default="./data/examples")

    # sweep
    p.add_argument("--snr_list", type=str, default="0-20")
    p.add_argument("--steps", type=str, default="200,600")
    p.add_argument("--run_semantic", type=parse_boolean, default=True)
    p.add_argument("--run_digital", type=parse_boolean, default=True)

    # image / loader
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_images", type=int, default=32)
    p.add_argument("--shuffle", type=parse_boolean, default=False)
    p.add_argument("--seed", type=int, default=42)

    # semantic optimization
    p.add_argument("--channel_mode", type=str, default="ofdm", choices=["awgn", "ofdm"])
    p.add_argument("--no_noises", type=parse_boolean, default=True)
    p.add_argument("--w_plus", type=parse_boolean, default=True)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--lambda_l1", type=float, default=0.3)
    p.add_argument("--lambda_lpips", type=float, default=1.0)
    p.add_argument("--lambda_ssim", type=float, default=0.0)
    p.add_argument("--lambda_mean", type=float, default=0.0)
    p.add_argument("--latent_mean_samples", type=int, default=4096)
    p.add_argument("--save_recon", type=parse_boolean, default=True)

    # OFDM params
    p.add_argument("--n_fft", type=int, default=32)
    p.add_argument("--n_cp", type=int, default=8)
    p.add_argument("--n_pilot", type=int, default=1)
    p.add_argument("--n_data", type=int, default=24, help="ACTIVE DATA subcarriers used for payload (<= n_fft). Set 0 to use all n_fft.")
    p.add_argument("--taps", type=int, default=8)
    p.add_argument("--decay", type=float, default=2.0)
    p.add_argument("--eq", type=str, default="mmse", choices=["mmse", "zf"])
    p.add_argument("--snr_mode", type=str, default="avg", choices=["avg", "ins"])
    p.add_argument("--ofdm_tx_pwr", type=float, default=1.0)
    p.add_argument("--ofdm_symbols", type=int, default=0)
    p.add_argument("--ofdm_fix_channel", type=parse_boolean, default=True)
    p.add_argument("--ofdm_clip", type=parse_boolean, default=False)
    p.add_argument("--ofdm_cr", type=float, default=1.5)
    p.add_argument("--ofdm_pilot_seed", type=int, default=0)
    p.add_argument("--bw_hz", type=float, default=1e6)

    # digital jpeg
    p.add_argument("--jpeg_q_list", type=str, default="10,40,70")
    p.add_argument("--jpeg_subsampling", type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--jpeg_optimize", type=parse_boolean, default=False)
    p.add_argument("--jpeg_progressive", type=parse_boolean, default=False)

    # misc
    p.add_argument("--overwrite", type=parse_boolean, default=False)
    p.add_argument("--debug", type=parse_boolean, default=False)
    p.add_argument("--debug_every", type=int, default=50)

    # NEW: focused semantic timing logs
    p.add_argument("--log_every_batch", type=int, default=1,
                   help="Print semantic timing/progress every N batches (1=every batch).")
    p.add_argument("--log_quality", type=parse_boolean, default=True,
                   help="If True, print running avg PSNR/MS-SSIM/LPIPS in semantic logs.")

    args = p.parse_args()

    # sanitize n_data
    if int(args.n_data) < 0:
        args.n_data = 0
    if int(args.n_data) > 0 and int(args.n_data) > int(args.n_fft):
        print(f"[WARN] n_data({args.n_data}) > n_fft({args.n_fft}), clamping to n_fft.")
        args.n_data = int(args.n_fft)

    if not args.exp_name.strip():
        args.exp_name = time.strftime("exp_%m%d_%H%M%S", time.localtime())

    exp_root = os.path.join(args.outdir, args.exp_name)
    ensure_dir(exp_root)

    set_all_seeds(int(args.seed))

    snrs = parse_snr_list(args.snr_list)
    if not snrs:
        snrs = [0]

    steps = parse_int_list(args.steps)
    if not steps:
        steps = [100, 200, 300, 400]

    summary_csv = os.path.join(exp_root, "summary_all.csv")

    # percept model shared
    percept = lpips.LPIPS(net_type="vgg").to(device)

    # digital cache once
    jpeg_cache = None
    if bool(args.run_digital):
        print(f"[DIGITAL] Building JPEG cache for q={args.jpeg_q_list} ...")
        jpeg_cache = build_jpeg_cache(args, device, percept)
        print("[DIGITAL] JPEG cache ready. (computed once; reused for all SNRs/steps)")

    # semantic model load once
    g_ema = None
    if bool(args.run_semantic):
        print(f"[SEMANTIC] Loading ckpt: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        g_ema = make_model(ckpt["args"]).to(device).eval()
        g_ema.load_state_dict(ckpt["g_ema"])
        print(f"[SEMANTIC] Model loaded on {device}")

    # main sweep
    for st in steps:
        for snr in snrs:
            if bool(args.run_semantic):
                a = deepcopy(args)
                a.step = int(st)
                a.snr_db = int(snr)

                print(
                    f"[RUN][SEMANTIC] step={a.step} snr={a.snr_db} mode={a.channel_mode} "
                    f"n_fft={a.n_fft} n_data={get_n_data(a)}"
                )
                row = run_one_semantic(a, device, g_ema, percept, exp_root)
                append_summary_row(summary_csv, row)
                print(
                    f"[OK][SEMANTIC] avg_encode_ms_per_image={row['avg_encode_ms_per_image']:.2f} ms "
                    f"-> summary_all.csv"
                )

            if bool(args.run_digital):
                assert jpeg_cache is not None
                print(
                    f"[RUN][DIGITAL] step={st} snr={int(snr)} jpeg_q={args.jpeg_q_list} "
                    f"n_fft={args.n_fft} n_data={get_n_data(args)}"
                )
                drows = digital_rows_for_snr(args, jpeg_cache, snr_db=int(snr), step_for_grouping=int(st), exp_root=exp_root)
                for r in drows:
                    append_summary_row(summary_csv, r)
                print(f"[OK][DIGITAL] appended {len(drows)} rows -> summary_all.csv")

    print(f"[DONE] Summary: {summary_csv}")


if __name__ == "__main__":
    main()
