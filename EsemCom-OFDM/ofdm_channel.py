# ofdm_channel.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn

PI = math.pi


@dataclass
class OFDMParams:
    # OFDM numerology
    n_fft: int = 1024
    n_cp: int = 128
    n_pilot: int = 1  # number of pilot OFDM symbols

    # Multipath channel
    taps: int = 8       # number of time-domain taps (L)
    decay: float = 2.0  # exponential decay factor for power delay profile

    # Equalization
    eq: str = "mmse"    # "mmse" or "zf"

    # Noise calculation mode
    # "ins": noise variance based on instantaneous received signal power
    # "avg": noise variance based on configured tx power
    snr_mode: str = "avg"

    # TX power normalization target (average complex sample power)
    tx_pwr: float = 1.0

    # Clipping (optional, to simulate nonlinearity / reduce PAPR)
    clip: bool = False
    cr: float = 1.5     # clipping ratio

    # Deterministic pilot seed
    pilot_seed: int = 0


def _complex_gaussian(shape, device, dtype=torch.complex64, var: torch.Tensor | float = 1.0):
    """
    CN(0, var) complex Gaussian noise.
    var can be scalar or tensor broadcastable to shape.
    """
    if not torch.is_tensor(var):
        var = torch.tensor(var, device=device, dtype=torch.float32)
    var = var.to(device=device, dtype=torch.float32)
    std = torch.sqrt(var / 2.0)
    n_re = torch.randn(shape, device=device, dtype=torch.float32) * std
    n_im = torch.randn(shape, device=device, dtype=torch.float32) * std
    return torch.complex(n_re, n_im).to(dtype=dtype)


def _normalize_avg_power(x: torch.Tensor, target_pwr: float):
    """
    Normalize average power E[|x|^2] to target_pwr (per complex sample).
    """
    eps = 1e-12
    p = torch.mean(torch.abs(x) ** 2, dim=-1, keepdim=True)  # (..., 1)
    scale = torch.sqrt(torch.tensor(target_pwr, device=x.device, dtype=torch.float32) / (p + eps))
    return x * scale


def _papr(x_td: torch.Tensor) -> torch.Tensor:
    """
    Peak-to-average power ratio for time-domain complex samples.
    x_td: (B, N) complex
    return: (B,)
    """
    power = torch.abs(x_td) ** 2  # (B,N)
    p_avg = power.mean(dim=-1) + 1e-12
    p_peak = power.max(dim=-1).values
    return p_peak / p_avg


def _apply_clipping(x_td: torch.Tensor, cr: float) -> torch.Tensor:
    """
    Soft clipping by amplitude threshold: amp <= CR * sigma
    where sigma = sqrt(E[|x|^2]).
    """
    # sigma per batch
    p = torch.mean(torch.abs(x_td) ** 2, dim=-1, keepdim=True) + 1e-12
    sigma = torch.sqrt(p)
    thr = cr * sigma
    amp = torch.abs(x_td) + 1e-12
    scale = torch.minimum(thr / amp, torch.ones_like(amp))
    return x_td * scale


def _exp_decay_pdp(taps: int, decay: float, device) -> torch.Tensor:
    """
    Exponential PDP normalized to sum=1: p[l] ∝ exp(-l/decay)
    return: (taps,) float
    """
    l = torch.arange(taps, device=device, dtype=torch.float32)
    p = torch.exp(-l / float(decay))
    p = p / (p.sum() + 1e-12)
    return p


def _qpsk_pilot(n_fft: int, seed: int, device) -> torch.Tensor:
    """
    Generate deterministic QPSK pilot in frequency domain (BPSK on I/Q).
    Return: (n_fft,) complex
    """
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    bits_i = torch.randint(0, 2, (n_fft,), generator=g, device=device, dtype=torch.int64)
    bits_q = torch.randint(0, 2, (n_fft,), generator=g, device=device, dtype=torch.int64)
    sym_i = 2.0 * bits_i.float() - 1.0
    sym_q = 2.0 * bits_q.float() - 1.0
    pilot = torch.complex(sym_i, sym_q) / math.sqrt(2.0)
    return pilot.to(torch.complex64)


def _mmse_eq(H: torch.Tensor, Y: torch.Tensor, noise_var: torch.Tensor) -> torch.Tensor:
    """
    MMSE equalization: X = conj(H) / (|H|^2 + N0) * Y
    H: (B, M)
    Y: (B, S, M)
    noise_var: (B,) or (B,1) real
    """
    denom = (torch.abs(H) ** 2) + noise_var.view(-1, 1)  # (B,M)
    W = torch.conj(H) / (denom + 1e-12)                  # (B,M)
    return Y * W.unsqueeze(1)                             # (B,S,M)


def _zf_eq(H: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    ZF equalization: X = Y / H
    """
    return Y / (H.unsqueeze(1) + 1e-12)


class OFDMSystem(nn.Module):
    """
    End-to-end OFDM PHY block (differentiable w.r.t input symbols x_fd):
      x_fd (B,S,M) -> IFFT -> add CP -> serialize -> multipath -> AWGN -> sync -> rm CP -> FFT
      -> pilot channel est -> equalize -> x_hat_fd (B,S,M)

    Returns aux dict for debug:
      papr, noise_var, tx_pwr, rx_pwr, h_est_mse, etc.
    """
    def __init__(self, p: OFDMParams):
        super().__init__()
        self.p = p

    def sample_channel(self, B: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample time-domain taps h_td and frequency response H_fd_true.
        h_td: (B, L) complex
        H_fd_true: (B, M) complex
        """
        pdp = _exp_decay_pdp(self.p.taps, self.p.decay, device=device)  # (L,)
        # CN(0, pdp[l]) per tap
        # variance per complex tap = pdp[l]
        std = torch.sqrt(pdp / 2.0)  # (L,)
        n_re = torch.randn((B, self.p.taps), device=device) * std
        n_im = torch.randn((B, self.p.taps), device=device) * std
        h_td = torch.complex(n_re, n_im).to(torch.complex64)  # (B,L)

        # True channel freq response
        M = self.p.n_fft
        h_pad = torch.zeros((B, M), device=device, dtype=torch.complex64)
        h_pad[:, :self.p.taps] = h_td
        H_fd_true = torch.fft.fft(h_pad, dim=-1, norm="ortho")  # (B,M)
        return h_td, H_fd_true

    def forward(
        self,
        x_fd: torch.Tensor,
        snr_db: float | torch.Tensor,
        h_td: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        x_fd: (B, S, M) complex OFDM data symbols in frequency domain
        snr_db: scalar or (B,)
        h_td: optional fixed channel taps (B,L) complex
        """
        assert torch.is_complex(x_fd), "x_fd must be complex tensor"
        B, S, M = x_fd.shape
        device = x_fd.device
        assert M == self.p.n_fft, f"x_fd last dim must be n_fft={self.p.n_fft}"

        # snr tensor
        if not torch.is_tensor(snr_db):
            snr_db_t = torch.full((B,), float(snr_db), device=device, dtype=torch.float32)
        else:
            snr_db_t = snr_db.to(device=device, dtype=torch.float32).view(-1)
            if snr_db_t.numel() == 1:
                snr_db_t = snr_db_t.repeat(B)
            assert snr_db_t.numel() == B

        # Generate (deterministic) pilot in FD
        pilot_fd_1 = _qpsk_pilot(self.p.n_fft, self.p.pilot_seed, device=device)  # (M,)
        pilot_fd_1 = pilot_fd_1.unsqueeze(0)  # (1,M)
        pilot_fd = pilot_fd_1.repeat(B, 1)    # (B,M)

        # Normalize data symbol power in frequency domain
        x_fd = _normalize_avg_power(x_fd, self.p.tx_pwr)

        # Stack pilot symbols before data
        if self.p.n_pilot > 0:
            pilot_block = pilot_fd.unsqueeze(1).repeat(1, self.p.n_pilot, 1)  # (B, n_pilot, M)
            x_all_fd = torch.cat([pilot_block, x_fd], dim=1)                  # (B, n_pilot+S, M)
        else:
            x_all_fd = x_fd                                                   # (B, S, M)

        S_tot = x_all_fd.shape[1]

        # IFFT to time domain
        x_all_td = torch.fft.ifft(x_all_fd, dim=-1, norm="ortho")  # (B, S_tot, M)

        # Add CP
        if self.p.n_cp > 0:
            cp = x_all_td[:, :, -self.p.n_cp:]                     # (B,S_tot,n_cp)
            x_cp = torch.cat([cp, x_all_td], dim=-1)               # (B,S_tot,M+n_cp)
        else:
            x_cp = x_all_td                                        # (B,S_tot,M)

        # Serialize
        x_serial = x_cp.reshape(B, -1)  # (B, S_tot*(M+n_cp))

        # Optional clipping (time-domain PA nonlinearity)
        if self.p.clip:
            x_serial = _apply_clipping(x_serial, self.p.cr)

        papr = _papr(x_serial)  # (B,)

        # Channel
        if h_td is None:
            h_td, H_fd_true = self.sample_channel(B, device=device)  # (B,L), (B,M)
        else:
            assert h_td.shape[0] == B and h_td.shape[1] == self.p.taps
            h_td = h_td.to(device=device, dtype=torch.complex64)
            # compute true H
            h_pad = torch.zeros((B, M), device=device, dtype=torch.complex64)
            h_pad[:, :self.p.taps] = h_td
            H_fd_true = torch.fft.fft(h_pad, dim=-1, norm="ortho")

        # Linear convolution via FFT
        Lh = self.p.taps
        Nx = x_serial.shape[1]
        Nfft_conv = Nx + Lh - 1
        Xc = torch.fft.fft(x_serial, n=Nfft_conv, dim=-1)
        Hc = torch.fft.fft(h_td, n=Nfft_conv, dim=-1)
        y = torch.fft.ifft(Xc * Hc, dim=-1)  # (B, Nfft_conv) complex

        # Noise power
        with torch.no_grad():
            if self.p.snr_mode == "ins":
                rx_sig_pwr = torch.mean(torch.abs(y) ** 2, dim=-1)  # (B,)
                noise_var = rx_sig_pwr * 10.0 ** (-snr_db_t / 10.0)  # (B,)
            elif self.p.snr_mode == "avg":
                noise_var = torch.full((B,), float(self.p.tx_pwr), device=device) * 10.0 ** (-snr_db_t / 10.0)
            else:
                raise ValueError(f"Unknown snr_mode={self.p.snr_mode}")

        noise = _complex_gaussian(y.shape, device=device, var=noise_var.view(B, 1))
        y_noisy = y + noise

        # Perfect timing: discard first (Lh-1) samples so CP handles ISI
        start = Lh - 1
        total_len = S_tot * (M + self.p.n_cp)
        y_sync = y_noisy[:, start:start + total_len]  # (B, total_len)

        # Reshape to OFDM symbols
        y_sym = y_sync.reshape(B, S_tot, M + self.p.n_cp)

        # Remove CP
        if self.p.n_cp > 0:
            y_td = y_sym[:, :, self.p.n_cp:]  # (B,S_tot,M)
        else:
            y_td = y_sym

        # FFT to FD
        y_fd = torch.fft.fft(y_td, dim=-1, norm="ortho")  # (B,S_tot,M)

        # Pilot-based channel estimation
        if self.p.n_pilot > 0:
            y_pil = y_fd[:, :self.p.n_pilot, :]  # (B,n_pilot,M)
            # average pilots
            y_pil_mean = y_pil.mean(dim=1)       # (B,M)
            x_pil = pilot_fd                     # (B,M)
            H_ls = y_pil_mean / (x_pil + 1e-12)  # (B,M)
            H_est = H_ls
            y_data = y_fd[:, self.p.n_pilot:, :] # (B,S,M)
        else:
            # No pilot: assume perfect H
            H_est = H_fd_true
            y_data = y_fd

        # Equalization
        if self.p.eq == "mmse":
            x_hat_fd = _mmse_eq(H_est, y_data, noise_var)
        elif self.p.eq == "zf":
            x_hat_fd = _zf_eq(H_est, y_data)
        else:
            raise ValueError(f"Unknown eq={self.p.eq}")

        # Debug stats
        tx_pwr = torch.mean(torch.abs(x_serial) ** 2, dim=-1)  # (B,)
        rx_pwr = torch.mean(torch.abs(y_sync) ** 2, dim=-1)    # (B,)

        if self.p.n_pilot > 0:
            h_est_mse = torch.mean(torch.abs(H_est - H_fd_true) ** 2, dim=-1)  # (B,)
        else:
            h_est_mse = torch.zeros((B,), device=device)

        aux = {
            "papr": papr.detach(),
            "noise_var": noise_var.detach(),
            "snr_db": snr_db_t.detach(),
            "tx_pwr": tx_pwr.detach(),
            "rx_pwr": rx_pwr.detach(),
            "h_td": h_td.detach(),
            "h_fd_true": H_fd_true.detach(),
            "h_est_fd": H_est.detach(),
            "h_est_mse": h_est_mse.detach(),
            "sync_start": start,
            "total_len": total_len,
        }

        return x_hat_fd, aux
