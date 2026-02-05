# ofdm_wrapper.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Optional, Tuple, List
import math
import torch
import torch.nn as nn

from ofdm_channel import OFDMParams, OFDMSystem


class OFDMChannelWrapper(nn.Module):
    """
    Wrap OFDMSystem so that it can be used like:
        latent_out = channel(latent_in)

    It:
      (1) packs real latent tensor into complex symbols
      (2) maps to OFDM frequency grid (B,S,n_fft)
      (3) runs OFDMSystem forward with multipath + AWGN + pilot + equalization
      (4) unpacks complex symbols back to real latent tensor (same shape)

    Support "active data subcarriers":
      - n_fft: FFT size (total subcarriers / bins)
      - n_active: number of ACTIVE bins that carry payload (e.g., 24), remaining bins are null/guard/DC
    """

    def __init__(
        self,
        params: OFDMParams,
        snr_db: float = 15.0,
        ofdm_symbols: int = 0,      # 0 => auto symbols to fit payload
        fix_channel: bool = True,   # keep same channel taps across calls (recommended for inversion)
        n_active: int = 0,          # 0 => use all bins as payload (legacy). e.g., 24 for active subcarriers.
        active_idx: Optional[List[int]] = None,  # optional explicit active bin indices (len == n_active)
    ):
        super().__init__()
        self.params = params
        self.snr_db = float(snr_db)
        self.ofdm_symbols = int(ofdm_symbols)
        self.fix_channel = bool(fix_channel)

        # Active-subcarrier control
        self.n_active = int(n_active)
        self.active_idx = active_idx  # if provided, overrides default mapping

        self.ofdm = OFDMSystem(params)

        # cached channel taps for deterministic optimization
        self._fixed_h_td: Optional[torch.Tensor] = None

        # debug state
        self.debug: bool = False
        self.debug_every: int = 50
        self._call_count: int = 0
        self._ran_self_test: bool = False

        # latest aux for external logger
        self.last_aux: Optional[Dict] = None

    def set_debug(self, debug: bool = True, debug_every: int = 50):
        self.debug = bool(debug)
        self.debug_every = int(debug_every)

    def change_snr(self, snr_db: float):
        self.snr_db = float(snr_db)

    def _default_active_idx(self, M: int, n_active: int) -> List[int]:
        """
        Default active-bin mapping: exclude DC (bin 0), take symmetric bins around DC.
        For M=32, n_active=24 => pos: 1..12, neg: 20..31 (wrapped indices).
        """
        if n_active <= 0 or n_active >= M:
            return list(range(M))

        if (n_active % 2) != 0:
            raise ValueError("n_active should be even for symmetric DC-excluded mapping.")

        half = n_active // 2
        # Available positive bins excluding DC: 1..(M/2 - 1)
        max_half = (M // 2) - 1
        if half > max_half:
            raise ValueError(f"n_active too large for DC-excluded mapping: n_active={n_active}, M={M}")

        pos = list(range(1, 1 + half))       # +1 ... +half
        neg = list(range(M - half, M))       # -half ... -1 (wrapped indices)
        return pos + neg

    def _pack_latent_to_fd(self, latent: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        latent: (B, ...) real tensor
        return x_fd: (B, S, n_fft) complex
        meta: for unpack
        """
        assert not torch.is_complex(latent)
        B = latent.shape[0]
        device = latent.device
        M = int(self.params.n_fft)

        # ---- active bins (payload subcarriers) ----
        if self.n_active <= 0:
            n_active = M
        else:
            n_active = int(self.n_active)

        if self.active_idx is not None:
            active_idx = list(self.active_idx)
            if len(active_idx) != n_active:
                raise ValueError(f"active_idx length ({len(active_idx)}) must equal n_active ({n_active})")
            if any((i < 0 or i >= M) for i in active_idx):
                raise ValueError("active_idx contains out-of-range bin index.")
        else:
            active_idx = self._default_active_idx(M, n_active)

        # If active_idx covers all bins, treat as legacy full-grid payload
        if len(active_idx) == M and sorted(active_idx) == list(range(M)):
            active_idx_use: Optional[List[int]] = None
            n_active = M
        else:
            active_idx_use = active_idx

        orig_shape = latent.shape
        flat = latent.reshape(B, -1)  # (B, Nreal)
        n_real = flat.shape[1]

        # make even length for IQ pairing
        if (n_real % 2) == 1:
            flat = torch.cat([flat, torch.zeros((B, 1), device=device, dtype=flat.dtype)], dim=1)
        n_real_even = flat.shape[1]
        n_complex = n_real_even // 2

        I = flat[:, 0::2]
        Q = flat[:, 1::2]
        z = torch.complex(I, Q).to(torch.complex64)  # (B, n_complex)

        # number of OFDM symbols to carry payload (over active bins)
        if self.ofdm_symbols > 0:
            S = int(self.ofdm_symbols)
        else:
            S = int(math.ceil(n_complex / float(n_active)))

        total_slots = S * n_active
        if total_slots < n_complex:
            raise ValueError("Internal error: total_slots < n_complex")

        # pad payload to fill S*n_active
        if total_slots > n_complex:
            pad = torch.zeros((B, total_slots - n_complex), device=device, dtype=torch.complex64)
            z_pad = torch.cat([z, pad], dim=1)
        else:
            z_pad = z

        z_grid = z_pad.reshape(B, S, n_active)  # payload grid over active bins

        # build full FFT grid (B,S,M)
        if active_idx_use is None:
            x_fd = z_grid.reshape(B, S, M)
        else:
            x_fd = torch.zeros((B, S, M), device=device, dtype=torch.complex64)
            x_fd[:, :, active_idx_use] = z_grid

        meta = {
            "orig_shape": orig_shape,
            "n_real": n_real,
            "n_real_even": n_real_even,
            "n_complex": n_complex,
            "S": S,
            "n_active": n_active,
            "active_idx": active_idx_use,  # None => all bins payload
            "n_fft": M,
        }
        return x_fd, meta

    def _unpack_fd_to_latent(self, x_fd: torch.Tensor, meta: Dict) -> torch.Tensor:
        """
        x_fd: (B,S,n_fft) complex
        return latent_hat: (B, ...) real with orig_shape
        """
        B = x_fd.shape[0]
        device = x_fd.device

        n_real = int(meta["n_real"])
        n_real_even = int(meta["n_real_even"])
        n_complex = int(meta["n_complex"])
        n_active = int(meta.get("n_active", x_fd.shape[-1]))
        active_idx = meta.get("active_idx", None)

        # extract payload subcarriers
        if active_idx is None:
            z_all = x_fd.reshape(B, -1)
        else:
            z_all = x_fd[:, :, active_idx].reshape(B, -1)  # (B, S*n_active)

        z = z_all[:, :n_complex]  # remove padding

        out = torch.zeros((B, n_real_even), device=device, dtype=torch.float32)
        out[:, 0::2] = z.real.float()
        out[:, 1::2] = z.imag.float()

        out = out[:, :n_real]
        out = out.reshape(meta["orig_shape"])
        return out

    @torch.no_grad()
    def _self_test_pack_unpack(self, latent: torch.Tensor):
        x_fd, meta = self._pack_latent_to_fd(latent)
        latent_back = self._unpack_fd_to_latent(x_fd, meta)
        err = (latent_back - latent).abs().max().item()
        print(
            f"[OFDM-SELFTEST] pack->unpack max|err|={err:.3e} "
            f"latent_shape={tuple(latent.shape)} S={meta['S']} n_fft={meta['n_fft']} n_active={meta['n_active']}"
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        self._call_count += 1

        # one-time self test
        if self.debug and (not self._ran_self_test):
            self._self_test_pack_unpack(latent)
            self._ran_self_test = True

        x_fd, meta = self._pack_latent_to_fd(latent)

        # fix channel taps for stability in inversion
        if self.fix_channel:
            if (
                self._fixed_h_td is None
                or self._fixed_h_td.shape[0] != latent.shape[0]
                or self._fixed_h_td.device != latent.device
            ):
                h_td, _ = self.ofdm.sample_channel(latent.shape[0], device=latent.device)
                self._fixed_h_td = h_td
            h_td_use = self._fixed_h_td
        else:
            h_td_use = None

        x_hat_fd, aux = self.ofdm(x_fd, snr_db=self.snr_db, h_td=h_td_use)
        self.last_aux = aux

        if self.debug and (self._call_count % self.debug_every == 0):
            papr = aux.get("papr", None)
            noise_var = aux.get("noise_var", None)
            h_mse = aux.get("h_est_mse", None)
            tx_pwr = aux.get("tx_pwr", None)
            rx_pwr = aux.get("rx_pwr", None)

            xmag = torch.abs(x_hat_fd).mean().item()
            msg = f"[OFDM] call={self._call_count} |x_hat_fd|mean={xmag:.3e}"
            if papr is not None:
                msg += f" PAPR(mean)={papr.mean().item():.3f}"
            if noise_var is not None:
                msg += f" noise_var(mean)={noise_var.mean().item():.3e}"
            if h_mse is not None:
                msg += f" h_est_mse(mean)={h_mse.mean().item():.3e}"
            if tx_pwr is not None:
                msg += f" tx_pwr(mean)={tx_pwr.mean().item():.3e}"
            if rx_pwr is not None:
                msg += f" rx_pwr(mean)={rx_pwr.mean().item():.3e}"
            print(msg)

        # NaN/Inf guard
        if (not torch.isfinite(x_hat_fd.real).all()) or (not torch.isfinite(x_hat_fd.imag).all()):
            raise FloatingPointError("[OFDM] x_hat_fd has NaN/Inf. Try MMSE, increase SNR, or reduce lr.")

        latent_hat = self._unpack_fd_to_latent(x_hat_fd, meta)

        # latent guard
        if not torch.isfinite(latent_hat).all():
            raise FloatingPointError("[OFDM] latent_hat has NaN/Inf after unpack.")

        return latent_hat
