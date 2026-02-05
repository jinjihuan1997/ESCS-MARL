import SoapySDR
from SoapySDR import *
import numpy as np
import torch
import torch.nn as nn
import time


def safe_set(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return True
    except Exception:
        return False


def make_barker13():
    b = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float32)
    return b.astype(np.complex64)


def build_preamble(repeats=64):
    return np.tile(make_barker13(), repeats).astype(np.complex64)


def ls_snr_estimate(r_pre, s_pre, eps=1e-12):
    h_hat = np.vdot(s_pre, r_pre) / (np.vdot(s_pre, s_pre) + eps)
    sig = h_hat * s_pre
    e = r_pre - sig
    snr_lin = (np.mean(np.abs(sig) ** 2) + eps) / (np.mean(np.abs(e) ** 2) + eps)
    snr_db = 10.0 * np.log10(snr_lin)
    return float(snr_db), h_hat


def estimate_cfo_from_repeated_preamble(r_pre, fs, barker_len=13, repeats=64):
    L = barker_len
    nblocks = repeats
    if len(r_pre) < L * nblocks:
        return 0.0
    r = r_pre[:L * nblocks].reshape(nblocks, L)
    phs = []
    for k in range(nblocks - 1):
        v = np.vdot(r[k], r[k + 1])
        phs.append(np.angle(v + 1e-12))
    dphi = np.median(phs)
    dt = L / fs
    cfo_hz = dphi / (2 * np.pi * dt)
    return float(cfo_hz)


def apply_cfo_correction(x, fs, cfo_hz):
    n = np.arange(len(x), dtype=np.float32)
    rot = np.exp(-1j * 2 * np.pi * cfo_hz * n / fs).astype(np.complex64)
    return x * rot


class SDRChannel(nn.Module):
    """
    不可微物理信道：TX(USRP) -> 空口/线缆 -> RX(Lime)
    forward() 输入 latent tensor，输出接收后的 tensor（同形状）
    同时内部更新：
      self.last_snr_db / self.last_cfo_hz / self.last_sync_peak
    """

    def __init__(
        self,
        tx_serial="30B584E",
        rx_serial="00090726074D281F",
        freq=920e6,
        samp_rate=1e6,
        bw=1e6,
        tx_gain=30,
        rx_gain=40,
        rx_antenna="LNAL",
        tx_antenna=None,
        preamble_repeats=64,
        target_rms=0.2,
        rx_timeout_s=0.2,
        rx_buff_size=120000,
    ):
        super().__init__()
        self.freq = float(freq)
        self.fs = float(samp_rate)
        self.bw = float(bw)
        self.tx_gain = float(tx_gain)
        self.rx_gain = float(rx_gain)
        self.rx_antenna = str(rx_antenna)
        self.tx_antenna = tx_antenna
        self.target_rms = float(target_rms)
        self.timeoutUs = int(rx_timeout_s * 1e6)
        self.rx_buff_size = int(rx_buff_size)

        self.preamble_repeats = int(preamble_repeats)
        self.preamble = build_preamble(self.preamble_repeats)
        self.preamble_len = len(self.preamble)

        self.last_snr_db = None
        self.last_cfo_hz = None
        self.last_sync_peak = None

        # ---- TX: USRP ----
        TX_ARGS = f"driver=uhd,serial={tx_serial}"
        self.tx = SoapySDR.Device(TX_ARGS)
        safe_set(self.tx.setSampleRate, SOAPY_SDR_TX, 0, self.fs)
        safe_set(self.tx.setFrequency, SOAPY_SDR_TX, 0, self.freq)
        safe_set(self.tx.setGain, SOAPY_SDR_TX, 0, self.tx_gain)
        safe_set(self.tx.setBandwidth, SOAPY_SDR_TX, 0, self.bw)
        if self.tx_antenna:
            safe_set(self.tx.setAntenna, SOAPY_SDR_TX, 0, self.tx_antenna)
        self.tx_stream = self.tx.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32)
        self.tx.activateStream(self.tx_stream)

        # ---- RX: Lime ----
        RX_ARGS = f"driver=lime,serial={rx_serial}"
        self.rx = SoapySDR.Device(RX_ARGS)
        safe_set(self.rx.setSampleRate, SOAPY_SDR_RX, 0, self.fs)
        safe_set(self.rx.setFrequency, SOAPY_SDR_RX, 0, self.freq)
        safe_set(self.rx.setGain, SOAPY_SDR_RX, 0, self.rx_gain)
        safe_set(self.rx.setBandwidth, SOAPY_SDR_RX, 0, self.bw)
        safe_set(self.rx.setAntenna, SOAPY_SDR_RX, 0, self.rx_antenna)
        self.rx_stream = self.rx.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self.rx.activateStream(self.rx_stream)

        self.rx_buff = np.zeros(self.rx_buff_size, np.complex64)

        # flush
        junk = np.zeros(8192, np.complex64)
        self.rx.readStream(self.rx_stream, [junk], len(junk), timeoutUs=int(2e5))

    def close(self):
        try:
            self.tx.deactivateStream(self.tx_stream)
            self.tx.closeStream(self.tx_stream)
        except Exception:
            pass
        try:
            self.rx.deactivateStream(self.rx_stream)
            self.rx.closeStream(self.rx_stream)
        except Exception:
            pass

    @torch.no_grad()
    def forward(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        # 限制 batch=1（你当前流程也是 batch=1）
        if latent_tensor.size(0) != 1:
            raise RuntimeError("SDRChannel currently supports batch_size=1 for physical TX/RX.")

        original_shape = latent_tensor.shape

        # ---- 1) flatten -> complex payload ----
        data = latent_tensor.detach().float().cpu().numpy().reshape(-1)

        if len(data) % 2 != 0:
            data = np.append(data, 0.0)

        payload = (data[0::2] + 1j * data[1::2]).astype(np.complex64)

        # ---- 2) 控制发射 RMS（比 max 归一化更稳定）----
        rms = np.sqrt(np.mean(np.abs(payload) ** 2) + 1e-12)
        if rms > 0:
            payload = payload * (self.target_rms / rms)

        # ---- 3) 组帧 ----
        silence = np.zeros(4000, np.complex64)
        tx_frame = np.concatenate([silence, self.preamble, payload, silence]).astype(np.complex64)

        # ---- 4) 发射（可发两次提高命中概率）----
        for _ in range(2):
            self.tx.writeStream(self.tx_stream, [tx_frame], len(tx_frame))
            time.sleep(0.005)

        # ---- 5) 接收 ----
        # 先读一点丢掉（flush）
        self.rx.readStream(self.rx_stream, [self.rx_buff], 1024, timeoutUs=2000)

        sr = self.rx.readStream(self.rx_stream, [self.rx_buff], self.rx_buff_size, timeoutUs=self.timeoutUs)
        if sr.ret <= 0:
            # 超时/失败：返回小噪声，防止主程序崩
            return torch.randn_like(latent_tensor) * 0.05

        x = self.rx_buff[:sr.ret].copy()
        x = x - np.mean(x)

        # ---- 6) 相关同步 ----
        corr = np.abs(np.correlate(x, self.preamble.conj(), mode="valid"))
        peak = int(np.argmax(corr))
        peak_val = float(corr[peak])
        self.last_sync_peak = peak_val

        # preamble 是否截断
        if peak + self.preamble_len > len(x):
            return torch.randn_like(latent_tensor) * 0.05

        r_pre = x[peak:peak + self.preamble_len]

        # ---- 7) CFO 估计 + 校正 ----
        cfo = estimate_cfo_from_repeated_preamble(r_pre, self.fs, barker_len=13, repeats=self.preamble_repeats)
        self.last_cfo_hz = cfo
        x_c = apply_cfo_correction(x, self.fs, cfo)

        r_pre_c = x_c[peak:peak + self.preamble_len]
        snr_db, _ = ls_snr_estimate(r_pre_c, self.preamble)
        self.last_snr_db = snr_db

        # ---- 8) 提取 payload ----
        start = peak + self.preamble_len
        end = start + len(payload)
        if end > len(x_c):
            return torch.randn_like(latent_tensor) * 0.05

        rx_payload = x_c[start:end]

        # ---- 9) complex -> float vector ----
        rx_flat = np.zeros(len(rx_payload) * 2, dtype=np.float32)
        rx_flat[0::2] = rx_payload.real
        rx_flat[1::2] = rx_payload.imag

        # ---- 10) 还原形状 ----
        rx_tensor = torch.from_numpy(rx_flat).view(original_shape).to(latent_tensor.device)
        return rx_tensor
