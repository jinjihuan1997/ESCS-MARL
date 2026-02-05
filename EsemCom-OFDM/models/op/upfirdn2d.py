# models/op/upfirdn2d.py
# Pure PyTorch fallback for Windows (no custom CUDA/C++ extension)
import os
import torch
import torch.nn.functional as F

# Print once
if os.environ.get("GANSECOM_DISABLE_EXT", "0").lower() in ("1", "true", "yes"):
    print("[INFO] upfirdn2d extension disabled by GANSECOM_DISABLE_EXT=1 (fallback to PyTorch).")


def make_kernel(k):
    # k can be list/np/tensor; keep behavior consistent
    if isinstance(k, torch.Tensor):
        k = k.to(dtype=torch.float32)
    else:
        k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[:, None] * k[None, :]

    k /= k.sum()
    return k


def _parse_scaling(v):
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _parse_padding(pad):
    # supports (p0,p1) or (px0,px1,py0,py1)
    if isinstance(pad, (int, float)):
        pad = (int(pad), int(pad))

    if len(pad) == 2:
        px0, px1 = int(pad[0]), int(pad[1])
        py0, py1 = int(pad[0]), int(pad[1])
    elif len(pad) == 4:
        px0, px1, py0, py1 = map(int, pad)
    else:
        raise ValueError("pad must be int, (p0,p1) or (px0,px1,py0,py1)")
    return px0, px1, py0, py1


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """
    input:  (N, C, H, W)
    kernel: (kh, kw) or (k,)  -> internally converted to 2D
    up/down: int or (x,y)
    pad: (p0,p1) or (px0,px1,py0,py1)
    """
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    px0, px1, py0, py1 = _parse_padding(pad)

    x = input
    N, C, H, W = x.shape

    # kernel -> tensor on same device/dtype
    if not isinstance(kernel, torch.Tensor):
        kernel = torch.tensor(kernel, dtype=torch.float32, device=x.device)
    else:
        kernel = kernel.to(device=x.device, dtype=torch.float32)

    if kernel.ndim == 1:
        kernel = kernel[:, None] * kernel[None, :]

    # conv2d uses correlation; flip to match FIR filtering
    k = kernel.flip([0, 1]).to(dtype=x.dtype)
    k = k[None, None, :, :]  # (1,1,kh,kw)

    # 1) Upsample by inserting zeros
    if upx > 1 or upy > 1:
        x = x.reshape(N, C, H, 1, W, 1)
        x = F.pad(x, (0, upx - 1, 0, 0, 0, upy - 1))
        x = x.reshape(N, C, H * upy, W * upx)

    # 2) Pad
    if px0 != 0 or px1 != 0 or py0 != 0 or py1 != 0:
        x = F.pad(x, (px0, px1, py0, py1))

    # 3) Filter (depthwise: do it via (N*C,1,H,W))
    x = x.reshape(N * C, 1, x.shape[2], x.shape[3])
    x = F.conv2d(x, k, stride=1, padding=0)
    x = x.reshape(N, C, x.shape[2], x.shape[3])

    # 4) Downsample
    if downx > 1 or downy > 1:
        x = x[:, :, ::downy, ::downx]

    return x


def upsample2d(x, kernel, factor=2):
    k = make_kernel(kernel) * (factor * factor)
    p = k.shape[0] - factor
    pad0 = (p + 1) // 2 + factor - 1
    pad1 = p // 2
    return upfirdn2d(x, k, up=factor, down=1, pad=(pad0, pad1))


def downsample2d(x, kernel, factor=2):
    k = make_kernel(kernel)
    p = k.shape[0] - factor
    pad0 = (p + 1) // 2
    pad1 = p // 2
    return upfirdn2d(x, k, up=1, down=factor, pad=(pad0, pad1))


def blur2d(x, kernel, pad=(0, 0)):
    k = make_kernel(kernel)
    return upfirdn2d(x, k, up=1, down=1, pad=pad)
