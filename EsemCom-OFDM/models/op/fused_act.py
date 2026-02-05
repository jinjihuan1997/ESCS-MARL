# models/op/fused_act.py
import os
import torch
from torch import nn
from torch.nn import functional as F

# 允许通过环境变量强制禁用扩展（Windows 上建议默认禁用）
# PowerShell: $env:GANSECOM_DISABLE_EXT="1"
_DISABLE = os.environ.get("GANSECOM_DISABLE_EXT", "0") in ("1", "true", "True")

fused = None
if not _DISABLE:
    try:
        from torch.utils.cpp_extension import load
        module_path = os.path.dirname(__file__)
        fused = load(
            name="fused",
            sources=[
                os.path.join(module_path, "fused_bias_act.cpp"),
                os.path.join(module_path, "fused_bias_act_kernel.cu"),
            ],
            verbose=False,
        )
    except Exception as e:
        fused = None
        print("[WARN] fused extension load failed -> fallback to PyTorch. err =", e)
else:
    print("[INFO] fused extension disabled by GANSECOM_DISABLE_EXT=1 (fallback to PyTorch).")


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    """
    Pure PyTorch fallback:
      y = scale * leaky_relu(x + bias)
    """
    if bias is not None:
        if input.dim() == 2:
            input = input + bias.view(1, -1)
        else:
            input = input + bias.view(1, -1, 1, 1)
    out = F.leaky_relu(input, negative_slope=negative_slope)
    return out * scale


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.negative_slope = negative_slope
        self.scale = scale
        self.bias = nn.Parameter(torch.zeros(channel)) if bias else None

    def forward(self, input):
        # If extension is available and provides fused_bias_act, use it.
        if fused is not None and self.bias is not None:
            try:
                if hasattr(fused, "fused_bias_act"):
                    return fused.fused_bias_act(input, self.bias, self.negative_slope, self.scale)
            except Exception:
                pass

        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)
