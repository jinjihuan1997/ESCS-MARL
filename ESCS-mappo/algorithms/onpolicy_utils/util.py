#  Filename:algorithms/onpolicy_utils/util.py
# utils/util.py
import math
import numpy as np
import torch


def check(x):
    """
    统一把 numpy 转成 torch.Tensor；其余保持不变。
    - x 为 None -> None
    - x 为 np.ndarray -> torch.from_numpy(x)
    - x 已是 torch.Tensor -> 原样返回
    - 其他标量/列表 -> torch.as_tensor(x)
    """
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if torch.is_tensor(x):
        return x
    # 标量/列表等
    return torch.as_tensor(x)


def get_gard_norm(it):
    """和原逻辑一致：累计参数梯度的 L2 范数。"""
    sum_grad = 0.0
    for p in it:
        if p.grad is None:
            continue
        # 兼容稀疏/致密
        g = p.grad.data
        if g.is_sparse:
            sum_grad += g.coalesce().values().norm() ** 2
        else:
            sum_grad += g.norm() ** 2
    return math.sqrt(float(sum_grad))


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """线性下降学习率：从 initial_lr 线性衰减到 0。"""
    lr = float(initial_lr) * max(0.0, 1.0 - (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def huber_loss(e, d):
    """Huber 损失：与原实现一致。"""
    a = (torch.abs(e) <= d).float()
    b = (torch.abs(e) > d).float()
    return a * e ** 2 / 2 + b * d * (torch.abs(e) - d / 2)


def mse_loss(e):
    """与原实现一致：0.5 * e^2"""
    return e ** 2 / 2


def get_shape_from_obs_space(obs_space):
    """
    兼容常见观测空间：
    - gym.spaces.Box -> 返回其 shape
    - Python list    -> 直接返回（与原逻辑保持兼容）
    """
    name = obs_space.__class__.__name__
    if name == "Box":
        return tuple(obs_space.shape)
    if name == "list":
        return obs_space
    # 可按需扩展 Dict/Tuple 等
    raise NotImplementedError(f"Unsupported obs_space type: {name}")


def get_shape_from_act_space(act_space):
    """
    统一返回“动作向量最后一维长度”的整数（供 buffer 分配、actor 输出对齐）：
      - Discrete(n)         -> 1
      - MultiDiscrete(nvec) -> len(nvec)   （优先使用 Gym 的 nvec；无则回退 high/low）
      - MultiBinary(n)      -> n
      - Box(shape[0])       -> shape[0]
      - Tuple(…)            -> 各子空间长度相加（递归）
    说明：对于 MultiDiscrete，我们的 buffer/runner 都按“头数 H”存放**整数向量**，
          每个头的类别数由策略头部内部（ACTLayer）根据 nvec 管理。
    """
    name = act_space.__class__.__name__

    if name == "Discrete":
        return 1

    elif name == "MultiDiscrete":
        # Gym 规范：nvec 是每个头的类别数数组
        if hasattr(act_space, "nvec"):
            return int(len(np.asarray(act_space.nvec)))
        # 兼容早期自定义类（有 high/low）
        if hasattr(act_space, "high") and hasattr(act_space, "low"):
            return int(len(np.asarray(act_space.high)))
        raise AttributeError("Unsupported MultiDiscrete: missing nvec and (high/low).")

    elif name == "MultiBinary":
        if hasattr(act_space, "n"):
            return int(act_space.n)
        return int(act_space.shape[0])

    elif name == "Box":
        return int(act_space.shape[0])

    elif name == "Tuple":
        # Gym 的 Tuple(spaces=[...]) 或可迭代容器
        try:
            spaces = getattr(act_space, "spaces", None)
            if spaces is None:
                spaces = list(act_space)
            return int(sum(get_shape_from_act_space(s) for s in spaces))
        except Exception as e:
            raise AttributeError(f"Unsupported Tuple-like action space: {act_space}") from e

    # 兜底：若带 shape[0]，取之；否则返回 1
    try:
        return int(act_space.shape[0])
    except Exception:
        return 1


def tile_images(img_nhwc):
    """
    把 N 张 (H,W,C) 图平铺成一张大图，尽量接近正方形的网格。
    输入：img_nhwc，list 或 ndarray，形状 (N, H, W, C)
    输出：大图，形状 (H*, W*, C)
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    # padding 到 H*W
    if N < H * W:
        pad = np.zeros((H * W - N, h, w, c), dtype=img_nhwc.dtype)
        img_nhwc = np.concatenate([img_nhwc, pad], axis=0)
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c

