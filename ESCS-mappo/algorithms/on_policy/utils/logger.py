# utils/logger.py
import logging
import sys
import os
import json
from logging.handlers import RotatingFileHandler

__all__ = [
    "init_logging",
    "setup_logger",
    "set_log_enabled",
    "is_log_enabled",
    "should_log",
    "kv",
    "log",
    "tstat",
]

_LEVELS = {
    "CRITICAL": 50, "ERROR": 40, "WARNING": 30,
    "INFO": 20, "DEBUG": 10, "NOTSET": 0
}

# ---- 全局总开关（进程级）----
_LOG_ENABLED = True


def is_log_enabled() -> bool:
    return _LOG_ENABLED


def set_log_enabled(enabled: bool):
    """运行期可切换。False 时彻底禁用所有 logging 调用。"""
    global _LOG_ENABLED
    _LOG_ENABLED = bool(enabled)
    if _LOG_ENABLED:
        logging.disable(logging.NOTSET)
    else:
        logging.disable(logging.CRITICAL + 1)


def _parse_level(level) -> int:
    if isinstance(level, int):
        return level
    return _LEVELS.get(str(level).upper(), logging.INFO)


def init_logging(level="INFO", log_file: str = "", enabled: bool = True):
    """
    入口初始化：设定 root logger 的级别/handler，并应用全局开关。
    - level: 'INFO'/'DEBUG'/... 或整数
    - log_file: 为空仅输出到控制台；否则也写滚动文件
    - enabled: False 则彻底静音（也可用环境变量 SC_LOG_ENABLE=0 覆盖）
    可重复调用（更新级别 & 追加文件 handler）。
    """
    # 环境变量兜底
    env_flag = os.getenv("SC_LOG_ENABLE", "").strip()
    if env_flag != "":
        try:
            enabled = bool(int(env_flag))
        except Exception:
            pass

    set_log_enabled(enabled)

    lvl = _parse_level(level)
    root = logging.getLogger()
    root.setLevel(lvl)

    # 若 root 无 handler，则创建一个控制台 handler
    if not root.handlers:
        fmt = os.getenv("LOG_FMT", "[%(asctime)s][%(levelname)s][%(name)s] %(message)s")
        datefmt = "%H:%M:%S"
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(lvl)
        ch.setFormatter(logging.Formatter(fmt, datefmt))
        root.addHandler(ch)

    # 同步所有已存在 handler 的级别
    for h in root.handlers:
        h.setLevel(lvl)

    # 可选：滚动文件
    if log_file:
        absf = os.path.abspath(log_file)
        exists = any(
            isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", None) == absf
            for h in root.handlers
        )
        if not exists:
            fmt = os.getenv("LOG_FMT", "[%(asctime)s][%(levelname)s][%(name)s] %(message)s")
            datefmt = "%H:%M:%S"
            fh = RotatingFileHandler(absf, maxBytes=50_000_000, backupCount=5, encoding="utf-8")
            fh.setLevel(lvl)
            fh.setFormatter(logging.Formatter(fmt, datefmt))
            root.addHandler(fh)


def setup_logger(name, level=logging.INFO):
    """
    返回命名 logger。默认让消息上冒到 root（不额外加 handler），避免重复输出。
    若你希望和旧行为一致（为每个 logger 单独加控制台 handler），
    可自行在外部给该 logger 添加 handler 并把 propagate 设为 False。
    """
    # 确保 root 至少有一个 handler（即使用户没显式调 init_logging）
    if not logging.getLogger().handlers:
        init_logging(level=level)

    logger = logging.getLogger(name)
    logger.setLevel(_parse_level(level))
    logger.propagate = True  # 交给 root 统一输出
    # 清掉该 logger 下面意外加过的 handler，避免重复
    for h in list(logger.handlers):
        logger.removeHandler(h)
    return logger


def should_log(step: int, every: int, warmup: int = 5) -> bool:
    """按步采样：前 warmup 次必打，其后每 every 次打一条。"""
    if step is None:
        return True
    if step < warmup:
        return True
    if not every or every <= 0:
        return False
    return (step % every) == 0


def kv(**kw) -> str:
    """
    将字段序列化为紧凑 JSON 字符串。
    注意：在调用方先判断 is_log_enabled() 与 logger.isEnabledFor(level) 再调用，避免无谓开销。
    """
    import numpy as _np

    def to_plain(x):
        if isinstance(x, _np.ndarray):
            return x.tolist()
        try:
            import torch  # 延迟导入
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().tolist()
        except Exception:
            pass
        return x

    return json.dumps({k: to_plain(v) for k, v in kw.items()},
                      ensure_ascii=False, separators=(",", ":"))


def log(logger: logging.Logger, level: int, msg_prefix: str = "", **fields):
    """
    统一的字段日志接口：带总开关 + 级别预判。
    只有在需要输出时，才会进行 kv() 组装（关闭时零开销）。
    用法：log(lg, logging.INFO, "[TAG] ", step=1, foo="bar")
    """
    if (not _LOG_ENABLED) or (logger is None) or (not logger.isEnabledFor(level)):
        return
    if fields:
        logger.log(level, "%s%s", msg_prefix, kv(**fields))
    else:
        logger.log(level, "%s", msg_prefix)


def tstat(x, name: str = "", keep_float: bool = False):
    """给 numpy/torch 张量做稳健统计；返回字符串。"""
    try:
        import torch, numpy as np
        if isinstance(x, torch.Tensor):
            x = x.detach().float().cpu().numpy()
        x = x.reshape(-1)
        if x.size == 0:
            return f"{name}(empty)"
        finite = np.isfinite(x)
        nf = finite.sum()
        if nf == 0:
            return f"{name}(all non-finite, n={x.size})"
        xf = x[finite]
        mn, mx = float(xf.min()), float(xf.max())
        mu, sd = float(xf.mean()), float(xf.std())
        nan_cnt = int(np.isnan(x).sum())
        inf_cnt = int(np.isinf(x).sum())
        extra = f", nan={nan_cnt}, inf={inf_cnt}" if (nan_cnt or inf_cnt) else ""
        if keep_float:
            return f"{name}(min={mn:.6g}, max={mx:.6g}, mean={mu:.6g}, std={sd:.6g}, n={x.size}{extra})"
        else:
            return f"{name}(min={mn:.3g}, max={mx:.3g}, mean={mu:.3g}, std={sd:.3g}, n={x.size}{extra})"
    except Exception as e:
        return f"{name}(stat_error={e})"
