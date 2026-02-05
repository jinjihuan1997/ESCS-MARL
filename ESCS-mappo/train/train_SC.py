# Filepath: train/train_discrete.py
# -*- coding: utf-8 -*-

import sys
import os
import setproctitle
import numpy as np
from pathlib import Path
import torch

# 允许从项目根目录运行
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(parent_dir)

from config import get_config
from envs.env_wrappers import DummyVecEnv
from envs.env_discrete import DiscreteActionEnv


# =============== Vec-Env =============== #
def make_train_env(all_args):
    """创建 DiscreteActionEnv；并行 n_rollout_threads 个副本。"""
    def get_env_fn(rank):
        def init_env():
            seed = all_args.seed + rank * 50
            print("training seed ", seed)
            comm_mode = "SC"
            print("comm mode ", comm_mode)
            env = DiscreteActionEnv(seed=seed, debug=False, comm_mode=comm_mode)
            if hasattr(env, "seed"):
                env.seed(seed)
            return env
        return init_env
    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    """评估环境：并行 n_eval_rollout_threads 个副本。"""
    def get_env_fn(rank):
        def init_env():
            seed = all_args.seed + rank * 50
            print("training seed ", seed)
            comm_mode = "SC"
            print("comm mode ", comm_mode)
            env = DiscreteActionEnv(seed=seed, debug=True, comm_mode=comm_mode)
            if hasattr(env, "seed"):
                env.seed(seed)
            return env
        return init_env
    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


# =============== Args =============== #
def parse_args(args, parser):
    # 固定跑当前离散环境
    parser.add_argument("--scenario_name", type=str, default="Discrete",
                        help="Scenario name (fixed to Discrete)")
    # 是否共享策略（保留参数，默认共享）
    parser.add_argument("--share_policy", action="store_false", default=True,
                        help="Whether agents share the same policy")
    return parser.parse_known_args(args)[0]


# =============== Main =============== #
def main(argv):
    parser = get_config()
    all_args = parse_args(argv, parser)
    all_args.model_dir = None  # 训练从头开始

    # ---- 算法前置断言 ----
    if all_args.algorithm_name == "r_mappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), "r_mappo 需启用 RNN。"
    elif all_args.algorithm_name == "mappo":
        assert (not all_args.use_recurrent_policy and not all_args.use_naive_recurrent_policy), "mappo 需关闭 RNN。"
    else:
        raise NotImplementedError(f"Unsupported algorithm: {all_args.algorithm_name}")

    # ---- 设备 ----
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # ---- 结果目录 ----
    all_args.env_name = "SCEnv/1actor"   # 自定义名，保持与日志/结果目录一致
    run_root = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    run_root.mkdir(parents=True, exist_ok=True)

    # 找一个新的 runX 目录
    existing = [f for f in run_root.iterdir() if f.name.startswith("run")]
    if not existing:
        curr_run = "run1"
    else:
        exst = [int(f.name.replace("run", "")) for f in existing if f.name.replace("run", "").isdigit()]
        curr_run = f"run{(max(exst) + 1) if exst else 1}"
    run_dir = run_root / curr_run
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    setproctitle.setproctitle(
        f"{all_args.algorithm_name}-{all_args.env_name}-{all_args.experiment_name}@{all_args.user_name}"
    )

    # ---- 随机种子 ----
    torch.manual_seed(all_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # ---- 创建环境 ----
    envs = make_train_env(all_args)
    print("[check] train action_space[0]:", type(envs.action_space[0]).__name__, envs.action_space[0])
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    # ---- 回写关键信息到 all_args（供策略/runner使用）---- #
    env0 = envs.envs[0]
    # 真实 agent 数
    all_args.num_agents = int(getattr(env0, "num_agents",
                                      getattr(env0, "num_agent",
                                      getattr(env0, "agent_num", 1))))

    # train/train_discrete.py
    if not hasattr(all_args, "use_obs_instead_of_state"):
        all_args.use_obs_instead_of_state = False
    if not hasattr(all_args, "use_centralized_V"):
        all_args.use_centralized_V = True

    # ---- 组装 Runner 的 config ----
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # ---- 选择 Runner（EnvRunner 已适配 MultiDiscrete 与掩码）---- #
    from runner.env_runner import EnvRunner as Runner
    runner = Runner(config)

    # ---- 开跑 ----
    runner.run()

    # ---- 清理 ----
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    # ---- 导出 TB 标量 ----
    # 注意：runner.log_dir 是字符串路径
    from json import dumps
    try:
        runner.writter.export_scalars_to_json(os.path.join(runner.log_dir, "summary.json"))
    except Exception:
        # 某些版本的 SummaryWriter 可能没有该方法，忽略即可
        with open(os.path.join(runner.log_dir, "summary.json"), "w", encoding="utf-8") as f:
            f.write(dumps({}, ensure_ascii=False))
    runner.writter.close()


if __name__ == "__main__":
    import sys as _sys
    main(_sys.argv[1:])
