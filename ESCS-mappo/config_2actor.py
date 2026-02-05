# Filename: config_2actor.py
import argparse


def get_config():
    """
    Common + 2-actor (BS/SD) overrides.

    你可以通过下列分组参数分别设置两套 Actor（Critic 依然共享）：
        --bs_actor_lr / --sd_actor_lr
        --bs_entropy_coef / --sd_entropy_coef
        --bs_clip_param / --sd_clip_param
        --bs_ppo_epoch / --sd_ppo_epoch
        --bs_num_mini_batch / --sd_num_mini_batch
        --bs_data_chunk_length / --sd_data_chunk_length

    未显式提供时，这些分组参数将回退到通用参数（如 lr、entropy_coef、clip_param 等）。
    """
    parser = argparse.ArgumentParser(
        description="onpolicy", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ===== 算法选择 =====
    parser.add_argument(
        "--algo_group",
        type=str,
        default="on_policy",
        choices=["on_policy", "off_policy"],
        help="Which folder of algorithms to use: on_policy or off_policy",
    )
    parser.add_argument(
        "--algorithm_name",
        type=str,
        default="r_mappo",
        choices=[
            # on_policy
            "r_mappo", "mappo", "hatrpo", "happo", "mat",
            # off_policy
            "maddpg", "qmix", "matd3", "vdn", "mvdn", "r_maddpg", "r_matd3",
        ],
        help="Algorithm name; must match subfolder in on_policy/ or off_policy/",
    )
    parser.add_argument("--experiment_name", type=str, default="check")
    parser.add_argument("--seed", type=int, default=51)

    parser.add_argument("--cuda", action="store_false", default=True)
    parser.add_argument("--cuda_deterministic", action="store_false", default=True)

    parser.add_argument("--n_training_threads", type=int, default=20)
    parser.add_argument("--n_rollout_threads", type=int, default=20)
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1)
    parser.add_argument("--n_render_rollout_threads", type=int, default=1)

    parser.add_argument("--num_env_steps", type=int, default=5_000_000)
    parser.add_argument("--user_name", type=str, default="marl")

    # ===== Env =====
    parser.add_argument("--env_name", type=str, default="SCEnv")
    parser.add_argument("--use_obs_instead_of_state", action="store_true", default=False)

    # ===== Replay Buffer =====
    parser.add_argument("--episode_length", type=int, default=400)

    # ===== Network =====
    parser.add_argument("--use_local_obs", default=False)
    parser.add_argument("--use_centralized_V", action="store_false", default=True)
    parser.add_argument("--stacked_frames", type=int, default=1)
    parser.add_argument("--use_stacked_frames", action="store_true", default=False)

    parser.add_argument("--use_ReLU", action="store_false", default=True)

    parser.add_argument("--use_popart", action="store_true", default=False)
    parser.add_argument("--use_valuenorm", action="store_false", default=True)
    parser.add_argument("--use_feature_normalization", action="store_false", default=True)
    parser.add_argument("--use_orthogonal", action="store_false", default=True)
    parser.add_argument("--gain", type=float, default=0.01)

    # ===== Recurrent =====
    parser.add_argument("--use_naive_recurrent_policy", action="store_true", default=False)
    parser.add_argument("--use_recurrent_policy", action="store_false", default=True)
    parser.add_argument("--recurrent_N", type=int, default=3)
    parser.add_argument("--data_chunk_length", type=int, default=100)

    # ===== Optimizer (shared defaults) =====
    parser.add_argument("--lr", type=float, default=3e-4, help="default actor lr")
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--opti_eps", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # >>> 分组 Actor 覆写（可选，不给就用上面的全局默认） <<<
    parser.add_argument("--bs_actor_lr", type=float, default=3e-4)
    parser.add_argument("--sd_actor_lr", type=float, default=3e-4)

    # ===== PPO (shared defaults) =====
    parser.add_argument("--ppo_epoch", type=int, default=15)
    parser.add_argument("--use_clipped_value_loss", action="store_false", default=True)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--num_mini_batch", type=int, default=1)
    parser.add_argument("--entropy_coef", type=float, default=0.05)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--use_max_grad_norm", action="store_false", default=True)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--use_gae", action="store_false", default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--use_proper_time_limits", action="store_true", default=False)
    parser.add_argument("--use_huber_loss", action="store_false", default=True)
    parser.add_argument("--use_value_active_masks", action="store_false", default=True)
    parser.add_argument("--use_policy_active_masks", action="store_false", default=True)
    parser.add_argument("--huber_delta", type=float, default=10.0)

    # >>> 分组 Actor 的 PPO 覆写（可选） <<<
    parser.add_argument("--bs_entropy_coef", type=float, default=None)
    parser.add_argument("--sd_entropy_coef", type=float, default=None)

    parser.add_argument("--bs_clip_param", type=float, default=None)
    parser.add_argument("--sd_clip_param", type=float, default=None)

    parser.add_argument("--bs_ppo_epoch", type=int, default=None)
    parser.add_argument("--sd_ppo_epoch", type=int, default=None)

    parser.add_argument("--bs_num_mini_batch", type=int, default=None)
    parser.add_argument("--sd_num_mini_batch", type=int, default=None)

    parser.add_argument("--bs_data_chunk_length", type=int, default=None)
    parser.add_argument("--sd_data_chunk_length", type=int, default=None)

    # ===== Run / Save / Log =====
    parser.add_argument("--use_linear_lr_decay", action="store_true", default=True)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=5)

    # ===== Eval =====
    parser.add_argument("--use_eval", action="store_true", default=False)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=1)

    # ===== Render =====
    parser.add_argument("--save_gifs", action="store_true", default=False)
    parser.add_argument("--use_render", action="store_true", default=False)
    parser.add_argument("--render_episodes", type=int, default=1)
    parser.add_argument("--ifi", type=float, default=0.1)

    parser.add_argument("--actor_hidden_size", type=int, default=64)
    parser.add_argument("--critic_hidden_size", type=int, default=128)
    parser.add_argument("--actor_layer_N", type=int, default=2)
    parser.add_argument("--critic_layer_N", type=int, default=3)

    return parser
