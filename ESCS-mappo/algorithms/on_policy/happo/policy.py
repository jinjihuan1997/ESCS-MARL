import torch
from algorithms.on_policy.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from algorithms.onpolicy_utils.util import update_linear_schedule


class HAPPO_Policy:
    """
    HAPPO Policy: decentralized actor + centralized critic（单-actor环境适配）
    - 兼容：Actor.evaluate_actions 可能返回【标量】或【逐头向量】的 log_prob。
    - 本文件在 get_actions / evaluate_actions 中做了“逐头形状兜底”（可选增强）；
      即将 (B,) 或 (B,1) 的 log_prob 扩展为 (B, H)，H 为 MultiDiscrete 动作头数。
      这样 Buffer 里 old_action_log_probs 统一是逐头向量，更便于分析与对齐。
    - 若你不想改这里，也没问题：trainer.py 已做广播兜底。
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        # ---- networks ----
        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        # 显式放到目标设备
        self.actor.to(self.device)
        self.critic.to(self.device)

        # ---- optim ----
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay
        )

        # 动作头数（MultiDiscrete 时等于 len(nvec)；否则为 1）
        self.action_heads = self._infer_action_heads(self.act_space)

    # ---------------- helpers ----------------
    @staticmethod
    def _infer_action_heads(act_space):
        """推断 MultiDiscrete 的动作头数；非 MultiDiscrete 时为 1。"""
        try:
            # gym/gymnasium MultiDiscrete 都有 nvec
            if hasattr(act_space, "nvec"):
                return int(len(act_space.nvec))
        except Exception:
            pass
        return 1

    def _ensure_headwise(self, logp: torch.Tensor) -> torch.Tensor:
        """
        将 log_prob 统一为 (B, H)：
        - 若为 (B,) 或 (B,1) 且 H>1，则扩展为 (B, H)
        - 若已为 (B, H) 则原样返回
        """
        if not isinstance(logp, torch.Tensor):
            return logp
        if logp.dim() == 1:
            logp = logp.unsqueeze(-1)  # (B,) -> (B,1)
        if logp.size(-1) == 1 and self.action_heads > 1:
            logp = logp.expand(logp.size(0), self.action_heads)
        return logp

    # ---------------- schedulers ----------------
    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    # ---------------- act & value ----------------
    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks,
                    available_actions=None, deterministic=False):
        """
        rollout：给定 obs 采样动作；给定 cent_obs 估计 V(s)。
        为了让 Buffer 里 old_action_log_probs 统一为 (B, H)，这里做了形状兜底。
        """
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        # 形状兜底（可选增强）
        action_log_probs = self._ensure_headwise(action_log_probs)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    # ---------------- train-time evaluate ----------------
    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        train：评估给定 action 的 log_prob & 熵，并估计 V(s)。
        同样对 action_log_probs 做逐头形状兜底，保证 (B, H)。
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )
        # 形状兜底（可选增强）
        action_log_probs = self._ensure_headwise(action_log_probs)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    # ---------------- rollout-time act ----------------
    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        return actions, rnn_states_actor
