# Filepath: algorithms/on_policy/r_mappo/algorithm/r_actor_critic_2actor.py
# -*- coding: utf-8 -*-
import copy
import torch
import torch.nn as nn

from algorithms.on_policy.utils.util import init, check
from algorithms.on_policy.utils.cnn import CNNBase
from algorithms.on_policy.utils.mlp import MLPBase
from algorithms.on_policy.utils.rnn import RNNLayer
from algorithms.on_policy.utils.act_2actor import ACTLayer
from algorithms.on_policy.utils.popart import PopArt
from utils.util import get_shape_from_obs_space


class R_Actor(nn.Module):
    """
    Recurrent Actor for MAPPO.

    适配点：
      - 同时支持 MultiDiscrete（如 SD 的 [dir, fly, ds]）与 Box（如 BS 的连续传输权重向量）。
      - 仅对离散类动作（Discrete/MultiDiscrete/MultiBinary）处理 available_actions 掩码；
        对 Box 忽略掩码（应由分布本身或 env 侧约束）。
      - available_actions 的归一化交由 ACTLayer.normalize_available_actions() 完成（若实现）。
      - 观测：向量 -> MLPBase；图像 -> CNNBase（由 get_shape_from_obs_space 自动判断）。

    本版本支持为 Actor 单独设置 hidden_size 与层数 layer_N：
      --actor_hidden_size / --actor_layer_N
    未提供时回退到 --hidden_size / --layer_N。
    """

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()

        # ===== 独立的 Actor hidden/层数（未提供则回退到全局） =====
        self.hidden_size = args.actor_hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N

        self.tpdv = dict(dtype=torch.float32, device=device)

        # 记录动作空间类型，区分连续/离散
        self._act_space_name = action_space.__class__.__name__
        self._is_discrete_like = self._act_space_name in ("Discrete", "MultiDiscrete", "MultiBinary")

        # ----- feature base -----
        obs_shape = get_shape_from_obs_space(obs_space)
        Base = CNNBase if len(obs_shape) == 3 else MLPBase

        # 仅对 Actor 的 Base 覆写 hidden_size / layer_N
        args_base = copy.deepcopy(args)
        args_base.hidden_size = self.hidden_size
        args_base.layer_N = args.actor_layer_N

        self.base = Base(args_base, obs_shape)

        # ----- RNN -----
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        # ----- Action head -----
        # 需要 ACTLayer 已支持：MultiDiscrete(多头Categorical) 与 Box(如高斯/β分布, 支持边界)
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    # ---- helper: 归一化 / 适配 available_actions 掩码到 ACTLayer 期望的格式 ----
    def _normalize_available_actions(self, available_actions):
        """
        Env 可能返回以下几类 action mask（仅对离散空间有意义）：
          (a) None
          (b) shape=[B, sum(nvec)] 的拼接掩码
          (c) shape=[B, H, max_dim] 的按头填充掩码
          (d) list/tuple，长度 H，每个元素 shape=[B, dim_h]
        若 ACTLayer 暴露 normalize_available_actions()，则优先调用以获得与其内部一致的格式；
        否则按原样返回，由 ACTLayer 内部自行处理（若已支持）。
        """
        if (available_actions is None) or (not self._is_discrete_like):
            return None
        if hasattr(self.act, "normalize_available_actions"):
            return self.act.normalize_available_actions(available_actions)
        return available_actions

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        :param obs:                (np.ndarray / torch.Tensor) 观测
        :param rnn_states:         (np.ndarray / torch.Tensor) RNN 隐状态
        :param masks:              (np.ndarray / torch.Tensor) episode mask（用于重置隐状态）
        :param available_actions:  (np.ndarray / torch.Tensor / list) 离散动作可行掩码（Box 将被忽略，可为 None）
        :param deterministic:      (bool) True 则取 mode；False 则采样
        :return actions:           (Tensor) [B, H]（MultiDiscrete 的头数；Box 为动作维度）
        :return action_log_probs:  (Tensor) [B, 1]（与 rMAPPO 计算对齐）
        :return rnn_states:        (Tensor) 更新后的隐状态
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # 注意：不要把可为 list/tuple 的 available_actions 强转成 tensor，交给规范化函数处理
        norm_masks = self._normalize_available_actions(available_actions)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, norm_masks, deterministic)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        训练阶段计算 log_prob 与 entropy。
        :param action: (Tensor) [B, H]，MultiDiscrete 为各头的整型动作；Box 为连续动作（与 ACTLayer 一致）
        :param available_actions: 同 forward；仅离散时生效
        :param active_masks: (Tensor) [B, 1]，若启用 _use_policy_active_masks，用于熵的掩码
        :return action_log_probs: (Tensor) [B, 1]
        :return dist_entropy:     (Tensor) 标量熵或按样本平均后的熵（取决于 ACTLayer 实现）
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        norm_masks = self._normalize_available_actions(available_actions)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            norm_masks,
            active_masks=active_masks if self._use_policy_active_masks else None
        )
        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """
    Recurrent Critic for MAPPO.
    - 在 MAPPO 下通常输入 centralized obs（如环境提供的全局状态向量）。
    - 若使用 IPPO，可将 cent_obs_space 直接设为各自局部观测空间。

    本版本支持为 Critic 单独设置 hidden_size 与层数 layer_N：
      --critic_hidden_size / --critic_layer_N
    未提供时回退到 --hidden_size / --layer_N。
    """

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()

        # ===== 独立的 Critic hidden/层数（未提供则回退到全局） =====
        self.hidden_size = args.critic_hidden_size

        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart

        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        Base = CNNBase if len(cent_obs_shape) == 3 else MLPBase

        # 仅对 Critic 的 Base 覆写 hidden_size / layer_N
        args_base = copy.deepcopy(args)
        args_base.hidden_size = self.hidden_size
        args_base.layer_N = args.critic_layer_N

        self.base = Base(args_base, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        :param cent_obs:  (np.ndarray / torch.Tensor) centralized obs
        :param rnn_states:(np.ndarray / torch.Tensor) RNN 隐状态
        :param masks:     (np.ndarray / torch.Tensor) episode mask（重置隐状态）
        :return values:   (Tensor) [B, 1]
        :return rnn_states: 更新后的隐状态
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        values = self.v_out(critic_features)
        return values, rnn_states
