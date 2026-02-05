# Filepath: algorithms/on_policy/utils/act_2actor.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import Bernoulli, Categorical, DiagGaussian


class ACTLayer(nn.Module):
    """
    Action head that builds the appropriate distribution(s) based on the action space.

    支持动作掩码：
      - Discrete: available_actions -> [B, A] 或 [A]（自动广播到 batch）
      - MultiDiscrete: 支持以下三种输入之一（会规范化为 list[len=H]）：
          (a) [B, sum(nvec)] 扁平拼接
          (b) [B, H, max_dim] 按头填充（仅取前 nvec[h] 位）
          (c) list/tuple，len = H，每个元素 [B, nvec[h]] 或 [nvec[h]]
      其余保持兼容（None 视为全可用）。

    输出形状（与常见 MAPPO/RMAPPO 一致）：
      - MultiDiscrete:
          forward():            actions [B,H] (long), action_log_probs [B,1]（对各头求和）
          evaluate_actions():   action_log_probs [B,1]（对各头求和）, dist_entropy 标量（头熵平均）
      - Discrete: 与底层分布一致（log_prob 为 [B,1]）
      - Box / MultiBinary: 与底层分布一致
    """

    def __init__(self, action_space, inputs_dim, use_orthogonal, gain):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.continuous_action = False

        # 调试打印节流
        self._dbg_cnt = 0

        space_name = action_space.__class__.__name__

        if space_name == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
            self.action_dims = [action_dim]
            self.n_heads = 1

        elif space_name == "Box":
            # 连续动作：改为 Sigmoid(Gaussian) -> y∈(0,1)（含可选仿射缩放避免全 0）
            self.continuous_action = True
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
            self.action_dims = [action_dim]
            self.n_heads = 1

            # ====== 新增配置：把 Box 输出压到 [0,1]，并在 log_prob 中做雅可比修正 ======
            self._squash_to_unit = True
            self._eps = 1e-6            # 数值安全边界，避免 log(0)
            self._eps_affine = 1e-3     # y = eps + (1-2eps)*sigmoid(z)，避免一开始全部为 0
            # print(f"[ACT2] Box head online, squash_to_unit={self._squash_to_unit}, "
            #       f"dim={action_dim}, eps={self._eps}, eps_affine={self._eps_affine}")

        elif space_name == "MultiBinary":
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)
            self.action_dims = [action_dim]
            self.n_heads = 1

        elif space_name == "MultiDiscrete":
            self.multi_discrete = True
            if hasattr(action_space, "nvec"):
                dims = action_space.nvec
            elif hasattr(action_space, "high") and hasattr(action_space, "low"):
                dims = (action_space.high - action_space.low + 1)
            else:
                raise AttributeError("Unsupported MultiDiscrete: missing nvec and (high/low).")
            self.action_dims = [int(x) for x in list(dims)]  # python int list
            self.n_heads = len(self.action_dims)
            self.action_outs = nn.ModuleList(
                [Categorical(inputs_dim, d, use_orthogonal, gain) for d in self.action_dims]
            )

        else:  # mixed: [Box, Discrete] —— 如未使用可忽略
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_outs = nn.ModuleList(
                [
                    DiagGaussian(inputs_dim, continous_dim, use_orthogonal, gain),
                    Categorical(inputs_dim, discrete_dim, use_orthogonal, gain),
                ]
            )
            self.action_dims = [continous_dim, discrete_dim]
            self.n_heads = 2

    # ------------------------- 工具：掩码转 tensor / 广播 ------------------------- #
    @staticmethod
    def _ensure_tensor(x, dtype=torch.float32, device=None):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.to(dtype=dtype, device=device)
        return torch.as_tensor(x, dtype=dtype, device=device)

    @staticmethod
    def _maybe_expand_batch(mask, B):
        """若 mask 是 [D] 且需要广播到 [B,D]，在此处理。"""
        if mask is None:
            return None
        if mask.dim() == 1:
            return mask.unsqueeze(0).expand(B, -1).contiguous()
        return mask

    def _prep_discrete_mask(self, mask, B, A, device):
        """
        Discrete 情况下规范化掩码：
          - 接受 [A] 或 [B,A]，转换到 tensor，搬到 device
          - 返回 [B,A] 或 None
        """
        if mask is None:
            return None
        m = self._ensure_tensor(mask, device=device)
        if m.dim() == 1:
            if m.size(0) != A:
                return None
            m = m.unsqueeze(0).expand(B, -1).contiguous()
        elif m.dim() == 2:
            if m.size(0) != B or m.size(1) != A:
                return None
        else:
            return None
        return m

    # ===================== Sigmoid 压缩 / 反变换 与 雅可比修正 ===================== #
    def _squash01(self, z: torch.Tensor) -> torch.Tensor:
        """
        把实数域 z 压到 (0,1)，并可选仿射缩放：y = eps + (1-2eps)*sigmoid(z)
        最终 clamp 避免 log(0)。
        """
        y0 = torch.sigmoid(z)
        eps_aff = getattr(self, "_eps_affine", 0.0)
        if eps_aff > 0.0:
            y = eps_aff + (1.0 - 2.0 * eps_aff) * y0
        else:
            y = y0
        return y.clamp(self._eps, 1.0 - self._eps)

    def _unsquash_to_logit(self, y: torch.Tensor):
        """
        对 forward 产生的 y∈(0,1) 做反变换：
          若 y = eps + (1-2eps)*sigmoid(z)，则 y0 = (y-eps)/(1-2eps)，z = logit(y0)
        返回 (z, y0) 供 log_prob 与雅可比修正使用。
        """
        y = y.clamp(self._eps, 1.0 - self._eps)
        eps_aff = getattr(self, "_eps_affine", 0.0)
        if eps_aff > 0.0:
            scale = (1.0 - 2.0 * eps_aff)
            y0 = (y - eps_aff) / scale
            y0 = y0.clamp(self._eps, 1.0 - self._eps)
        else:
            y0 = y
            scale = 1.0
        z = torch.log(y0) - torch.log1p(-y0)  # logit(y0)
        return z, y0, scale

    def _logprob_after_sigmoid(self, base_dist, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        变量替换：z ~ N(mu, sigma)，y = eps + (1-2eps)*sigmoid(z)
        log p_Y(y) = log p_Z(z) - sum_i [ log(y0_i) + log(1 - y0_i) ] - D*log(1-2eps)
        其中 y0 = (y - eps) / (1 - 2eps)；当 eps=0 时退化为标准 Sigmoid 情形。
        base_dist.log_probs(z) 的结果形状为 [B,1]（各维已求和）。
        """
        # 先把 y 反变换到 y0 以计算 Sigmoid 的雅可比；同时得到仿射缩放因子 scale
        _, y0, scale = self._unsquash_to_logit(y)
        lp_z = base_dist.log_probs(z)  # [B,1]
        # Sigmoid 的 log|Jac| = sum_i log(y0) + log(1-y0)
        log_jac_sig = torch.sum(torch.log(y0) + torch.log1p(-y0), dim=-1, keepdim=True)
        # 仿射缩放的 log|Jac| = D * log(scale)；注意 scale = 1-2eps
        if isinstance(scale, float):
            # 常量 scale
            if scale == 1.0:
                log_jac_affine = 0.0
            else:
                D = y.size(-1)
                log_jac_affine = y.new_full((y.size(0), 1), D * torch.log(y.new_tensor(scale)))
        else:
            # 理论上 scale 为常量，这里以防万一
            D = y.size(-1)
            log_jac_affine = torch.sum(torch.log(scale).expand_as(y), dim=-1, keepdim=True)
        return lp_z - log_jac_sig - log_jac_affine

    # ===================== 掩码规范化（MultiDiscrete 专用） ===================== #
    def normalize_available_actions(self, available_actions):
        """
        将 MultiDiscrete 的可行动作掩码规范化为 list[len=H]，每个元素 shape [B, action_dims[h]] 或 None。
        允许输入：
          - None
          - torch.Tensor [B, sum(nvec)] 或 [sum(nvec)]
          - torch.Tensor [B, H, max_dim]
          - list/tuple 长度 H，每个元素 [B, nvec[h]] 或 [nvec[h]]
        非 MultiDiscrete 情况下，原样返回（由底层分布处理）。
        """
        if not self.multi_discrete:
            return available_actions

        H = self.n_heads

        # 1) None
        if available_actions is None:
            return [None] * H

        # 2) list/tuple
        if isinstance(available_actions, (list, tuple)):
            if len(available_actions) != H:
                return [None] * H
            out = []
            B = None
            dev = None
            # 先探 batch 大小/设备
            for m in available_actions:
                if m is None:
                    continue
                t = self._ensure_tensor(m)
                if t.dim() == 2:
                    B = t.size(0)
                    dev = t.device
                    break
            for h in range(H):
                mh = self._ensure_tensor(available_actions[h], device=dev)
                if mh is None:
                    out.append(None)
                    continue
                if mh.dim() == 1:
                    Bh = 1 if B is None else B
                    mh = mh.unsqueeze(0).expand(Bh, -1).contiguous()
                out.append(mh[:, : self.action_dims[h]])
            return out

        # 3) torch.Tensor
        t = self._ensure_tensor(available_actions)
        if t.dim() == 1:
            # [sum(nvec)] -> 先广播 batch=1
            t = t.unsqueeze(0)
        if t.dim() == 2:
            # [B, sum(nvec)] —— split
            splits = list(torch.split(t, self.action_dims, dim=-1))
            return splits
        if t.dim() == 3:
            # [B, H, max_dim] —— 逐头裁切
            B, Hin, Dp = t.size()
            if Hin != H:
                return [None] * H
            return [t[:, h, : self.action_dims[h]] for h in range(H)]

        return [None] * H

    # ============================== 前向：采样动作 ============================== #
    def forward(self, x, available_actions=None, deterministic=False):
        """
        :return actions:
            - MultiDiscrete: [B, H]（long）
            - 其他：与底层分布一致
        :return action_log_probs:
            - MultiDiscrete: [B, 1]（逐头 log-prob 之和）
            - 其他：与底层分布一致（通常 [B,1]）
        """
        B = x.size(0)

        if self.mixed_action:
            actions, logps = [], []
            for action_out in self.action_outs:
                logits = action_out(x)
                a = logits.mode() if deterministic else logits.sample()
                lp = logits.log_probs(a)  # [B,1] 或 [B,dim]
                actions.append(a.float())
                logps.append(lp if lp.dim() == 2 else lp.unsqueeze(-1))
            actions = torch.cat(actions, dim=-1)
            action_log_probs = torch.cat(logps, dim=-1).sum(dim=-1, keepdim=True)
            return actions, action_log_probs

        if self.multi_discrete:
            masks = self.normalize_available_actions(available_actions)
            acts, logps = [], []
            for h, action_out in enumerate(self.action_outs):
                # 将每个头的掩码搬到与 x 相同的 device
                mask_h = masks[h] if isinstance(masks, list) else None
                if isinstance(mask_h, torch.Tensor):
                    mask_h = mask_h.to(x.device, non_blocking=True)
                    # 仅取前 action_dims[h]
                    mask_h = mask_h[:, : self.action_dims[h]]
                logits_h = action_out(x, mask_h)
                a_h = logits_h.mode() if deterministic else logits_h.sample()   # [B,1] long
                lp_h = logits_h.log_probs(a_h)                                  # [B,1]
                acts.append(a_h.squeeze(-1))                                    # [B]
                logps.append(lp_h)                                              # [B,1]
            actions = torch.stack(acts, dim=-1).long()                          # [B,H]
            action_log_probs = torch.cat(logps, dim=-1).sum(dim=-1, keepdim=True)  # [B,1]
            return actions, action_log_probs

        if self.continuous_action:
            logits = self.action_out(x)                                # 高斯（实数域）
            z = logits.mode() if deterministic else logits.sample()    # [B, D]
            if getattr(self, "_squash_to_unit", False):
                y = self._squash01(z)                                  # 压到 (0,1)，含仿射缩放
                action_log_probs = self._logprob_after_sigmoid(logits, z, y)   # [B,1]

                # 调试打印（每 10 次打印一次）
                self._dbg_cnt += 1
                if self._dbg_cnt % 10 == 1:
                    with torch.no_grad():
                        z_min = z.min().item(); z_max = z.max().item(); z_mean = z.mean().item()
                        y_min = y.min().item(); y_max = y.max().item(); y_mean = y.mean().item()
                        # print(f"[ACT2.Box] z[min/mean/max]={z_min:.3f}/{z_mean:.3f}/{z_max:.3f} "
                        #       f"y[min/mean/max]={y_min:.6f}/{y_mean:.6f}/{y_max:.6f}")
                return y, action_log_probs
            else:
                actions = z
                action_log_probs = logits.log_probs(actions)           # [B,1]
                return actions, action_log_probs

        # Discrete
        A = self.action_dims[0]
        mask = self._prep_discrete_mask(available_actions, B, A, x.device)
        action_logits = self.action_out(x, mask)
        actions = action_logits.mode() if deterministic else action_logits.sample()
        action_log_probs = action_logits.log_probs(actions)  # [B,1]
        return actions, action_log_probs

    # ============================== 概率（可选） ============================== #
    def get_probs(self, x, available_actions=None):
        """
        返回动作概率/均值：
        - MultiDiscrete: 按头取 probs 后在最后一维拼接 -> [B, sum(nvec)]
        - Discrete/MultiBinary: 返回底层分布 probs
        - Box: 返回均值（若启用 [0,1] 压缩，则返回 Sigmoid(+仿射) 后的均值，仅作观测显示）
        """
        B = x.size(0)

        if self.mixed_action:
            probs = []
            for action_out in self.action_outs:
                probs.append(action_out(x).probs)
            return torch.cat(probs, dim=-1)

        if self.multi_discrete:
            masks = self.normalize_available_actions(available_actions)
            probs = []
            for h, action_out in enumerate(self.action_outs):
                mask_h = masks[h] if isinstance(masks, list) else None
                if isinstance(mask_h, torch.Tensor):
                    mask_h = mask_h.to(x.device, non_blocking=True)
                    mask_h = mask_h[:, : self.action_dims[h]]
                probs.append(action_out(x, mask_h).probs)  # [B, nvec[h]]
            return torch.cat(probs, dim=-1)

        if self.continuous_action:
            logits = self.action_out(x)
            p = logits.probs  # 通常为高斯均值（z-空间的）
            if getattr(self, "_squash_to_unit", False):
                # 仅用于监控显示：对均值做同样的非线性（并非严格意义的 E[sigmoid(Z))]）
                y0 = torch.sigmoid(p)
                eps_aff = getattr(self, "_eps_affine", 0.0)
                if eps_aff > 0.0:
                    y = eps_aff + (1.0 - 2.0 * eps_aff) * y0
                else:
                    y = y0
                return y.clamp(self._eps, 1.0 - self._eps)
            return p

        # Discrete
        A = self.action_dims[0]
        mask = self._prep_discrete_mask(available_actions, B, A, x.device)
        return self.action_out(x, mask).probs

    # ============================== 评估（训练用） ============================== #
    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        """
        返回：
          action_log_probs: [B,1] —— MultiDiscrete 对各头 log_prob 求和，其他与底层一致
          dist_entropy:     标量（按 batch 平均）；MultiDiscrete 对各头熵做平均
        """
        B = x.size(0)

        if self.mixed_action:
            logps, ents = [], []
            a_cont, a_disc = action[..., :-1], action[..., -1:].long()
            parts = [a_cont, a_disc]
            for action_out, act in zip(self.action_outs, parts):
                logits = action_out(x)
                lp = logits.log_probs(act)
                ent = logits.entropy()
                if active_masks is not None:
                    ent = (ent * active_masks.squeeze(-1)).sum() / active_masks.sum().clamp(min=1e-6)
                else:
                    ent = ent.mean()
                logps.append(lp if lp.dim() == 2 else lp.unsqueeze(-1))
                ents.append(ent)
            action_log_probs = torch.cat(logps, dim=-1).sum(dim=-1, keepdim=True)
            dist_entropy = torch.stack(ents).mean()
            return action_log_probs, dist_entropy

        if self.multi_discrete:
            # action: [B, H] 或 [B, H, 1]
            if action.dim() == 3 and action.size(-1) == 1:
                action = action.squeeze(-1)
            assert action.dim() == 2 and action.size(-1) == self.n_heads, \
                f"MultiDiscrete 动作形状应为 [B,H={self.n_heads}]，但得到 {tuple(action.shape)}"

            masks = self.normalize_available_actions(available_actions)

            logp_list, ent_list = [], []
            for h, action_out in enumerate(self.action_outs):
                m_h = masks[h] if isinstance(masks, list) else None
                if isinstance(m_h, torch.Tensor):
                    m_h = m_h.to(x.device, non_blocking=True)
                    m_h = m_h[:, : self.action_dims[h]]
                logits = action_out(x, m_h)
                a_h = action[:, h].unsqueeze(-1).long()           # [B,1]
                logp_list.append(logits.log_probs(a_h))           # [B,1]
                ent_h = logits.entropy()                          # [B,1] 或 [B]
                if ent_h.dim() == 2 and ent_h.size(-1) == 1:
                    ent_h = ent_h.squeeze(-1)                     # [B]
                if active_masks is not None:
                    ent_h = (ent_h * active_masks.squeeze(-1)).sum() / active_masks.sum().clamp(min=1e-6)
                else:
                    ent_h = ent_h.mean()
                ent_list.append(ent_h)

            action_log_probs = torch.cat(logp_list, dim=-1).sum(dim=-1, keepdim=True)  # [B,1]
            dist_entropy = torch.stack(ent_list).mean()
            return action_log_probs, dist_entropy

        if self.continuous_action:
            logits = self.action_out(x)
            if getattr(self, "_squash_to_unit", False):
                # 动作来自环境/回放缓冲：y∈[0,1]，需反变换到 z 再带雅可比修正
                z, y0, scale = self._unsquash_to_logit(action)
                action_log_probs = self._logprob_after_sigmoid(logits, z, action)  # [B,1]

                # 熵：使用底层高斯熵的和（简单稳定，常见做法）
                ent = logits.entropy()
                if ent.dim() == 2 and ent.size(-1) > 1:
                    ent = ent.sum(-1)  # -> [B]
                elif ent.dim() == 2 and ent.size(-1) == 1:
                    ent = ent.squeeze(-1)  # -> [B]
                if active_masks is not None:
                    am = active_masks.squeeze(-1)  # [B]
                    ent = (ent * am).sum() / am.sum().clamp(min=1e-6)
                else:
                    ent = ent.mean()
                return action_log_probs, ent
            else:
                action_log_probs = logits.log_probs(action)  # [B,1]
                ent = logits.entropy()  # [B, action_dim] 或 [B,1]
                if ent.dim() == 2 and ent.size(-1) > 1:
                    ent = ent.sum(-1)  # -> [B]
                elif ent.dim() == 2 and ent.size(-1) == 1:
                    ent = ent.squeeze(-1)  # -> [B]
                if active_masks is not None:
                    ent = (ent * active_masks.squeeze(-1)).sum() / active_masks.sum().clamp(min=1e-6)
                else:
                    ent = ent.mean()
                return action_log_probs, ent

        # Discrete
        A = self.action_dims[0]
        mask = self._prep_discrete_mask(available_actions, B, A, x.device)
        logits = self.action_out(x, mask)
        action_log_probs = logits.log_probs(action)  # [B,1]
        ent = logits.entropy()
        if active_masks is not None:
            ent = (ent * active_masks.squeeze(-1)).sum() / active_masks.sum().clamp(min=1e-6)
        else:
            ent = ent.mean()
        return action_log_probs, ent
