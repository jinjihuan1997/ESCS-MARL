# Filepath: algorithms/on_policy/utils/distributions.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from .util import init

"""
Standardized distribution wrappers:
- FixedCategorical:  sample/mode return shape [B,1]; log_probs returns [B,1]
- FixedNormal:       Gaussian with Diag covariance; log_probs sums over action dims -> [B,1]
- FixedBernoulli:    Multi-binary; log_probs sums over dims -> [B,1]

Heads:
- Categorical:  linear -> FixedCategorical (supports available_actions mask)
- DiagGaussian: fc_mean + learned logstd -> FixedNormal
- Bernoulli:    linear -> FixedBernoulli
"""


# ----------------------------- Fixed wrappers ----------------------------- #

class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        # [B] -> [B,1]
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        # actions: [B,1] (long). Torch Categorical expects [B]
        # Return shape [B,1]
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        # [B,1]
        return self.probs.argmax(dim=-1, keepdim=True)


class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        # Sum over action dims -> [B,1]
        return super().log_prob(actions).sum(-1, keepdim=True)

    def mode(self):
        return self.mean


class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        # [B, dim] -> [B,1]
        return super().log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        # [B]
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


# ----------------------------- Heads (nn.Modules) ----------------------------- #

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.action_dim = int(num_outputs)
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        """
        x:                 [B, num_inputs]
        available_actions: [B, num_outputs] 或 [num_outputs] 的 0/1 掩码，或 None
        return: FixedCategorical over num_outputs
        """
        logits = self.linear(x)

        if available_actions is not None:
            # to tensor & move to same device as logits
            if not torch.is_tensor(available_actions):
                mask_src = torch.as_tensor(available_actions, device=logits.device)
            else:
                mask_src = available_actions.to(device=logits.device)

            # 支持 [A] 或 [B,A]；其余形状直接尝试广播到 [B,A]
            if mask_src.dim() == 1:
                # [A] -> [B,A]
                if mask_src.size(0) != logits.size(1):
                    raise ValueError(f"available_actions dim mismatch: expect {logits.size(1)}, got {mask_src.size(0)}")
                mask_src = mask_src.unsqueeze(0).expand(logits.size(0), -1).contiguous()
            elif mask_src.dim() == 2:
                if mask_src.size(0) != logits.size(0) or mask_src.size(1) != logits.size(1):
                    # 尝试广播（如 batch 维为 1 的情况）
                    try:
                        mask_src = mask_src.expand_as(logits)
                    except Exception:
                        raise ValueError(
                            f"available_actions shape {tuple(mask_src.shape)} not broadcastable to logits {tuple(logits.shape)}"
                        )
            else:
                raise ValueError(f"available_actions must be [A] or [B,A], got {tuple(mask_src.shape)}")

            # 允许 float/bool/int；<=0 视为不可选
            mask_bool = (mask_src <= 0).to(dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(mask_bool, -1e10)

        return FixedCategorical(logits=logits)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(DiagGaussian, self).__init__()
        self.action_dim = int(num_outputs)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        # Create zeros on the same device/dtype as action_mean
        zeros = torch.zeros_like(action_mean)
        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.action_dim = int(num_outputs)
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        logits = self.linear(x)
        return FixedBernoulli(logits=logits)


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        # keep device & dtype
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1).to(dtype=x.dtype, device=x.device)
        else:
            bias = self._bias.t().view(1, -1, 1, 1).to(dtype=x.dtype, device=x.device)
        return x + bias
