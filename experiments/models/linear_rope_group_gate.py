"""Linear-time RoPEGate8 with signed messages and separable scale normalization."""
import torch
import torch.nn.functional as F
from torch import nn

import bdh_best
from rope_group_gate import RoPEGroupGateBlock


class LinearNormalizedGram(nn.Module):
    """Exact prefix implementation of a signed Gram numerator and L1 upper-bound norm.

    numerator_t = q_t^T sum_{s<t} k_s v_s^T
    denominator_t = |q_t|^T sum_{s<t} |k_s|

    The denominator upper-bounds sum_s |q_t^T k_s| by the triangle inequality and
    is exactly separable, unlike the original pairwise absolute-degree divisor.
    """

    def __init__(self, feature_width, theta=2**16):
        super().__init__()
        freqs = bdh_best.get_freqs(feature_width, theta=theta, dtype=torch.float32)
        self.register_buffer("freqs", freqs.view(1, 1, 1, feature_width), persistent=False)

    def _rotate_normalize(self, q, k):
        length = q.size(-2)
        positions = torch.arange(length, device=q.device, dtype=self.freqs.dtype)
        phases = positions.view(1, 1, -1, 1) * self.freqs
        return (
            F.normalize(bdh_best.rope(phases, q), dim=-1),
            F.normalize(bdh_best.rope(phases, k), dim=-1),
        )

    def forward(self, q, k, v, use_rope=True, normalize=False):
        if v.dim() == 4 and v.size(1) == 1:
            v = v.squeeze(1)
        if use_rope:
            q, k = self._rotate_normalize(q, k)
        elif normalize:
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        value = v.unsqueeze(1).to(k.dtype)
        outer = k.unsqueeze(-1) * value.unsqueeze(-2)
        state = outer.cumsum(dim=2)
        state = torch.cat((torch.zeros_like(state[:, :, :1]), state[:, :, :-1]), dim=2)
        numerator = (q.unsqueeze(-1) * state).sum(dim=-2)

        key_scale = k.abs().cumsum(dim=2)
        key_scale = torch.cat(
            (torch.zeros_like(key_scale[:, :, :1]), key_scale[:, :, :-1]), dim=2
        )
        denominator = (q.abs() * key_scale).sum(dim=-1, keepdim=True).clamp_min(1e-2)
        return (numerator / denominator).to(v.dtype)

    def quadratic_reference(self, q, k, v, use_rope=True, normalize=False):
        """O(T^2) reference for numerical equivalence tests only."""
        if v.dim() == 4 and v.size(1) == 1:
            v = v.squeeze(1)
        if use_rope:
            q, k = self._rotate_normalize(q, k)
        elif normalize:
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        length = q.size(-2)
        mask = torch.ones(length, length, dtype=torch.bool, device=q.device).tril(-1)
        scores = (q @ k.mT).masked_fill(~mask, 0)
        numerator = torch.einsum("bhts,bsd->bhtd", scores, v)
        key_scale = torch.einsum("ts,bhsf->bhtf", mask.to(k.dtype), k.abs())
        denominator = (q.abs() * key_scale).sum(dim=-1, keepdim=True).clamp_min(1e-2)
        return (numerator / denominator).to(v.dtype)


class LinearSignedGram(LinearNormalizedGram):
    """Exact causal signed Gram numerator; scale is handled by downstream LayerNorm."""

    def forward(self, q, k, v, use_rope=True, normalize=False):
        if v.dim() == 4 and v.size(1) == 1:
            v = v.squeeze(1)
        if use_rope:
            q, k = self._rotate_normalize(q, k)
        elif normalize:
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        value = v.unsqueeze(1).to(k.dtype)
        state = (k.unsqueeze(-1) * value.unsqueeze(-2)).cumsum(dim=2)
        state = torch.cat((torch.zeros_like(state[:, :, :1]), state[:, :, :-1]), dim=2)
        return (q.unsqueeze(-1) * state).sum(dim=-2).to(v.dtype)


class LinearRoPEGroupGateBlock(RoPEGroupGateBlock):
    def __init__(self, config, groups=8, scale_normalization=True):
        super().__init__(config, groups=groups)
        cls = LinearNormalizedGram if scale_normalization else LinearSignedGram
        self.attn = cls(self.qk_width)


class LinearRoPEGroupGate(bdh_best.BDHBest):
    def __init__(self, config, groups=8, learned_absolute=True, scale_normalization=True):
        super().__init__(config)
        self.blocks = nn.ModuleList(
            [LinearRoPEGroupGateBlock(config, groups=groups,
                                      scale_normalization=scale_normalization)
             for _ in range(config.layers)]
        )
        if not learned_absolute:
            del self.position_embedding

    def forward(self, idx):
        hidden = self.token_embedding(idx)
        if hasattr(self, "position_embedding"):
            positions = torch.arange(idx.size(1), device=idx.device)
            hidden = hidden + self.position_embedding(positions)
        hidden = self.input_norm(hidden).unsqueeze(1)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.final_norm(hidden.squeeze(1)))
