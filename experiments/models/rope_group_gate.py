"""CutDeep-Norm with a low-cost nonlinear gate aligned to RoPE pairs."""
import torch
import torch.nn.functional as F
from torch import nn

import bdh_best
from gram_norm import GramAttentionNorm


class RoPEGroupGateBlock(bdh_best.BDHBestBlock):
    """Replace coordinate-wise Q/K ReLU with shared frequency-group gates."""

    def __init__(self, config, groups=8):
        super().__init__(config, recurrences=1)
        pairs = self.qk_width // 2
        if self.qk_width % 2 or groups > pairs:
            raise ValueError("qk_width must be even and contain at least one pair per group")
        self.groups = groups
        self.heads = config.heads
        self.gate_proj = nn.Linear(config.hidden_size, config.heads * groups, bias=False)
        nn.init.zeros_(self.gate_proj.weight)
        pair_groups = torch.arange(pairs) * groups // pairs
        group_expand = F.one_hot(pair_groups, num_classes=groups).T.to(torch.float32)
        self.register_buffer("group_expand", group_expand, persistent=False)
        self.attn = GramAttentionNorm(self.qk_width, posify=False)

    def _project_qk(self, enc):
        batch, length, _ = enc.shape
        q = (enc @ self.query_proj).view(batch, length, self.heads, -1, 2)
        k = (enc @ self.key_proj).view(batch, length, self.heads, -1, 2)
        gates = 2.0 * self.gate_proj(enc).sigmoid()
        gates = gates.view(batch, length, self.heads, self.groups)
        # Fixed matrix expansion avoids advanced indexing in compiled Triton kernels.
        gates = gates @ self.group_expand.to(dtype=gates.dtype)
        gates = gates.unsqueeze(-1)
        return (q * gates).flatten(-2), (k * gates).flatten(-2)


class RawSignedGramAttention(bdh_best.GramAttention):
    """Explicit causal signed Gram aggregation without a degree denominator."""

    def forward(self, q, k, v, use_rope=True, normalize=False):
        length = q.size(-2)
        positions = torch.arange(length, device=q.device, dtype=self.freqs.dtype)
        phases = positions.view(1, 1, -1, 1) * self.freqs
        q = bdh_best.rope(phases, q) if use_rope else q
        k = bdh_best.rope(phases, k) if use_rope else k
        if normalize:
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        scores = q @ k.mT
        mask = torch.ones(length, length, dtype=torch.bool, device=q.device).tril(-1)
        return scores.masked_fill(~mask, 0) @ v


class SignedExpGramAttention(GramAttentionNorm):
    """Degree-normalized message followed by bounded signed exponential shaping."""

    def forward(self, q, k, v, use_rope=True, normalize=False):
        message = super().forward(q, k, v, use_rope=use_rope, normalize=normalize)
        return message.sign() * (-torch.expm1(-message.abs()))


class RoPEGroupGate(bdh_best.BDHBest):
    def __init__(self, config, groups=8):
        super().__init__(config)
        self.blocks = nn.ModuleList(
            [RoPEGroupGateBlock(config, groups=groups) for _ in range(config.layers)]
        )


class RoPEGroupGateRaw(RoPEGroupGate):
    """RoPEGate8 with explicit signed aggregation and no degree denominator."""

    def __init__(self, config, groups=8, learned_absolute=True):
        super().__init__(config, groups=groups)
        for block in self.blocks:
            block.attn = RawSignedGramAttention(block.qk_width)
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


class RoPEGroupGateSignedExp(RoPEGroupGate):
    """RoPEGate8 with degree normalization and bounded signed exponential messages."""

    def __init__(self, config, groups=8, learned_absolute=True):
        super().__init__(config, groups=groups)
        for block in self.blocks:
            block.attn = SignedExpGramAttention(block.qk_width, posify=False)
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


class RoPEGroupGateNoAbs(RoPEGroupGate):
    """RoPEGroupGate using only graph-internal RoPE for position information."""

    def __init__(self, config, groups=8):
        super().__init__(config, groups=groups)
        del self.position_embedding

    def forward(self, idx):
        hidden = self.input_norm(self.token_embedding(idx)).unsqueeze(1)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.final_norm(hidden.squeeze(1)))
