"""Small autoregressive GDN-2 model for the Sudoku-9 comparison.

Uses NVIDIA's official GatedDeltaNet2 token mixer and Triton chunk kernel.
No learned positional embedding: temporal order is represented by the recurrent
state, channel-wise decay, and causal short convolutions.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from lit_gpt.gdn2 import GatedDeltaNet2


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inner = math.ceil((8 * dim / 3) / 8) * 8
        self.gate_up = nn.Linear(dim, 2 * inner, bias=False)
        self.down = nn.Linear(inner, dim, bias=False)

    def forward(self, x):
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class GDN2Block(nn.Module):
    def __init__(self, dim, heads, head_dim, layer_idx):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.mixer = GatedDeltaNet2(
            hidden_size=dim,
            head_dim=head_dim,
            num_heads=heads,
            num_v_heads=heads,
            mode="chunk",
            use_short_conv=True,
            conv_size=4,
            layer_idx=layer_idx,
        )
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim)

    def forward(self, x):
        message, _, _ = self.mixer(self.norm1(x), use_cache=False)
        x = x + message
        return x + self.mlp(self.norm2(x))


class GDN2Sudoku(nn.Module):
    def __init__(self, vocab_size=11, hidden_size=160, layers=8, heads=5, head_dim=32):
        super().__init__()
        assert heads * head_dim == hidden_size
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList(
            [GDN2Block(hidden_size, heads, head_dim, i) for i in range(layers)]
        )
        self.final_norm = RMSNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)

    def forward(self, idx):
        x = self.token_embedding(idx)
        for block in self.blocks:
            x = block(x)
        return self.output(self.final_norm(x))


def count_params(hidden_size, layers, heads, head_dim):
    model = GDN2Sudoku(11, hidden_size, layers, heads, head_dim)
    return sum(p.numel() for p in model.parameters())
