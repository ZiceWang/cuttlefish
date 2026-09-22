"""标准 Transformer —— pre-norm 缩放点积注意力 + SwiGLU FFN。

对应 experiments/bdh_sudoku4/train_transformer_sudoku4.py 的 TransformerBlock，
并加上 token+position embedding（与 BDH 对齐，便于公平对比）。

统一接口：Model(Config).forward(idx) -> logits[B, T, vocab]
"""
import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, value):
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(value.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        internal = math.ceil((8 * hidden_size / 3) / 8) * 8
        self.gate_up = nn.Linear(hidden_size, 2 * internal, bias=False)
        self.down = nn.Linear(internal, hidden_size, bias=False)

    def forward(self, value):
        gate, up = self.gate_up(value).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.attention_norm = RMSNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.attention_output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = SwiGLU(hidden_size)

    def forward(self, hidden):
        batch, length, _ = hidden.shape
        h = self.attention_norm(hidden)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q, k, v = [
            t.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
            for t in (q, k, v)
        ]
        message = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        message = message.transpose(1, 2).reshape(batch, length, hidden.shape[-1])
        hidden = hidden + self.attention_output(message)
        return hidden + self.ffn(self.ffn_norm(hidden))


@dataclasses.dataclass
class Config:
    hidden_size: int = 400
    heads: int = 4
    layers: int = 5
    vocab_size: int = 65
    context_length: int = 256


class Transformer(nn.Module):
    """对齐 BDH 的 token+position embedding + pre-norm 堆叠。"""

    def __init__(self, config: Config):
        super().__init__()
        c = config
        assert c.hidden_size % c.heads == 0
        self.token_embedding = nn.Embedding(c.vocab_size, c.hidden_size)
        self.position_embedding = nn.Embedding(c.context_length, c.hidden_size)
        self.input_norm = RMSNorm(c.hidden_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(c.hidden_size, c.heads) for _ in range(c.layers)]
        )
        self.final_norm = RMSNorm(c.hidden_size)
        self.output = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        positions = torch.arange(idx.size(1), device=idx.device)
        hidden = self.token_embedding(idx) + self.position_embedding(positions)
        hidden = self.input_norm(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.final_norm(hidden))


if __name__ == "__main__":
    cfg = Config()
    m = Transformer(cfg).cuda()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_length)).cuda()
    logits = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params={n_params:,}  logits={tuple(logits.shape)}  "
          f"loss={F.cross_entropy(logits.reshape(-1, cfg.vocab_size), x.reshape(-1)).item():.3f}")
