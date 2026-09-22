"""LIV 卷积 block（Gated-Short-Conv）+ 对齐的语言模型包装。

LIVOp / SwiGLUFFN / LIVBlock 为原始实现原样保留：
- LIVOp: in_proj → 三份 [B,D,3H] → gateB*h_tilde → 因果 depth-wise Conv1d(k=3, 左padding) → gateC → out_proj
- LIVBlock: [norm → LIVOp → 残差] + [norm → SwiGLU FFN → 残差]

统一接口：Model(Config).forward(idx) -> logits[B, T, vocab]
（注意 LIV 内部 x 为 [B, seq_len, hidden]，即 B=batch, D=seq, H=hidden）
"""
import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class LIVOp(nn.Module):
    """
    LIV 算子（Gated‑Short‑Conv，仅卷积+双门控，不含Norm/FFN）
    Input:  x: [B,D,H]
    Output: y: [B,D,H]
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.H = hidden_dim
        self.in_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        # causal depth‑wise conv1d kernel=3
        self.dw_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=2,
            groups=hidden_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        B, D, H = x.shape
        proj = self.in_proj(x)  # [B,D,3H]
        gateB, gateC, h_tilde = torch.chunk(proj, 3, dim=-1)

        y = gateB * h_tilde  # first gate B
        y = y.transpose(1, 2)  # B,D,H → B,H,D for Conv1d
        y = self.dw_conv(y)  # [B,H,D+2]
        y = y[..., :D]  # causal truncate
        y = y.transpose(1, 2)  # [B,D,H]

        z = gateC * y  # second gate C
        out = self.out_proj(z)
        return out


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w3 = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class LIVBlock(nn.Module):
    """
    LFM2 完整LIV卷积Block（论文conv‑block）
    Input x: [B,D,H]
    Output y: [B,D,H]
    """
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.liv_op = LIVOp(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        self.ffn = SwiGLUFFN(hidden_dim, intermediate_dim)

    def forward(self, x: torch.Tensor):
        # sub‑block 1: LIV convolution
        r1 = x
        x = self.norm1(x)
        x = self.liv_op(x)
        x = x + r1

        # sub‑block 2: SwiGLU FFN
        r2 = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + r2
        return x


@dataclasses.dataclass
class Config:
    hidden_size: int = 212
    intermediate_mult: int = 4  # SwiGLU FFN 中间维度 = mult * hidden
    layers: int = 6
    vocab_size: int = 65
    context_length: int = 256


class LIV(nn.Module):
    """LIV 语言模型：token+pos embedding → LIVBlocks → final norm → head。"""

    def __init__(self, config: Config):
        super().__init__()
        c = config
        d = c.hidden_size
        self.token_embedding = nn.Embedding(c.vocab_size, d)
        self.position_embedding = nn.Embedding(c.context_length, d)
        self.blocks = nn.ModuleList(
            [LIVBlock(d, c.intermediate_mult * d) for _ in range(c.layers)]
        )
        self.final_norm = RMSNorm(d)
        self.output = nn.Linear(d, c.vocab_size, bias=False)
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
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.final_norm(hidden))


if __name__ == "__main__":
    cfg = Config()
    m = LIV(cfg).cuda()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_length)).cuda()
    logits = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params={n_params:,}  logits={tuple(logits.shape)}  "
          f"loss={F.cross_entropy(logits.reshape(-1, cfg.vocab_size), x.reshape(-1)).item():.3f}")
