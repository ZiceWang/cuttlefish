"""原始 BDH —— 单共享编码器 + 无 softmax Gram 图 + 乘积绑定。

对应 repo experiments/bdh_sudoku4/bdh.py 的 BDH 类（core_mode=sparse_binding,
sparse_binding=product, relation_mode=gram），也是你贴的 BDH_GPU 的参数化版本。

统一接口：Model(Config).forward(idx) -> logits[B, T, vocab]
"""
import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return 1.0 / (theta ** (quantize(torch.arange(0, n, dtype=dtype)) / n)) / (2 * math.pi)


def rope(phases, value):
    """标准 RoPE：相邻 pair 旋转。同 repo bdh.py Attention.rotate。"""
    phases = phases[..., ::2]
    phases_cos = torch.cos((phases % 1) * (2 * math.pi))
    phases_sin = torch.sin((phases % 1) * (2 * math.pi))
    paired = value.unflatten(-1, (-1, 2))
    even, odd = paired.unbind(dim=-1)
    return torch.stack(
        (
            (even * phases_cos).to(value.dtype) - (odd * phases_sin).to(value.dtype),
            (odd * phases_cos).to(value.dtype) + (even * phases_sin).to(value.dtype),
        ),
        dim=-1,
    ).flatten(-2)


class GramAttention(nn.Module):
    """无 softmax 的 Gram 内核：scores = q @ k^T，tril(-1) 严格因果，无归一化。"""

    def __init__(self, feature_width, theta=2**16):
        super().__init__()
        freqs = get_freqs(feature_width, theta=theta, dtype=torch.float32)
        self.register_buffer("freqs", freqs.view(1, 1, 1, feature_width), persistent=False)

    def forward(self, q, k, v, use_rope=True):
        t = q.size(-2)
        positions = torch.arange(t, device=q.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)
        phases = positions * self.freqs
        qr = rope(phases, q) if use_rope else q
        kr = rope(phases, k) if use_rope else k
        scores = qr @ kr.mT  # [B, H, T, T] 未缩放、未归一化
        mask = torch.ones(t, t, dtype=torch.bool, device=q.device).tril(diagonal=-1)
        return scores.masked_fill(~mask, 0) @ v  # 不含自身


@dataclasses.dataclass
class Config:
    hidden_size: int = 256
    heads: int = 4
    neurons: int = 32768  # 总稀疏神经元（每头 neurons//heads）
    layers: int = 6
    vocab_size: int = 65
    context_length: int = 256
    dropout: float = 0.05
    use_rope: bool = True
    init_std: float = 0.02


class BDHOriginal(nn.Module):
    """原始 BDH：q = k = 共享 ReLU 特征，v = 原始残差，乘积绑定读出。"""

    def __init__(self, config: Config):
        super().__init__()
        c = config
        d, nh = c.hidden_size, c.heads
        assert c.neurons % nh == 0
        n = c.neurons // nh  # 每头神经元
        self.ln = nn.LayerNorm(d, elementwise_affine=False, bias=False)
        self.wte = nn.Embedding(c.vocab_size, d)
        self.drop = nn.Dropout(c.dropout)
        self.encoder = nn.Parameter(torch.zeros((c.neurons, d)).normal_(std=c.init_std))
        self.decoder_x = nn.Parameter(torch.zeros((nh, d, n)).normal_(std=c.init_std))
        self.decoder_y = nn.Parameter(torch.zeros((nh, d, n)).normal_(std=c.init_std))
        self.readout = nn.Parameter(torch.zeros((d, c.vocab_size)).normal_(std=c.init_std))
        self.attn = GramAttention(n)
        self.n_layers = c.layers

    def forward(self, idx):
        B, T = idx.size()
        v_ast = self.ln(self.wte(idx).unsqueeze(1))  # B,1,T,D

        for _ in range(self.n_layers):
            x = F.relu(v_ast @ self.decoder_x)  # B,H,T,n（共享编码特征）
            a_ast = self.attn(q=x, k=x, v=v_ast)  # q=k=特征, v=原始残差
            y = F.relu(self.ln(a_ast) @ self.decoder_y) * x  # 读出 + 乘积绑定
            y = y.transpose(1, 2).reshape(B, 1, T, self.decoder_x.shape[0] * x.size(-1))
            y = self.drop(y)
            v_ast = v_ast + self.ln(y @ self.encoder)  # 解码回 D + 残差
            v_ast = self.ln(v_ast)

        return v_ast.squeeze(1) @ self.readout  # B,T,vocab


if __name__ == "__main__":
    cfg = Config()
    m = BDHOriginal(cfg).cuda()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_length)).cuda()
    logits = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params={n_params:,}  logits={tuple(logits.shape)}  "
          f"loss={F.cross_entropy(logits.reshape(-1, cfg.vocab_size), x.reshape(-1)).item():.3f}")
