"""Hyena 算子 + Hyena Block —— 基于 safari (HazyResearch/safari) 官方实现。

官方源码: /tmp/neu_tmp/safari/standalone_hyena.py + src/models/sequence/hyena.py
论文: Hyena Hierarchy (arXiv:2302.10866)

核心思想: 用「隐式长卷积 (FFT 快速卷积) + 逐元素门控」的递归替代 softmax 注意力。
  Hyena_N: y = x^N · (h^N ∗ (x^{N-1} · (h^{N-1} ∗ (··· v))))
  - 滤波器 h^n 由 FFN(位置编码) + 指数衰减窗隐式生成，参数与序列长度解耦
  - 长卷积用 FFT 实现 O(L log L)，因果性靠 2L 零填充 + 截断保证

统一接口：Model(Config).forward(idx) -> logits[B, T, vocab]
"""
import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fftconv(u, k, D):
    """FFT 长卷积：y = causal_conv(k, u) + u * D（残差 + 可学习 bias 项）。"""
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


class Sin(nn.Module):
    """周期激活 sin(w·x)，w 可学习（高频内容，缓解低频偏置）。"""

    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = (
            nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)
        )

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(nn.Module):
    """复数指数位置编码（实数化）：z = [t, cos(f·ω), -sin(f·ω)]，可学习。"""

    def __init__(self, emb_dim, seq_len, lr_pos_emb=1e-5):
        super().__init__()
        assert emb_dim % 2 != 0 and emb_dim >= 3, "emb_dim 必须为 >=3 的奇数 (t, cos, sin)"
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        bands = (emb_dim - 1) // 2
        self.register_buffer("t", torch.linspace(0, 1, seq_len)[None, :, None])  # 1,L,1
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]  # 1,L,1
        omega = 2 * math.pi * t_rescaled / seq_len  # 1,L,1
        f = torch.linspace(1e-4, bands - 1, bands)[None, None]  # 1,1,bands
        z_cos = torch.cos(f * omega)  # 1,L,bands
        z_sin = -torch.sin(f * omega)  # 1,L,bands
        z = torch.cat([torch.ones_like(self.t), z_cos, z_sin], dim=-1)  # 1,L,1+2bands
        self.z = nn.Parameter(z)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(nn.Module):
    """指数衰减窗：不同通道不同衰减率，让滤波器长度多样化。"""

    def __init__(self, d_model, fast_decay_pct=0.3, slow_decay_pct=1.5,
                 target=1e-2, shift=0.0, modulate=True):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register_buffer("deltas", deltas)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


class HyenaFilter(nn.Module):
    """隐式长滤波器：h(t) = Window(t) · FFN(位置编码(t))，参数与序列长度解耦。"""

    def __init__(self, d_model, emb_dim=3, order=16, seq_len=1024, num_inner_mlps=2,
                 w=10, bias=True, dropout=0.0, **modulation_args):
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.randn(d_model))
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len)
        act = Sin(dim=order, w=w)
        layers = [nn.Linear(emb_dim, order), act]
        for _ in range(num_inner_mlps):
            layers.append(nn.Linear(order, order))
            layers.append(act)
        layers.append(nn.Linear(order, d_model, bias=False))
        self.implicit_filter = nn.Sequential(*layers)
        self.modulation = ExponentialModulation(d_model, **modulation_args)

    def filter(self, L):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None):
        if k is None:
            k = self.filter(L)
        k = k[0] if type(k) is tuple else k
        if bias is None:
            bias = self.bias
        bias = bias if self.use_bias else 0 * bias
        return fftconv(x, k, bias)


class HyenaOperator(nn.Module):
    """Hyena 算子：short-conv 投影 → 门控 × 长卷积 递归 → 输出。"""

    def __init__(self, d_model, l_max, order=2, filter_order=64, dropout=0.0,
                 filter_dropout=0.0, **filter_args):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model, d_model)
        # 显式短卷积（depthwise causal conv k=3），从输入生成门控
        self.short_filter = nn.Conv1d(
            inner_width, inner_width, 3, padding=2, groups=inner_width
        )
        self.filter_fn = HyenaFilter(
            d_model * (order - 1), order=filter_order, seq_len=l_max,
            dropout=filter_dropout, **filter_args
        )

    def forward(self, u):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        u = self.in_proj(u)
        u = u.transpose(1, 2)  # B,L,D -> B,D,L
        uc = self.short_filter(u)[..., :l_filter]  # causal 截断
        xs = uc.split(self.d_model, dim=1)  # (order+1) 份 [B,D,L]
        *x, v = xs  # x: order 份门控, v: 值
        k = self.filter_fn.filter(l_filter)[0]  # L, d*(order-1)
        k = k.reshape(l_filter, self.order - 1, self.d_model).permute(1, 2, 0)  # o,d,L
        bias = self.filter_fn.bias.reshape(self.order - 1, self.d_model)  # o,d
        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)  # 门控（reversed: 从最深阶开始）
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])  # 长卷积
        y = (v * x[0]).transpose(1, 2)  # B,L,D
        return self.out_proj(y)


class MLP(nn.Module):
    """标准 GELU FFN（Hyena 论文的 block FFN）。"""

    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class HyenaBlock(nn.Module):
    """Hyena Block：pre-norm [HyenaOp] + pre-norm [FFN] 双残差。"""

    def __init__(self, hidden_dim, l_max, order=2, filter_order=64, dropout=0.0,
                 ffn_mult=4, eps=1e-6, **op_args):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.hyena = HyenaOperator(hidden_dim, l_max, order=order,
                                   filter_order=filter_order, dropout=dropout, **op_args)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = MLP(hidden_dim, ffn_mult * hidden_dim)

    def forward(self, x):
        r = x
        x = x + self.hyena(self.norm1(x))
        x = x + self.ffn(self.norm2(r))
        return x


@dataclasses.dataclass
class Config:
    hidden_size: int = 212
    layers: int = 6
    order: int = 2  # Hyena 递归深度
    filter_order: int = 64  # 隐式滤波器 FFN 宽度
    ffn_mult: int = 4
    dropout: float = 0.0
    vocab_size: int = 65
    context_length: int = 256


class Hyena(nn.Module):
    """Hyena 语言模型：token+pos embedding → HyenaBlocks → final norm → head。"""

    def __init__(self, config: Config):
        super().__init__()
        c = config
        d = c.hidden_size
        self.token_embedding = nn.Embedding(c.vocab_size, d)
        self.position_embedding = nn.Embedding(c.context_length, d)
        self.blocks = nn.ModuleList(
            [
                HyenaBlock(d, c.context_length, order=c.order, filter_order=c.filter_order,
                           dropout=c.dropout, ffn_mult=c.ffn_mult)
                for _ in range(c.layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, c.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
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
    m = Hyena(cfg).cuda()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_length)).cuda()
    logits = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params={n_params:,}  logits={tuple(logits.shape)}  "
          f"loss={F.cross_entropy(logits.reshape(-1, cfg.vocab_size), x.reshape(-1)).item():.3f}")
