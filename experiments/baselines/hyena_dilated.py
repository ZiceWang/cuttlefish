"""HyenaDilated —— Hyena 但把 FFT 长卷积换成空洞卷积金字塔。

保留 Hyena 全部结构（in_proj 门控 + short_filter + 多阶递归 + out_proj），
仅把 filter_fn(隐式核 + FFT conv) 替换为 N 层 dilated causal depthwise conv
（dilation 1,2,4,...），纯实数、compile 友好、可流式。
"""
import dataclasses

import torch
import torch.nn.functional as F
from torch import nn

import hyena


class DilatedCausalConv(nn.Module):
    """N 层 dilated causal depthwise conv，感受野 = 1 + Σ(d_i)。"""

    def __init__(self, dim, n_layers=5):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=3, padding=2 ** i * 2, dilation=2 ** i,
                      groups=dim, bias=False)
            for i in range(n_layers)
        ])
        self.silu = nn.SiLU()

    def forward(self, u):
        # u: [B, D, L]
        length = u.size(-1)
        for conv in self.convs:
            u = conv(u)[..., :length]  # causal 截断
            u = self.silu(u)
        return u


class DilatedHyenaOperator(hyena.HyenaOperator):
    """覆盖长卷积部分：filter_fn+fftconv → DilatedCausalConv。"""

    def __init__(self, d_model, l_max, order=2, filter_order=64, dropout=0.0, **kw):
        super().__init__(d_model, l_max, order=order, filter_order=filter_order,
                         dropout=dropout, **kw)
        n_layers = 6  # 感受野 1+(1+2+4+8+16+32)=64 → 每阶更宽，两阶递归覆盖 ~128
        self.long_conv = DilatedCausalConv(d_model, n_layers=n_layers)
        del self.filter_fn  # 移除隐式滤波器

    def forward(self, u):
        l = u.size(-2)
        u = self.in_proj(u)
        u = u.transpose(1, 2)  # B,L,D -> B,D,L
        uc = self.short_filter(u)[..., :l]
        xs = uc.split(self.d_model, dim=1)
        *x, v = xs
        # 多阶门控递归：每阶过一次 dilated 长卷积（替代隐式核 fftconv）
        for x_i in reversed(x[1:]):
            v = self.dropout(v * x_i)
            v = self.long_conv(v)
        y = (v * x[0]).transpose(1, 2)
        return self.out_proj(y)


class HyenaDilated(hyena.Hyena):
    """结构与 hyena.Hyena 相同，HyenaOperator 换成 Dilated 版。"""

    def __init__(self, config):
        super().__init__(config)
        for block in self.blocks:
            op = block.hyena
            new_op = DilatedHyenaOperator(
                op.d_model, op.l_max, order=op.order,
                filter_order=64, dropout=op.dropout.p if hasattr(op.dropout, "p") else 0.0)
            new_op.in_proj = op.in_proj
            new_op.out_proj = op.out_proj
            new_op.short_filter = op.short_filter
            block.hyena = new_op


if __name__ == "__main__":
    cfg = hyena.Config(hidden_size=208, layers=5, order=2, filter_order=64,
                       ffn_mult=4, vocab_size=11, context_length=163)
    m = HyenaDilated(cfg).cuda()
    x = torch.randint(0, 11, (2, 163)).cuda()
    logits = m(x)
    print(f"params={sum(p.numel() for p in m.parameters()):,}  logits={tuple(logits.shape)}")
