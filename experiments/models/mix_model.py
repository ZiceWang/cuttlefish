"""BDH + Transformer 混合堆叠。

按 pattern 交替堆叠标准 Transformer 块（pre-norm + SwiGLU FFN）与
BDH 加宽GLU 块（循环内纯图传播 + 循环外宽 GLU 门控）。
统一接口：MixModel(Config).forward(idx) -> logits[B,T,vocab]
"""
import dataclasses

import torch
from torch import nn

import bdh_best
import transformer


@dataclasses.dataclass
class MixConfig:
    hidden_size: int = 232
    heads: int = 4
    layers: int = 6
    pattern: tuple = ("T", "T", "B", "T", "T", "B")  # T=Transformer, B=BDH
    recurrences: int = 2  # BDH 循环次数
    mixer_width: int = 1160  # BDH GLU 内部宽度
    mlp_mult: int = 16
    vocab_size: int = 256
    context_length: int = 256
    dropout: float = 0.1


class MixModel(nn.Module):
    def __init__(self, config: MixConfig):
        super().__init__()
        c = config
        d = c.hidden_size
        self.token_embedding = nn.Embedding(c.vocab_size, d)
        self.position_embedding = nn.Embedding(c.context_length, d)
        bdh_cfg = bdh_best.Config(
            hidden_size=d, heads=c.heads, mlp_mult=c.mlp_mult, layers=1,
            recurrences=c.recurrences, vocab_size=c.vocab_size,
            context_length=c.context_length, dropout=c.dropout,
            mixer_mode="swiglu", swiglu_in_loop=False, mixer_width=c.mixer_width)
        blocks = []
        for kind in c.pattern:
            if kind == "B":
                blocks.append(bdh_best.BDHBestBlock(bdh_cfg))
            else:
                blocks.append(transformer.TransformerBlock(d, c.heads))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = transformer.RMSNorm(d)
        self.output = nn.Linear(d, c.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        d = self.token_embedding.embedding_dim
        h = self.token_embedding(idx) + self.position_embedding(
            torch.arange(T, device=idx.device))
        for blk in self.blocks:
            if isinstance(blk, bdh_best.BDHBestBlock):
                h = blk(h.unsqueeze(1)).squeeze(1)  # B,1,T,d <-> B,T,d
            else:
                h = blk(h)
        return self.output(self.final_norm(h))
