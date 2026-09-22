"""HyenaFish —— Hyena 骨架 + Cuttlefish 思想。

Cuttlefish 三件套（bdh_best wide-GLU 最优配置）：
  1. 循环内 R 次共享参数线性传播（delta 演化，深度 = L×R）
  2. 循环外超宽 GLU(cat[状态, 最后消息]) 门控读出（宽度 > 块数）
  3. 块残差 block_skip

替换点：循环内 mixer 由 Gram 图（QKᵀ，O(T²)）换成 HyenaOp
（FFT 长卷积 + 门控，O(T log T)）——解决长上下文软肋与图构建 FLOPs。

统一接口：Model(Config).forward(idx) -> logits[B, T, vocab]
"""
import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn

from hyena import HyenaOperator


@dataclasses.dataclass
class Config:
    hidden_size: int = 128
    layers: int = 4
    recurrences: int = 2
    mixer_width: int = 0  # 0 -> 5 * hidden_size
    order: int = 2
    filter_order: int = 64
    vocab_size: int = 11
    context_length: int = 163
    dropout: float = 0.0


class FishBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        c = config
        d = c.hidden_size
        self.recurrences = c.recurrences
        self.mixer = HyenaOperator(d, l_max=c.context_length, order=c.order,
                                   filter_order=c.filter_order, dropout=c.dropout)
        n = c.mixer_width or 5 * d
        self.mixer_gate_up = nn.Linear(2 * d, 2 * n, bias=False)
        self.mixer_down = nn.Linear(n, d, bias=False)
        self.block_output = nn.Linear(d, d, bias=False)
        self.ln = nn.LayerNorm(d, elementwise_affine=False, bias=False)

    def forward(self, x):
        # x: B,1,T,d
        block_input = x
        x = self.ln(x)
        last_gm = None
        for _ in range(self.recurrences):
            residual = x
            enc = x.squeeze(1)  # B,T,d
            msg = self.mixer(enc).unsqueeze(1)  # B,1,T,d
            gm = self.ln(msg)
            last_gm = gm
            x = self.ln(residual + gm)  # 循环内仅消息残差（delta 演化）
        current = x.squeeze(1)
        gm_flat = last_gm.squeeze(1)
        gate, value = self.mixer_gate_up(torch.cat((current, gm_flat), dim=-1)).chunk(2, dim=-1)
        y_mlp = self.mixer_down(F.silu(gate) * value).unsqueeze(1)
        x = self.ln(x + self.ln(y_mlp))
        return block_input + self.block_output(self.ln(x))


class HyenaFish(nn.Module):
    """HyenaFish 语言模型：token+pos embedding → FishBlocks → final norm → head。"""

    def __init__(self, config: Config):
        super().__init__()
        c = config
        d = c.hidden_size
        self.token_embedding = nn.Embedding(c.vocab_size, d)
        self.position_embedding = nn.Embedding(c.context_length, d)
        self.input_norm = nn.LayerNorm(d, elementwise_affine=False, bias=False)
        self.blocks = nn.ModuleList([FishBlock(c) for _ in range(c.layers)])
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, bias=False)
        self.output = nn.Linear(d, c.vocab_size, bias=False)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)

    def forward(self, idx):
        positions = torch.arange(idx.size(1), device=idx.device)
        hidden = self.token_embedding(idx) + self.position_embedding(positions)
        hidden = self.input_norm(hidden).unsqueeze(1)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.final_norm(hidden.squeeze(1)))


if __name__ == "__main__":
    cfg = Config()
    m = HyenaFish(cfg).cuda()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_length)).cuda()
    logits = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params={n_params:,}  logits={tuple(logits.shape)}  "
          f"loss={F.cross_entropy(logits.reshape(-1, cfg.vocab_size), x.reshape(-1)).item():.3f}")
