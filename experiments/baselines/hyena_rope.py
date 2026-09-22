"""HyenaRoPE —— Hyena 但用 RoPE 替代 learned position embedding。

原版 hyena.py: token_embedding + position_embedding(learned) + HyenaBlocks。
本版: 无 position_embedding，embedding 后对 hidden 施加 RoPE 旋转
（偶奇通道对旋转，与 Cuttlefish 同款 theta=2^16），其余结构不变。
"""
import torch
from torch import nn

import hyena
from bdh_best import get_freqs, rope


class HyenaRoPE(hyena.Hyena):
    def __init__(self, config):
        super().__init__(config)
        del self.position_embedding  # 移除 learned 位置嵌入
        d = config.hidden_size
        freqs = get_freqs(d, theta=2**16, dtype=torch.float32)
        self.register_buffer("freqs", freqs.view(1, 1, d), persistent=False)

    def forward(self, idx):
        hidden = self.token_embedding(idx)
        t = idx.size(1)
        phases = torch.arange(t, device=idx.device, dtype=self.freqs.dtype).view(1, -1, 1) * self.freqs
        hidden = rope(phases, hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.final_norm(hidden))


if __name__ == "__main__":
    cfg = hyena.Config(hidden_size=208, layers=5, order=2, filter_order=64,
                       ffn_mult=4, vocab_size=11, context_length=163)
    m = HyenaRoPE(cfg).cuda()
    x = torch.randint(0, 11, (2, 163)).cuda()
    logits = m(x)
    print(f"params={sum(p.numel() for p in m.parameters()):,}  logits={tuple(logits.shape)}")
