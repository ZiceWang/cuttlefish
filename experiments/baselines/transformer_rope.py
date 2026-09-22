"""TransformerRoPE —— 标准 TF 但去掉 learned position embedding，Q/K 用 RoPE（theta=2^16）。

与 Cuttlefish 的 RoPE 同源（bdh_best.get_freqs / rope），公平对比纯相对位置。
"""
import torch
import torch.nn.functional as F
from torch import nn

import transformer
from bdh_best import get_freqs, rope


class RoPEBlock(transformer.TransformerBlock):
    def __init__(self, hidden_size, heads):
        super().__init__(hidden_size, heads)
        freqs = get_freqs(self.head_dim, theta=2**16, dtype=torch.float32)
        self.register_buffer("freqs", freqs.view(1, 1, 1, self.head_dim), persistent=False)

    def forward(self, hidden):
        batch, length, _ = hidden.shape
        h = self.attention_norm(hidden)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q, k, v = [
            t.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
            for t in (q, k, v)
        ]
        positions = torch.arange(length, device=hidden.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)
        phases = positions * self.freqs
        q = rope(phases, q)
        k = rope(phases, k)
        message = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        message = message.transpose(1, 2).reshape(batch, length, hidden.shape[-1])
        hidden = hidden + self.attention_output(message)
        return hidden + self.ffn(self.ffn_norm(hidden))


class TransformerRoPE(transformer.Transformer):
    def __init__(self, config):
        super().__init__(config)
        del self.position_embedding
        self.blocks = nn.ModuleList([
            RoPEBlock(config.hidden_size, config.heads) for _ in range(config.layers)
        ])

    def forward(self, idx):
        hidden = self.token_embedding(idx)
        hidden = self.input_norm(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.final_norm(hidden))


if __name__ == "__main__":
    cfg = transformer.Config(hidden_size=168, heads=4, layers=8, vocab_size=11, context_length=163)
    m = TransformerRoPE(cfg).cuda()
    x = torch.randint(0, 11, (2, 163)).cuda()
    print(f"params={sum(p.numel() for p in m.parameters()):,}  logits={tuple(m(x).shape)}")
