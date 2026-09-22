"""TransformerCosine —— TF-RoPE 但把 softmax attention 换成度归一化余弦图消息。

唯一变量：聚合核 softmax(QKᵀ) → (cos · V) / Σ|cos|（CutDeep 同款，保留负边）。
其余（head_dim 42、W_V 投影、FFN、残差、RoPE）与 TF-RoPE 完全一致。
"""
import torch
import torch.nn.functional as F
from torch import nn

import transformer
import transformer_rope
from bdh_best import get_freqs, rope


class CosineBlock(transformer_rope.RoPEBlock):
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
        q = F.normalize(rope(phases, q), dim=-1)
        k = F.normalize(rope(phases, k), dim=-1)
        scores = q @ k.mT  # [B,H,T,T] ∈ [-1,1]
        mask = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril(diagonal=-1)
        scores = scores.masked_fill(~mask, 0)  # 因果，不含自身
        deg = scores.abs().sum(-1, keepdim=True).clamp_min(1e-2)
        message = (scores @ v) / deg  # 度归一化加权平均（保留负边）
        message = message.transpose(1, 2).reshape(batch, length, hidden.shape[-1])
        hidden = hidden + self.attention_output(message)
        return hidden + self.ffn(self.ffn_norm(hidden))


class TransformerCosine(transformer_rope.TransformerRoPE):
    def __init__(self, config):
        super().__init__(config)
        self.blocks = nn.ModuleList([
            CosineBlock(config.hidden_size, config.heads) for _ in range(config.layers)
        ])



if __name__ == "__main__":
    cfg = transformer.Config(hidden_size=168, heads=4, layers=8, vocab_size=11, context_length=163)
    m = TransformerCosine(cfg).cuda()
    x = torch.randint(0, 11, (2, 163)).cuda()
    print(f"params={sum(p.numel() for p in m.parameters()):,}  logits={tuple(m(x).shape)}")
