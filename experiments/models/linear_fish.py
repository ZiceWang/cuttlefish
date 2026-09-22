"""LinearFish —— Cuttlefish 的精确线性化实现（O(T²) → O(T)）。

数学事实：Cuttlefish 的 GramAttention 无 softmax、无行归一化，
且 relu/normalize/RoPE 均为逐 token 操作，因果掩码 = 前缀和：
    msg_t = Σ_{s<t} q̃_t·k̃_s · v_s = q̃_tᵀ (Σ_{s<t} k̃_s v_sᵀ)
其中 q̃/k̃ = normalize(RoPE(ReLU(W x)))。前缀外积和 cumsum 精确等价，零近似。

参数与 bdh_best 完全同名同形，可直接加载 Cuttlefish checkpoint。
"""
import torch
import torch.nn.functional as F
from torch import nn

import bdh_best
from bdh_best import get_freqs, rope


class GramLinearAttention(nn.Module):
    """Gram 内核的前缀和精确实现。接口与 GramAttention 相同。"""

    def __init__(self, feature_width, theta=2**16):
        super().__init__()
        freqs = get_freqs(feature_width, theta=theta, dtype=torch.float32)
        self.register_buffer("freqs", freqs.view(1, 1, 1, feature_width), persistent=False)

    def forward(self, q, k, v, use_rope=True, normalize=False):
        # q,k: [B,H,T,dk]  v: [B,T,d]
        t = q.size(-2)
        positions = torch.arange(t, device=q.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)
        phases = positions * self.freqs
        qr = rope(phases, q) if use_rope else q
        kr = rope(phases, k) if use_rope else k
        if normalize:
            qr = F.normalize(qr, dim=-1)
            kr = F.normalize(kr, dim=-1)
        v_h = v.unsqueeze(1).to(kr.dtype)  # [B,1,T,d] broadcast heads
        # 前缀外积和 S_t = Σ_{s≤t} k̃_s ⊗ v_s : [B,H,T,dk,d]
        outer = kr.unsqueeze(-1) * v_h.unsqueeze(-2)  # [B,H,T,dk,d]
        S = outer.cumsum(dim=2)
        # 不含自身（tril diagonal=-1）：用 t-1 时刻前缀
        S_prev = torch.cat(
            (torch.zeros_like(S[:, :, :1]), S[:, :, :-1]), dim=2
        )
        # msg_t = q̃_tᵀ S_{t-1} : [B,H,T,d]
        msg = (qr.unsqueeze(-1) * S_prev).sum(dim=-2)
        return msg.to(v.dtype)


class LinearFishBlock(bdh_best.BDHBestBlock):
    """与 BDHBestBlock 结构/参数完全一致，仅 attn 换成线性前缀和版。"""

    def __init__(self, config, recurrences=None):
        super().__init__(config, recurrences)
        self.attn = GramLinearAttention(self.qk_width)


class LinearFish(bdh_best.BDHBest):
    """参数与 BDHBest 完全同名同形（state_dict 兼容）。"""


if __name__ == "__main__":
    cfg = bdh_best.Config(hidden_size=128, heads=4, mlp_mult=16, layers=4,
                          recurrences=2, vocab_size=11, context_length=163,
                          dropout=0.0, mixer_mode="swiglu", swiglu_in_loop=False,
                          mixer_width=640)
    m = LinearFish(cfg).cuda()
    x = torch.randint(0, 11, (2, 163)).cuda()
    logits = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params={n_params:,}  logits={tuple(logits.shape)}  "
          f"loss={F.cross_entropy(logits.reshape(-1, 11), x.reshape(-1)).item():.3f}")
