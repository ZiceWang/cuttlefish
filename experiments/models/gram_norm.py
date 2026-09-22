"""GramAttentionNorm —— 度归一化版 Gram 图（解决线性聚合不稳定）。

原版: msg = (masked scores) @ V            # 权重和≠1、有负边 → 尺度漂移
本版: msg = (masked scores) @ V / Σ|scores| # 度归一化加权平均，尺度稳定
可选 posify=True: scores.clamp_min(0) 先消负边（纯正加权平均）。
"""
import torch
import torch.nn.functional as F

from bdh_best import rope
import bdh_best


class GramAttentionNorm(bdh_best.GramAttention):
    def __init__(self, feature_width, theta=2**16, posify=False):
        super().__init__(feature_width, theta=theta)
        self.posify = posify

    def forward(self, q, k, v, use_rope=True, normalize=False):
        t = q.size(-2)
        positions = torch.arange(t, device=q.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)
        phases = positions * self.freqs
        qr = rope(phases, q) if use_rope else q
        kr = rope(phases, k) if use_rope else k
        if normalize:
            qr = F.normalize(qr, dim=-1)
            kr = F.normalize(kr, dim=-1)
        scores = qr @ kr.mT  # [B,H,T,T] ∈ [-1,1]
        mask = torch.ones(t, t, dtype=torch.bool, device=q.device).tril(diagonal=-1)
        scores = scores.masked_fill(~mask, 0)
        if self.posify:
            scores = scores.clamp_min(0)
        deg = scores.abs().sum(-1, keepdim=True).clamp_min(1e-2)  # [B,H,T,1]
        return (scores @ v) / deg


class NormDeepBlock(bdh_best.BDHBestBlock):
    """BDHBestBlock 但 attn 换度归一化版（R=1 使用）。"""

    def __init__(self, config, recurrences=None, posify=False):
        super().__init__(config, recurrences)
        self.attn = GramAttentionNorm(self.qk_width, posify=posify)


class NormDeep(bdh_best.BDHBest):
    """CutDeep 架构（L8 d92 R1）但图消息度归一化。state_dict 兼容 bdh_best。"""

    def __init__(self, config, posify=False):
        super().__init__(config)
        for block in self.blocks:
            block.attn = GramAttentionNorm(block.qk_width, posify=posify)
