"""GatedFish —— Hyena 递归状态演化 × Cuttlefish 动态图投影 的真融合。

每头状态递推（RNN 式，内容依赖衰减门 = "图动态演变"语义）：
    S_t = λ(x_t) · S_{t-1} + k̃(x_t) ⊗ v(x_t)
    msg_t = q̃(x_t)ᵀ S_{t-1}                      # 因果，不含自身
其中 q̃/k̃ = normalize(RoPE(ReLU(W_Q/K x)))（Cuttlefish 超宽动态图投影，逐 token 重投影
→ R 递归中图随内容演变），λ = sigmoid(W_λ x)（Hyena 遗忘门，内容依赖）。

λ 的引入破坏 cumsum 可分解性 → log-space 加权前缀和（fp32 累计，O(T)）：
    S_t = exp(L_t) · cumsum(exp(-L_s)·k̃_s⊗v_s)_t,  L_t = Σ_{r≤t} log λ_r
λ bias 初始化 +4（λ≈0.98，防溢出）。R 递归 / GLU_5d 读出 / 块残差与 Cuttlefish 一致。
"""
import dataclasses

import torch
import torch.nn.functional as F
from torch import nn

import bdh_best
from bdh_best import get_freqs, rope


@dataclasses.dataclass
class Config:
    hidden_size: int = 128
    heads: int = 4
    mlp_mult: int = 16
    layers: int = 4
    recurrences: int = 2
    vocab_size: int = 11
    context_length: int = 163
    dropout: float = 0.0
    use_rope: bool = True
    normalize_gram: bool = True
    mixer_width: int = 0  # 0 -> 5 * hidden
    init_std: float = 0.02
    decay_bias: float = 4.0


def gated_linear_attn(q, k, lam, v, freqs, use_rope=True, normalize=True):
    """q,k,lam: [B,H,T,dk]  v: [B,T,d]  →  msg [B,H,T,d]，fp32 累计。"""
    t = q.size(-2)
    positions = torch.arange(t, device=q.device, dtype=freqs.dtype).view(1, 1, -1, 1)
    phases = positions * freqs
    qr = rope(phases, q) if use_rope else q
    kr = rope(phases, k) if use_rope else k
    if normalize:
        qr = F.normalize(qr.float(), dim=-1)
        kr = F.normalize(kr.float(), dim=-1)
    lam = lam.float().clamp(1e-4, 1.0 - 1e-4)
    L = torch.log(lam).cumsum(dim=2)                       # [B,H,T,dk]
    scale_back = torch.exp(L).clamp(max=1e30)
    scaled_k = kr * torch.exp(-L).clamp(max=1e30)          # exp(-L) = exp(|L|) 可能大
    v_h = v.unsqueeze(1).to(torch.float32)                 # [B,1,T,d]
    outer = scaled_k.unsqueeze(-1) * v_h.unsqueeze(-2)     # [B,H,T,dk,d]
    S = outer.cumsum(dim=2)
    S_prev = torch.cat((torch.zeros_like(S[:, :, :1]), S[:, :, :-1]), dim=2)
    # exp(L_t) 是 per-(t,dk) 的衰减补偿，乘在状态的 dk 行上
    S_prev = S_prev * scale_back.unsqueeze(-1)             # [B,H,T,dk,d]
    msg = (qr.unsqueeze(-1) * S_prev).sum(dim=-2)          # [B,H,T,d]
    return msg.to(v.dtype)


class GatedFishBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        c = config
        d, nh = c.hidden_size, c.heads
        self.recurrences = c.recurrences
        self.qk_width = c.mlp_mult * d // nh // 2
        qk_total = nh * self.qk_width
        self.query_proj = nn.Parameter(torch.empty(d, qk_total).normal_(std=c.init_std))
        self.key_proj = nn.Parameter(torch.empty(d, qk_total).normal_(std=c.init_std))
        self.decay_proj = nn.Linear(d, qk_total)
        nn.init.zeros_(self.decay_proj.bias)
        self.decay_proj.bias.data.fill_(c.decay_bias)
        n = c.mixer_width or 5 * d
        self.mixer_gate_up = nn.Linear(2 * d, 2 * n, bias=False)
        self.mixer_down = nn.Linear(n, d, bias=False)
        self.block_output = nn.Linear(d, d, bias=False)
        self.ln = nn.LayerNorm(d, elementwise_affine=False, bias=False)
        freqs = get_freqs(self.qk_width, theta=2**16, dtype=torch.float32)
        self.register_buffer("freqs", freqs.view(1, 1, 1, self.qk_width), persistent=False)

    def _attn(self, enc):
        # enc: [B,T,d] → 图消息 [B,H,T,d]
        B, T, _ = enc.shape
        nh, dk = self.recurrences and 4 or 4, self.qk_width  # heads 固定 4（与 bdh_best 一致）
        nh = self.query_proj.shape[1] // dk
        q = F.relu(enc @ self.query_proj).view(B, T, nh, dk).permute(0, 2, 1, 3)
        k = F.relu(enc @ self.key_proj).view(B, T, nh, dk).permute(0, 2, 1, 3)
        lam = torch.sigmoid(self.decay_proj(enc).view(B, T, nh, dk).permute(0, 2, 1, 3))
        msg = gated_linear_attn(q, k, lam, enc, self.freqs,
                                use_rope=True, normalize=True)
        return msg  # [B,H,T,d]

    def forward(self, x):
        block_input = x
        x = self.ln(x)
        last_gm = None
        for _ in range(self.recurrences):
            residual = x
            enc = x.squeeze(1)
            msg = self._attn(enc)
            gm = self.ln(msg.mean(dim=1, keepdim=True))  # 多头平均（与 Cuttlefish 一致）
            last_gm = gm
            x = self.ln(residual + gm)
        current = x.squeeze(1)
        gm_flat = last_gm.squeeze(1)
        gate, value = self.mixer_gate_up(torch.cat((current, gm_flat), dim=-1)).chunk(2, dim=-1)
        y_mlp = self.mixer_down(F.silu(gate) * value).unsqueeze(1)
        x = self.ln(x + self.ln(y_mlp))
        return block_input + self.block_output(self.ln(x))


class GatedFish(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        c = config
        d = c.hidden_size
        self.token_embedding = nn.Embedding(c.vocab_size, d)
        self.position_embedding = nn.Embedding(c.context_length, d)
        self.input_norm = nn.LayerNorm(d, elementwise_affine=False, bias=False)
        self.blocks = nn.ModuleList([GatedFishBlock(c) for _ in range(c.layers)])
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
    m = GatedFish(cfg).cuda()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_length)).cuda()
    logits = m(x)
    print(f"params={sum(p.numel() for p in m.parameters()):,}  logits={tuple(logits.shape)}")
