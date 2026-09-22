"""SharedGram —— 关系图算子参数绑定（全层共享），GLU 读出每层独立。

每层:
    msg = GramShared(x)          # QK 投影 + cosine Gram，参数全层绑定一份
    y = GLU_i(cat[x, ln(msg)])   # 宽 GLU 每层独立（用户指定）
    x = ln(x + ln(y)) → 块残差

图仍逐层动态（QK 由当前 x 投影），但"关系度量"全局唯一；独立的是读出。
"""
import dataclasses

import torch
import torch.nn.functional as F
from torch import nn

import bdh_best
import gram_norm


@dataclasses.dataclass
class Config:
    hidden_size: int = 112
    heads: int = 4
    mlp_mult: int = 16
    layers: int = 8
    vocab_size: int = 11
    context_length: int = 163
    dropout: float = 0.0
    mixer_width: int = 0
    init_std: float = 0.02
    n_groups: int = 1  # QK 关系图分组共享数（1=全共享，layers=全独立）
    norm_attn: bool = False  # 度归一化图消息
    posify: bool = False
    group_pattern: tuple = None  # 逐层组号序列（优先于 n_groups 连续分段）


class SharedGramModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        c = config
        d, nh = c.hidden_size, c.heads
        self.layers = c.layers
        qk_width = c.mlp_mult * d // nh // 2
        # ---- 分组共享的关系图算子（n_groups 份，组内参数绑定）----
        if getattr(c, "group_pattern", None) is not None:
            self.group_of = list(c.group_pattern)
            g = max(self.group_of) + 1
        else:
            g = max(1, min(c.n_groups, c.layers))
            self.group_of = [min(i * g // c.layers, g - 1) for i in range(c.layers)]
        self.shared_query = nn.ParameterList([
            nn.Parameter(torch.empty(d, nh * qk_width).normal_(std=c.init_std)) for _ in range(g)])
        self.shared_key = nn.ParameterList([
            nn.Parameter(torch.empty(d, nh * qk_width).normal_(std=c.init_std)) for _ in range(g)])
        attn_cls = gram_norm.GramAttentionNorm if c.norm_attn else bdh_best.GramAttention
        kw = {"posify": c.posify} if c.norm_attn else {}
        self.shared_attn = nn.ModuleList([attn_cls(qk_width, **kw) for _ in range(g)])
        # ---- 每层独立的 GLU 读出 ----
        n = c.mixer_width or 5 * d
        self.mixer_gate_up = nn.ModuleList(
            [nn.Linear(2 * d, 2 * n, bias=False) for _ in range(c.layers)])
        self.mixer_down = nn.ModuleList(
            [nn.Linear(n, d, bias=False) for _ in range(c.layers)])
        self.block_output = nn.ModuleList(
            [nn.Linear(d, d, bias=False) for _ in range(c.layers)])
        for lin in self.block_output:
            nn.init.normal_(lin.weight, std=c.init_std)
        self.ln = nn.LayerNorm(d, elementwise_affine=False, bias=False)
        self.token_embedding = nn.Embedding(c.vocab_size, d)
        self.position_embedding = nn.Embedding(c.context_length, d)
        self.input_norm = nn.LayerNorm(d, elementwise_affine=False, bias=False)
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, bias=False)
        self.output = nn.Linear(d, c.vocab_size, bias=False)
        nn.init.normal_(self.output.weight, std=c.init_std)
        self.d, self.nh, self.qk_width = d, nh, qk_width

    def _gram(self, enc, group):
        # enc: [B,T,d] → 多头图消息 [B,H,T,d]（RoPE + ReLU + cosine，同 bdh_best）
        B, T, _ = enc.shape
        q = F.relu(enc @ self.shared_query[group]).view(B, T, self.nh, self.qk_width).permute(0, 2, 1, 3)
        k = F.relu(enc @ self.shared_key[group]).view(B, T, self.nh, self.qk_width).permute(0, 2, 1, 3)
        return self.shared_attn[group](q, k, enc.unsqueeze(1), use_rope=True, normalize=True)

    def forward(self, idx):
        positions = torch.arange(idx.size(1), device=idx.device)
        hidden = self.token_embedding(idx) + self.position_embedding(positions)
        hidden = self.input_norm(hidden).unsqueeze(1)  # B,1,T,d
        for i in range(self.layers):
            block_input = hidden
            x = self.ln(hidden)
            enc = x.squeeze(1)
            msg = self._gram(enc, self.group_of[i])
            gm = self.ln(msg.mean(dim=1, keepdim=True))  # 多头平均
            current = x.squeeze(1)
            gm_flat = gm.squeeze(1)
            gate, value = self.mixer_gate_up[i](
                torch.cat((current, gm_flat), dim=-1)).chunk(2, dim=-1)
            y = self.mixer_down[i](F.silu(gate) * value).unsqueeze(1)
            hidden = block_input + self.block_output[i](self.ln(self.ln(x + self.ln(y))))
        return self.output(self.final_norm(hidden.squeeze(1)))


if __name__ == "__main__":
    cfg = Config()
    m = SharedGramModel(cfg).cuda()
    x = torch.randint(0, 11, (2, 163)).cuda()
    logits = m(x)
    print(f"params={sum(p.numel() for p in m.parameters()):,}  logits={tuple(logits.shape)}")
