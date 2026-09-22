"""最佳修改版 BDH —— gram_swiglu · separate_qk+relu_qk · normalize_gram · RoPE。

对应 experiments/bdh_sudoku4 的 mixed_bdh_l6r3_h212_qk_ng_c（Pareto 甜点，10.1M）：
StackedBDHLanguageModel + BDHBlock(recurrences=3, block_skip=True, dense_swiglu=True,
encoder_dropout=True, flat_encoder=True, use_rope=True, normalize_gram=True,
separate_qk=True, relu_qk=True)。

相对原始 BDH 的关键改动：
1. separate_qk：独立 query/key 投影（不再共享 encoder），两者各自 ReLU（非负）
2. normalize_gram：对 q/k 做 L2 归一化（余弦边），弥补无 softmax 的有界性
3. dense_swiglu：读出改为 SwiGLU 门控（cat[当前, 图消息] → gate*value）
4. block_skip：块级残差（输出投影 + ReZero 风格 LN），层内 recurrences 共享参数循环

统一接口：Model(Config).forward(idx) -> logits[B, T, vocab]
"""
import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return 1.0 / (theta ** (quantize(torch.arange(0, n, dtype=dtype)) / n)) / (2 * math.pi)


def rope(phases, value):
    phases = phases[..., ::2]
    phases_cos = torch.cos((phases % 1) * (2 * math.pi))
    phases_sin = torch.sin((phases % 1) * (2 * math.pi))
    paired = value.unflatten(-1, (-1, 2))
    even, odd = paired.unbind(dim=-1)
    return torch.stack(
        (
            (even * phases_cos).to(value.dtype) - (odd * phases_sin).to(value.dtype),
            (odd * phases_cos).to(value.dtype) + (even * phases_sin).to(value.dtype),
        ),
        dim=-1,
    ).flatten(-2)


class GramAttention(nn.Module):
    """Gram 内核 + 可选 cosine 归一化（normalize）与 softmax 边。"""

    def __init__(self, feature_width, theta=2**16):
        super().__init__()
        freqs = get_freqs(feature_width, theta=theta, dtype=torch.float32)
        self.register_buffer("freqs", freqs.view(1, 1, 1, feature_width), persistent=False)

    def forward(self, q, k, v, use_rope=True, normalize=False):
        t = q.size(-2)
        positions = torch.arange(t, device=q.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)
        phases = positions * self.freqs
        qr = rope(phases, q) if use_rope else q
        kr = rope(phases, k) if use_rope else k
        if normalize:
            qr = F.normalize(qr, dim=-1)
            kr = F.normalize(kr, dim=-1)
        scores = qr @ kr.mT  # [B,H,T,T]
        mask = torch.ones(t, t, dtype=torch.bool, device=q.device).tril(diagonal=-1)
        return scores.masked_fill(~mask, 0) @ v  # 无 softmax、不含自身


@dataclasses.dataclass
class Config:
    hidden_size: int = 212
    heads: int = 4
    mlp_mult: int = 16  # n = mlp_mult * hidden // heads（每头特征宽度）
    layers: int = 6
    recurrences: int = 3  # 层内共享参数循环次数
    vocab_size: int = 65
    context_length: int = 256
    dropout: float = 0.1
    use_rope: bool = True
    normalize_gram: bool = True
    separate_qk: bool = True
    relu_qk: bool = True
    swiglu_in_loop: bool = True  # False: dense SwiGLU 移出递归循环（每层一次）
    mixer_mode: str = "swiglu"  # "swiglu"|"linear"|"add"|"had"|"merged"|"gate"：读出融合方式
    swiglu_mlp: bool = False  # True: 块级后加 h->2h->h SwiGLU MLP + 残差流
    ffn_mult: int = 2  # 块级 FFN 内部宽度倍数（internal = ffn_mult * d）
    mixer_width: int = 0  # GLU 内部宽度 n_mixer（0=用 mlp_mult 逻辑），与 QK 宽度解耦
    init_std: float = 0.02


class BDHBestBlock(nn.Module):
    """gram_swiglu 块：separate_qk + ReLU + cosine Gram + dense SwiGLU 绑定读出 + 块残差。"""

    def __init__(self, config: Config, recurrences=None):
        super().__init__()
        c = config
        d, nh = c.hidden_size, c.heads
        self.recurrences = recurrences if recurrences is not None else c.recurrences
        n = c.mlp_mult * d // nh
        n_mixer = c.mixer_width if c.mixer_width > 0 else n
        self.qk_width = n // 2
        assert self.qk_width > 0, "qk_width 必须 >=1（增大 mlp_mult）"
        self.query_proj = nn.Parameter(torch.empty(d, nh * self.qk_width).normal_(std=c.init_std))
        self.key_proj = nn.Parameter(torch.empty(d, nh * self.qk_width).normal_(std=c.init_std))
        self.attn = GramAttention(self.qk_width)
        if c.mixer_mode == "linear":
            self.mixer_gate_up = None
            self.mixer_down = None
            self.mixer_lin = nn.Linear(2 * d, d, bias=False)
            self.merged_gate_up = None
            self.merged_down = None
        elif c.mixer_mode in ("add", "had"):
            # 无参数：融合(add/hadamard) + SiLU 非线性，无任何投影
            self.mixer_gate_up = None
            self.mixer_down = None
            self.mixer_lin = None
            self.merged_gate_up = None
            self.merged_down = None
        elif c.mixer_mode == "merged":
            # 合并单块：FFN(cat[状态, 图消息])，2d -> 2*(2d) -> d，门控+容量一体
            self.mixer_gate_up = None
            self.mixer_down = None
            self.mixer_lin = None
            self.merged_gate_up = nn.Linear(2 * d, 4 * d, bias=False)
            self.merged_down = nn.Linear(2 * d, d, bias=False)
            nn.init.normal_(self.merged_gate_up.weight, std=c.init_std)
            nn.init.normal_(self.merged_down.weight, std=c.init_std)
            self.narrow_gate_up = None
            self.narrow_down = None
        elif c.mixer_mode == "gate":
            # 窄 GLU：不扩容门控 2d->2d（gate/value 各 d）+ d->d down；容量全交给块级 FFN
            self.mixer_gate_up = None
            self.mixer_down = None
            self.mixer_lin = None
            self.merged_gate_up = None
            self.merged_down = None
            self.narrow_gate_up = nn.Linear(2 * d, 2 * d, bias=False)
            self.narrow_down = nn.Linear(d, d, bias=False)
            nn.init.normal_(self.narrow_gate_up.weight, std=c.init_std)
            nn.init.normal_(self.narrow_down.weight, std=c.init_std)
        else:
            self.mixer_gate_up = nn.Linear(2 * d, 2 * n_mixer, bias=False)  # dense_swiglu
            self.mixer_down = nn.Linear(n_mixer, d, bias=False)
            self.mixer_lin = None
            self.merged_gate_up = None
            self.merged_down = None
            self.narrow_gate_up = None
            self.narrow_down = None
        self.ln = nn.LayerNorm(d, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(c.dropout)
        self.block_output = nn.Linear(d, d, bias=False)  # block_skip
        nn.init.normal_(self.block_output.weight, std=c.init_std)
        self.use_rope = c.use_rope
        self.normalize_gram = c.normalize_gram
        self.relu_qk = c.relu_qk
        self.swiglu_in_loop = c.swiglu_in_loop
        self.mixer_mode = c.mixer_mode
        self.swiglu_mlp = c.swiglu_mlp
        if c.swiglu_mlp:
            self.ffn_norm = nn.LayerNorm(d, elementwise_affine=False, bias=False)
            internal = c.ffn_mult * d  # 块级 FFN 内部宽度（默认 2d，可加大）
            self.ffn_gate_up = nn.Linear(d, 2 * internal, bias=False)  # h -> 2*internal
            self.ffn_down = nn.Linear(internal, d, bias=False)  # internal -> h
            nn.init.normal_(self.ffn_gate_up.weight, std=c.init_std)
            nn.init.normal_(self.ffn_down.weight, std=c.init_std)
        else:
            self.ffn_norm = None
            self.ffn_gate_up = None
            self.ffn_down = None
        self.n_layers = c.layers

    def _ffn(self, x):
        gate, up = self.ffn_gate_up(x).chunk(2, dim=-1)
        return self.ffn_down(F.silu(gate) * up)

    def _merged_ffn(self, x):
        # x: [B,T,2d]，一个块同时做门控+容量
        gate, up = self.merged_gate_up(x).chunk(2, dim=-1)  # 各 internal=2d
        return self.merged_down(F.silu(gate) * up)

    def _mix(self, current, graph_message):
        """读出：swiglu=GLU门控(cat)；linear=线性方阵(cat)；
        add=SiLU(cur+msg) 无参数；had=SiLU(cur*msg) 无参数。"""
        if self.mixer_mode == "linear":
            return self.mixer_lin(torch.cat((current, graph_message), dim=-1))
        if self.mixer_mode == "add":
            return current + graph_message  # 无参数、无非线性
        if self.mixer_mode == "had":
            return F.silu(current * graph_message)
        if self.mixer_mode == "gate":
            # 窄 GLU：2d -> 2d 不扩容门控（gate/value 各 d）
            x = torch.cat((current, graph_message), dim=-1)
            gate, value = self.narrow_gate_up(x).chunk(2, dim=-1)
            return self.narrow_down(F.silu(gate) * value)
        x = torch.cat((current, graph_message), dim=-1)
        gate, value = self.mixer_gate_up(x).chunk(2, dim=-1)
        return self.mixer_down(F.silu(gate) * value)

    def _project_qk(self, enc):
        proj_q = enc @ self.query_proj
        proj_k = enc @ self.key_proj
        if self.relu_qk:
            proj_q = F.relu(proj_q)
            proj_k = F.relu(proj_k)
        return proj_q, proj_k

    def forward(self, x):
        # x: B,1,T,d（块输入）
        block_input = x
        x = self.ln(x)  # block_skip 先归一化
        if self.swiglu_in_loop and self.mixer_mode not in ("merged", "gate"):
            for _ in range(self.recurrences):
                residual = x
                source = x  # pre_norm=False
                enc = source.squeeze(1)  # B,T,d
                proj_q, proj_k = self._project_qk(enc)
                x_sparse = proj_q.view(enc.size(0), enc.size(1), -1, self.qk_width).permute(0, 2, 1, 3)
                x_k = proj_k.view(enc.size(0), enc.size(1), -1, self.qk_width).permute(0, 2, 1, 3)
                msg = self.attn(x_sparse, x_k, source, use_rope=self.use_rope, normalize=self.normalize_gram)  # B,nh,T,d
                y_kv = self.ln(msg)
                graph_message = y_kv.mean(dim=1)  # B,T,d
                current = source.squeeze(1)
                y_mlp = self._mix(current, graph_message).unsqueeze(1)  # B,1,T,d
                update = self.ln(y_mlp)
                x = self.ln(residual + update)
        else:
            # 循环内：仅 Gram 图消息传递；dense SwiGLU 移出循环（每层一次）
            last_gm = None
            for _ in range(self.recurrences):
                residual = x
                source = x
                enc = source.squeeze(1)
                proj_q, proj_k = self._project_qk(enc)
                x_sparse = proj_q.view(enc.size(0), enc.size(1), -1, self.qk_width).permute(0, 2, 1, 3)
                x_k = proj_k.view(enc.size(0), enc.size(1), -1, self.qk_width).permute(0, 2, 1, 3)
                msg = self.attn(x_sparse, x_k, source, use_rope=self.use_rope, normalize=self.normalize_gram)
                gm = self.ln(msg).mean(dim=1, keepdim=True)  # B,1,T,d 图消息
                last_gm = gm
                x = self.ln(residual + gm)  # 循环内仅图消息残差
            # 循环外：dense SwiGLU 融合（当前状态 + 最后图消息）
            current = x.squeeze(1)
            gm_flat = last_gm.squeeze(1)
            if self.mixer_mode == "merged":
                y_mlp = self._merged_ffn(torch.cat((current, gm_flat), dim=-1)).unsqueeze(1)
            else:
                y_mlp = self._mix(current, gm_flat).unsqueeze(1)
            x = self.ln(x + self.ln(y_mlp))
        if self.swiglu_mlp:
            # 块级后加：h -> 2h -> h SwiGLU MLP + 残差流
            x = x + self._ffn(self.ffn_norm(x.squeeze(1))).unsqueeze(1)
        return block_input + self.block_output(self.ln(x))


class BDHBest(nn.Module):
    """6 层 × 3 递归 gram_swiglu BDH（mixed_bdh_l6r3_h212_qk_ng_c）。"""

    def __init__(self, config: Config):
        super().__init__()
        c = config
        d = c.hidden_size
        self.token_embedding = nn.Embedding(c.vocab_size, d)
        self.position_embedding = nn.Embedding(c.context_length, d)
        self.input_norm = nn.LayerNorm(d, elementwise_affine=False, bias=False)
        self.blocks = nn.ModuleList([BDHBestBlock(c) for _ in range(c.layers)])
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, bias=False)
        self.output = nn.Linear(d, c.vocab_size, bias=False)
        nn.init.normal_(self.output.weight, std=c.init_std)

    def forward(self, idx):
        positions = torch.arange(idx.size(1), device=idx.device)
        hidden = self.token_embedding(idx) + self.position_embedding(positions)
        hidden = self.input_norm(hidden).unsqueeze(1)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.final_norm(hidden.squeeze(1)))


if __name__ == "__main__":
    cfg = Config()
    m = BDHBest(cfg).cuda()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_length)).cuda()
    logits = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params={n_params:,}  logits={tuple(logits.shape)}  "
          f"loss={F.cross_entropy(logits.reshape(-1, cfg.vocab_size), x.reshape(-1)).item():.3f}")
