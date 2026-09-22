"""CutDeep-Norm 但去掉输入端 learned position embedding（纯 Gram 内 RoPE）。"""
import torch
import bdh_best
import gram_norm


class CutNoEmb(gram_norm.NormDeep):
    def __init__(self, config):
        super().__init__(config)
        del self.position_embedding

    def forward(self, idx):
        hidden = self.token_embedding(idx)
        hidden = self.input_norm(hidden).unsqueeze(1)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.final_norm(hidden.squeeze(1)))
