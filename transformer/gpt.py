import torch
import torch.nn as nn
from torch import Tensor

from transformer.attention import MultiHeadAttention
from transformer.config import GPTConfig
from transformer.embedding import GPTEmbedding
from transformer.feedforward import FeedForward


class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.emb = GPTEmbedding(
            cfg.context_len, cfg.vocab_size, cfg.emb_dim, cfg.drop_rate
        )
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx: Tensor):
        x = self.emb(in_idx)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        # some model use shared weights(weight tying) instead of a new layer
        logits = self.out_head(x)
        return logits

    def param_num(self) -> int:
        num = sum(p.numel() for p in self.parameters())
        return num


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.att = MultiHeadAttention(
            cfg.emb_dim,
            cfg.emb_dim,
            cfg.context_len,
            cfg.drop_rate,
            cfg.n_heads,
            cfg.qkv_bias,
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        shortcut = x
        # pre-layernorm(norm befor attention), better training result
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int, eps=1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x: Tensor):
        # keepdim: [[1,2],[3,4]]->mean(dim=-1)-> [1.5,3.5]->keepdim -> [[1.5],[3.5]]
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased: do not use bessel's correction, divided by n, not (n-1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # make mean=0, var=1
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
