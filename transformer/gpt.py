import torch
import torch.nn as nn
from torch import Tensor

from transformer.config import GPTConfig
from transformer.embedding import GPTEmbedding


class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.emb = GPTEmbedding(
            cfg.context_len, cfg.vocab_size, cfg.emb_dim, cfg.drop_rate
        )
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_heads)]
        )

        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx: Tensor):
        x = self.emb(in_idx)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()

    def forward(self, x):
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


