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
    def __init__(self, normalized_shape: int, eps=1e-5) -> None:
        super().__init__()

    def forward(self, x):
        return x
