import torch
import torch.nn as nn
from torch import Tensor

from transformer.config import GPTConfig


# Gaussian Error Linear Unit
class GELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.mid_dim = 4 * cfg.emb_dim

        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, self.mid_dim),
            GELU(),
            nn.Linear(self.mid_dim, cfg.emb_dim),
        )

    def forward(self, x):
        return self.layers(x)
