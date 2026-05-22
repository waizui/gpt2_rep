import torch
import torch.nn as nn
from torch import Tensor, softmax


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False) -> None:
        super().__init__()
        # replace nn.Parameter for bias operation and optimized initial weights
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: Tensor):
        q = self.W_q(x)  # [N,d_out] , x @ W_q
        k = self.W_k(x)
        v = self.W_v(x)

        atten_scorce = (
            q @ k.T
        )  # [N,d_out] @ [d_out,N], meaning: [i][j] = ith token's attention to jth token
        atten = softmax(atten_scorce / k.shape[-1] ** 0.5, dim=-1)
        context_vec = atten @ v  # [N, d_out]
        return context_vec


class CausalAttention(nn.Module):
    mask: Tensor

    def __init__(
        self, d_in: int, d_out: int, context_len: int, dropout: float, qkv_bias=False
    ) -> None:
        super().__init__()
        self.d_out = d_out
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # diagonal=1, set zero when upper element's distance to diagonal >= 1
        triu = torch.triu(torch.ones(context_len, context_len), diagonal=1)
        # tensors that not parameters
        self.register_buffer("mask", triu)

    def forward(self, x: Tensor):
        b, num_tokens, d_in = x.shape
        q: Tensor = self.W_q(x)
        k: Tensor = self.W_k(x)
        v: Tensor = self.W_v(x)

        atten_scorce = q @ k.transpose(1, 2)
        # to(torch.bool), non zero will be true.
        mask = self.mask.bool()[:num_tokens, :num_tokens]
        # fill_ is inplace-modify, mask will boardcast
        atten_scorce.masked_fill_(mask, -torch.inf)

        # if masking is set to zero, re-norm will be required, but use -inf, no need to re-norm
        atten = softmax(atten_scorce / k.shape[-1] ** 0.5, dim=-1)
        atten = self.dropout(atten)

        context_vec = atten @ v
        return context_vec
