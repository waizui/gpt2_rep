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


# is not casual, casual is relex
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


# is not casual, casual is relex
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_len: int,
        dropout: float,
        num_heads: int,
        qkv_bias=False,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_len, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    mask: Tensor

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_len: int,
        dropout: float,
        num_heads: int,
        qkv_bias=False,
    ) -> None:
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)

        # diagonal=1, set zero where upper element's distance to diagonal >= 1
        triu = torch.triu(torch.ones(context_len, context_len), diagonal=1)
        # tensors that not parameters
        self.register_buffer("mask", triu)

    def forward(self, x: Tensor):
        b, num_tokens, d_in = x.shape
        q: Tensor = self.W_q(x)  # [b, num_tokens, d_out]
        k: Tensor = self.W_k(x)
        v: Tensor = self.W_v(x)

        # [b,num_tokens, num_heads, head_dim] -> d_out = num_heads*head_dim
        # [1,2,3,4,5,6,7,8,9] -> [[1,2,3],[4,5,6],[7,8,9]]
        q = q.view(
            b, num_tokens, self.num_heads, self.head_dim
        )  # view can only used in contiguous tensor
        k = k.view(b, num_tokens, self.num_heads, self.head_dim)
        v = v.view(b, num_tokens, self.num_heads, self.head_dim)

        # [b,num_tokens, num_heads, head_dim] ->  [b, num_heads, num_tokens, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # [b, num_heads, num_tokens, num_tokens]
        atten_scorce = q @ k.transpose(2, 3)
        mask = self.mask.bool()[:num_tokens, :num_tokens]
        # boardcast from the right most dim: [num_tokens,num_tokens]-> [b,num_heads, num_tokens,num_tokens]
        atten_scorce.masked_fill_(mask, -torch.inf)

        # if masking is set to zero, re-norm will be required, but use -inf, no need to re-norm
        atten = softmax(atten_scorce / k.shape[-1] ** 0.5, dim=-1)
        atten: Tensor = self.dropout(atten)
        # [b, num_heads, num_tokens, num_tokens] @ [b, num_heads, num_tokens, head_dim]
        # -> transpose(1,2)-> [b,num_tokens, num_heads, head_dim]
        context_vec = (atten @ v).transpose(1, 2)
        # d_out = num_heads*head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        context_vec = self.out_proj(context_vec)
        return context_vec
