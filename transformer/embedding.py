import torch


class GPTEmbedding(torch.nn.Module):
    def __init__(
        self, contex_len: int, vocab_size: int, out_dim, drop_rate: float
    ) -> None:
        super().__init__()
        self.tok_emb = torch.nn.Embedding(vocab_size, out_dim)
        self.pos_emb = torch.nn.Embedding(contex_len, out_dim)
        self.drop_emb = torch.nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor):
        b, seq_len = x.shape
        x = self.tok_emb(x) + self.pos_emb(torch.arange(seq_len, device=x.device))
        return x
