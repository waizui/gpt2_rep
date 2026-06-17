from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size = 50257
    context_len = 1024
    emb_dim = 768
    n_heads = 12
    n_layers = 12
    drop_rate = 0.1
    qkv_bias = False


MODEL_CONFIGS = {
    "124M": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "355M": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "774M": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "1558M": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
