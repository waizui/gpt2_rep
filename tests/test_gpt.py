import torch
from transformer.config import GPTConfig
from transformer.gpt import GPTModel
from transformer.tokenizer import GPTTokenizer
from transformer.train import gen_text_simple


def test_gpt_model_prints_param_num():
    cfg = GPTConfig()
    model = GPTModel(cfg)

    param_num = model.param_num()

    print(f"GPTModel param_num: {param_num}")
    assert param_num == sum(p.numel() for p in model.parameters())


def test_gent_text_simple():
    text = "Hello, I am"
    tokenizer = GPTTokenizer()
    # add batch dim
    encoded = torch.tensor(tokenizer.encode(text)).unsqueeze(0)

    cfg = GPTConfig()
    model = GPTModel(cfg)

    out = gen_text_simple(
        model, encoded, max_new_tokens=6, context_size=cfg.context_len
    )

    decoded = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded)
