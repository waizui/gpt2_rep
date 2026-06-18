from argparse import ArgumentParser

import torch
from load_gpt import load_gpt_settings_params, load_weights_into_gpt
from transformer.config import MODEL_CONFIGS, GPTConfig
from transformer.gpt import GPTModel
from transformer.tokenizer import GPTTokenizer
from transformer.train import gen_text, text_to_token_ids, token_ids_to_text


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", default="Every effort moves you")
    parser.add_argument(
        "--model-size",
        default="124M",
        choices=tuple(MODEL_CONFIGS),
    )
    args = parser.parse_args()

    input = args.input
    model_size = args.model_size

    model_config = MODEL_CONFIGS[model_size]

    cfg = GPTConfig()
    cfg.emb_dim = model_config["emb_dim"]
    cfg.n_layers = model_config["n_layers"]
    cfg.n_heads = model_config["n_heads"]
    cfg.context_len = 1024
    cfg.qkv_bias = True
    model = GPTModel(cfg)
    print(f"model param num: {model.param_num()}")

    _, params = load_gpt_settings_params(model_size)

    load_weights_into_gpt(model, params)

    device = torch.device("cuda")

    model.to(device)
    tokenizer = GPTTokenizer()
    ids = gen_text(
        model,
        idx=text_to_token_ids(input, tokenizer).to(device),
        max_new_tokens=25,
        context_size=cfg.context_len,
        top_k=50,
        temperature=1.0,
    )

    print("LLM: \n", token_ids_to_text(ids.cpu(), tokenizer))


if __name__ == "__main__":
    main()
