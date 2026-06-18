from argparse import ArgumentParser
from enum import Flag
import json

import numpy as np
import torch
from transformer.config import MODEL_CONFIGS, GPTConfig
from transformer.dataset import format_input
from transformer.fine_tune import create_data
from transformer.gpt import GPTModel
from transformer.io import load_model
from transformer.tokenizer import GPTTokenizer
from transformer.train import gen_text, text_to_token_ids, token_ids_to_text


def run_test_data(model: GPTModel, tokenizer: GPTTokenizer, device, cfg, limit=10):

    train_loader, test_loader, val_loader, val_data = create_data(
        "./assets/instruction-data.json", batch_size=1, device=device
    )

    samples = np.random.choice(len(val_data), min(limit, len(val_data)), replace=False)

    for i in samples:
        input, response = format_input(val_data[i])
        ids = gen_text(
            model,
            idx=text_to_token_ids(input, tokenizer).to(device),
            max_new_tokens=100,
            context_size=cfg.context_len,
            top_k=50,
            temperature=1.0,
            eos_id=50256,
        )

        print(
            "-" * 200,
            "\nLLM: \n",
            token_ids_to_text(ids.cpu(), tokenizer),
            "\n\nExpected Response:",
            response,
        )


def main():
    parser = ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--instruction", default="Rewrite the sentence using a simile.")
    parser.add_argument("--input", default="The car is very fast.")
    parser.add_argument(
        "--model-size",
        default="355M",
        choices=tuple(MODEL_CONFIGS),
    )
    args = parser.parse_args()

    input, _ = format_input(json.loads(f"""
    {{
        "instruction": "{args.instruction}",
        "input": "{args.input}",
        "output": ""
    }}
    """))

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

    device = torch.device("cuda")
    model.to(device)

    load_model(model, None, device, f"./data/trained/fine-tune/{model_size}.pth")

    tokenizer = GPTTokenizer()

    if args.test:
        run_test_data(model, tokenizer, device, cfg)
        return

    ids = gen_text(
        model,
        idx=text_to_token_ids(input, tokenizer).to(device),
        max_new_tokens=100,
        context_size=cfg.context_len,
        top_k=50,
        temperature=1.0,
        eos_id=50256,
    )

    print("LLM: \n", token_ids_to_text(ids.cpu(), tokenizer))


if __name__ == "__main__":
    main()
