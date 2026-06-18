from argparse import ArgumentParser
import json

import torch
import torch.nn as nn
from torch import Tensor, cuda
from torch.optim.optimizer import Optimizer
from torch.types import Device

from load_gpt import load_gpt_settings_params, load_weights_into_gpt
from transformer.config import MODEL_CONFIGS, GPTConfig
from transformer.dataset import create_instruction_dataloader, format_input
from transformer.gpt import GPTModel
from transformer.io import save_model
from transformer.tokenizer import GPTTokenizer
from transformer.train import (
    gen_text,
    text_to_token_ids,
    token_ids_to_text,
    train_simple,
)


def create_data(file, batch_size, device):
    with open(file, "r", encoding="utf-8") as file:
        text_data = json.load(file)

    train_portion = int(0.85 * len(text_data))
    test_portion = int(0.1 * len(text_data))

    train_data = text_data[:train_portion]
    test_data = text_data[train_portion : train_portion + test_portion]
    val_data = text_data[train_portion + test_portion :]

    print(
        f"Training set length: {len(train_data)}\n"
        f"Test set length: {len(test_data)} \n"
        f"Validation set length: {len(val_data)}\n"
    )

    train_loader = create_instruction_dataloader(train_data, batch_size, device=device)
    test_loader = create_instruction_dataloader(test_data, batch_size, device=device)
    val_loader = create_instruction_dataloader(val_data, batch_size, device=device)
    return train_loader, test_loader, val_loader, val_data[0]


def fine_tune(file, model: GPTModel, optimizer: Optimizer, device, batch_size: int):
    torch.manual_seed(123)

    num_epochs = 2

    train_loader, test_loader, val_loader, val_data0 = create_data(
        file, batch_size=batch_size, device=device
    )

    input, _ = format_input(val_data0)

    model.to(device)

    train_losses, val_losses, token_seen = train_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context=input,
        tokenizer=GPTTokenizer(),
    )

    return model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file", default="./assets/instruction-data.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--model-size",
        default="124M",
        choices=tuple(MODEL_CONFIGS),
    )
    args = parser.parse_args()

    file = args.file
    model_size = args.model_size
    batch_size = args.batch_size

    model_config = MODEL_CONFIGS[model_size]

    cfg = GPTConfig()
    cfg.emb_dim = model_config["emb_dim"]
    cfg.n_layers = model_config["n_layers"]
    cfg.n_heads = model_config["n_heads"]
    cfg.context_len = 1024
    cfg.qkv_bias = True
    model = GPTModel(cfg)

    _, params = load_gpt_settings_params(model_size)
    load_weights_into_gpt(model, params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    device = torch.device("cuda")
    fine_tune(file, model, optimizer, device, batch_size)

    save_model(model, optimizer, f"./data/trained/fine-tune/{model_size}.pth")
