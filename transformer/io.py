import os

import torch
from torch.optim.optimizer import Optimizer

from transformer.gpt import GPTModel


def save_model(model: GPTModel, optimizer: Optimizer, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_model(
    model: GPTModel, optimizer: Optimizer | None, device: torch.device, path: str
):
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
