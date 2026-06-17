import torch
from torch.optim.optimizer import Optimizer

from transformer.gpt import GPTModel


def save(model: GPTModel, optimizer: Optimizer, path: str):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load(model: GPTModel, optimizer: Optimizer, device: torch.device, path: str):
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
