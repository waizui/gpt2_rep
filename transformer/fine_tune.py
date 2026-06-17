import torch
import torch.nn as nn
from torch import Tensor
from torch.types import Device
from torch.utils.data import DataLoader

from transformer.config import GPTConfig
from transformer.gpt import GPTModel
from transformer.tokenizer import GPTTokenizer, Tokenizer


def create_data(file, cfg: GPTConfig, batch_size):
    with open(file, "r", encoding="utf-8") as file:
        text_data = file.read()

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

    train_lader = create_dataloader(
        train_data, batch_size, cfg.context_len, stride=cfg.context_len, num_workers=0
    )
    test_lader = create_dataloader(
        test_data, batch_size, cfg.context_len, stride=cfg.context_len, num_workers=0
    )

    val_loader = create_dataloader(
        val_data, batch_size, cfg.context_len, stride=cfg.context_len, num_workers=0
    )

    return train_lader, test_lader, val_loader


def train():
    pass
