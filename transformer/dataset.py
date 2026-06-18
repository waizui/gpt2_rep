from functools import partial
from logging import error
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader

from transformer.tokenizer import GPTTokenizer, Tokenizer


class GPTDataset(Dataset):

    def __init__(self, text, tokenizer: Tokenizer, max_len, stride) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        ids = tokenizer.encode(text)

        for i in range(0, len(ids) - max_len, stride):
            input = ids[i : i + max_len]
            target = ids[i + 1 : i + max_len + 1]
            self.input_ids.append(torch.tensor(input))
            self.target_ids.append(torch.tensor(target))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.target_ids[index]


def create_dataloader(
    text,
    batch_size=4,
    max_len=256,
    stride=128,
    shuffle=True,
    drop_last=True,  # delete remaining data if less than batch size
    num_workers=0,
) -> DataLoader:
    if len(text) < max_len:
        raise Exception("error: Text too short")
    tokenizer = GPTTokenizer()
    dataset = GPTDataset(text, tokenizer, max_len, stride)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return loader


def format_input(entry):
    """
    Alpaca style

    entry:
    {
        "instruction":
        "input":
        "output":
    },
    """

    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    response_text = f"\n\n### Response:\n{entry['output']}"

    return instruction_text + input_text, response_text


def custom_collate_fn(
    batch,  # [batch_size, instruction_length]
    pad_token_id=50256,  # vocab size
    ignore_index=-100,
    allowed_max_len=None,
    device="cuda",
):
    batch_max_len = max(len(item) + 1 for item in batch)
    input_lst, target_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item.append(pad_token_id)

        padded = new_item + [pad_token_id] * (batch_max_len - len(new_item))

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            # fill all padding tokens with ignore_index except the first one
            targets[indices[1:]] = ignore_index

        if allowed_max_len is not None:
            inputs = inputs[:allowed_max_len]
            targets = targets[:allowed_max_len]

        input_lst.append(inputs)
        target_lst.append(targets)

    inputs_tensor = torch.stack(input_lst).to(device)
    targets_tensor = torch.stack(target_lst).to(device)

    return inputs_tensor, targets_tensor


class InstructionDataset(Dataset):

    def __init__(self, data, tokenizer: Tokenizer) -> None:
        self.data = data
        self.encoded_texts = []
        for entry in data:
            input, response = format_input(entry)
            full_text = input + response
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def create_instruction_dataloader(
    text,
    batch_size=4,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    device="cuda",
) -> DataLoader:
    tokenizer = GPTTokenizer()
    dataset = InstructionDataset(text, tokenizer)

    # prefill params
    collate_fn = partial(custom_collate_fn, allowed_max_len=512, device=device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return loader
