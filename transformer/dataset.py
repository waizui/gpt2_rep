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

    def __getitem__(self, index):
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
