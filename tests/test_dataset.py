from pathlib import Path

import torch

from transformer.dataset import GPTDataset, create_dataloader
from transformer.tokenizer import GPTTokenizer


ASSET_TEXT = Path(__file__).resolve().parents[1] / "assets" / "the-verdict.txt"


def _load_asset_text() -> str:
    text = ASSET_TEXT.read_text(encoding="utf-8")
    assert len(text) > 0
    return text


def test_gpt_dataset_loads_real_asset_text():
    text = _load_asset_text()
    tokenizer = GPTTokenizer()
    max_len = 8
    stride = 4

    ids = tokenizer.encode(text)
    dataset = GPTDataset(text, tokenizer, max_len=max_len, stride=stride)
    input_ids, target_ids = dataset[0]

    assert len(dataset) == len(range(0, len(ids) - max_len, stride))
    assert input_ids.tolist() == ids[:max_len]
    assert target_ids.tolist() == ids[1 : max_len + 1]


def test_create_dataloader_loads_real_asset_text():
    text = _load_asset_text()
    tokenizer = GPTTokenizer()
    batch_size = 2
    max_len = 8
    stride = 4

    loader = create_dataloader(
        text,
        batch_size=batch_size,
        max_len=max_len,
        stride=stride,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    input_batch, target_batch = next(iter(loader))
    ids = tokenizer.encode(text)

    assert tuple(input_batch.shape) == (batch_size, max_len)
    assert tuple(target_batch.shape) == (batch_size, max_len)
    assert input_batch[0].tolist() == ids[:max_len]
    assert target_batch[0].tolist() == ids[1 : max_len + 1]
    assert torch.equal(input_batch[:, 1:], target_batch[:, :-1])
