import torch
import torch.nn as nn
from torch import Tensor
from torch.types import Device
from torch.utils.data import DataLoader

from transformer import tokenizer
from transformer.config import GPTConfig
from transformer.dataset import create_dataloader
from transformer.gpt import GPTModel
from transformer.tokenizer import GPTTokenizer, Tokenizer


def gen_text(
    model: GPTModel,
    idx: Tensor,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
    eos_id=None,
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits: Tensor = model(idx_cond)

        logits = logits[:, -1, :]
        if top_k is not None:
            top_k_logits, _ = torch.topk(logits, top_k)
            min_val = top_k_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # break if encounter end of sequence id
        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def text_to_token_ids(text, tokenizer: Tokenizer) -> Tensor:
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)


def token_ids_to_text(ids: Tensor, tokenizer: Tokenizer) -> str:
    return tokenizer.decode(ids.squeeze(0).tolist())


def calc_loss_batch(
    input_batch: Tensor, target_batch: Tensor, model: GPTModel, device: Device
):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)

    # logits : [b,num_tokens, vocab_size] -> [b*num_tokens, vocab_size]
    # target : [b,num_tokens]->[b*num_tokens]
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

    return loss


def calc_loss_loader(
    loader: DataLoader, model: GPTModel, device: Device, num_batches=None
):

    total_loss = 0

    if len(loader) == 0:
        return float("nan")
    elif num_batches == None:
        num_batches = len(loader)
    else:
        num_batches = min(num_batches, len(loader))

    for i, (input_batch, target_batch) in enumerate(loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss

    return total_loss / num_batches


def train_simple(
    model: GPTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: Device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer: Tokenizer,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            # evaluate, optional
            if global_step % eval_freq == 0:
                train_loss, val_loss = eval_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Epoch {epoch} (Setp {global_step:06d}): "
                    f"Train loss {train_loss:.3f} "
                    f"Val loss {val_loss:.3f} "
                )
        gen_and_print_simple(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


def eval_model(
    model: GPTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: Device,
    eval_iter,
):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)

    model.train()
    return train_loss, val_loss


def gen_and_print_simple(
    model: GPTModel, tokenizer: Tokenizer, device: Device, start_conetxt: str
):
    model.eval()
    context_size = model.emb.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_conetxt, tokenizer).to(device)
    with torch.no_grad():
        token_ids = gen_text(
            model, encoded, 50, context_size, temperature=1.0, top_k=25
        )

    text = token_ids_to_text(token_ids, tokenizer)
    print(text.replace("\n", " "))
    model.train()


def create_data(file, cfg: GPTConfig, batch_size):
    with open(file, "r", encoding="utf-8") as file:
        text_data = file.read()

    ratio = 0.9
    split_idx = int(ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_lader = create_dataloader(
        train_data, batch_size, cfg.context_len, stride=cfg.context_len, num_workers=0
    )
    val_lader = create_dataloader(
        val_data, batch_size, cfg.context_len, stride=cfg.context_len, num_workers=0
    )

    return train_lader, val_lader


def train(file, start_context) -> tuple[GPTModel, GPTConfig]:
    torch.manual_seed(123)
    cfg = GPTConfig()
    cfg.context_len = 256
    model = GPTModel(cfg)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10

    train_loader, val_loader = create_data(file, cfg, batch_size=2)

    device = torch.device("cuda")

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
        start_context=start_context,
        tokenizer=GPTTokenizer(),
    )

    return model, cfg
