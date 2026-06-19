from argparse import ArgumentParser

from transformer.config import  GPTConfig
from transformer.gpt import GPTModel
from transformer.io import load_model
from transformer.tokenizer import GPTTokenizer
from transformer.train import gen_text, text_to_token_ids, token_ids_to_text
from transformer.utils import resolve_device


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", default="Every effort moves you")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context-length", type=int, default=256)
    args = parser.parse_args()

    input = args.input

    cfg = GPTConfig()
    cfg.context_len = args.context_length
    cfg.qkv_bias = True
    model = GPTModel(cfg)
    print(f"model param num: {model.param_num()}")

    device = resolve_device(args.device)
    model.to(device)

    load_model(model, None, device, "./data/trained/pre-trained.pth")

    tokenizer = GPTTokenizer()

    ids = gen_text(
        model,
        idx=text_to_token_ids(input, tokenizer).to(device),
        max_new_tokens=50,
        context_size=cfg.context_len,
        top_k=50,
        temperature=1.0,
        eos_id=50256,
    )

    print("LLM: \n", token_ids_to_text(ids.cpu(), tokenizer))


if __name__ == "__main__":
    main()
