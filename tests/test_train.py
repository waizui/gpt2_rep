from pathlib import Path

from torch import mode
from transformer.config import GPTConfig
from transformer.tokenizer import GPTTokenizer
from transformer.train import gen_text, text_to_token_ids, token_ids_to_text, train


def test_train():
    file = Path(__file__).resolve().parents[1] / "assets" / "the-verdict.txt"
    prompt = "Every effort moves you"

    # file = Path(__file__).resolve().parents[1] / "assets" / "ruozhiba-post-annual.txt"
    # prompt = "如果我是一个"
    model, cfg = train(file, prompt)
    
    model.eval()
    model.cpu()
    tokenizer = GPTTokenizer()

    for input in ["Hello world!", "Every effort moves you", "An apple a day keeps the"]:
        ids = gen_text(
            model,
            idx=text_to_token_ids(input, tokenizer),
            max_new_tokens=25,
            context_size=cfg.context_len,
            top_k=50,
            temperature=1.0,
        )

        print("LLM: \n", token_ids_to_text(ids, tokenizer))
