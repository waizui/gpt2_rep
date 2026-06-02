import json
from os import path
import os
import sys

import tensorflow

from gpt_download import download_and_load_gpt2, load_gpt2_params_from_tf_ckpt
from load_gpt import load_weights_into_gpt
from transformer.config import GPTConfig
from transformer.gpt import GPTModel
from transformer.tokenizer import GPTTokenizer
from transformer.train import gen_text, text_to_token_ids, token_ids_to_text


def main():

    argc = len(sys.argv)
    input = "Every effort moves you"
    if argc > 1:
        input = sys.argv[1]

    weight_path = "./data/openai-gpt2/"
    model_size = "124M"

    param_dir = path.join(weight_path, model_size)

    if not path.exists(param_dir):
        download_and_load_gpt2(model_size, weight_path)

    cfg124m = GPTConfig()
    cfg124m.emb_dim = 768
    cfg124m.n_layers = 12
    cfg124m.n_heads = 12
    cfg124m.context_len = 1024
    cfg124m.qkv_bias = True

    cfgs = {"124M": cfg124m}
    cfg = cfgs[model_size]
    model = GPTModel(cfg)

    settings = json.load(
        open(os.path.join(param_dir, "hparams.json"), "r", encoding="utf-8")
    )

    tf_ckpt_path = tensorflow.train.latest_checkpoint(param_dir)
    settings = json.load(
        open(os.path.join(param_dir, "hparams.json"), "r", encoding="utf-8")
    )
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    load_weights_into_gpt(model, params)
    tokenizer = GPTTokenizer()
    ids = gen_text(
        model,
        idx=text_to_token_ids(input, tokenizer),
        max_new_tokens=25,
        context_size=cfg.context_len,
        top_k=50,
        temperature=1.0,
    )

    print("LLM: \n", token_ids_to_text(ids, tokenizer))


if __name__ == "__main__":
    main()
