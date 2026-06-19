
## Introduction

This repo is the source code of my posts: [Build a GPT-2](https://waizui.github.io/posts/gpt/gpt_i.html)

Mostly hand coded, may contains many mistakes. 


## Requirements
python version should be `>=3.10,<3.13`.

run `pip install -r requirements.txt` will install dependencies with pip.

or run `uv sync` will install dependencies from `pyproject.toml` and `uv.lock`.

## Usage

### Self-train a model
run `python -m transformer.train` will pre-train a small gpt model.

run `python run_pre_trained_gpt.py ` will load the trained samll gpt model and generate text.

### Download a GPT-2 model
run `python gpt_download.py` will downlad openai-gpt2 weights to `./data/openai-gpt2`.

run `python run_gpt.py` will load openai-gpt2 weights and generate text.

### Fine-tuning a downloaded GPT-2 model
run `python -m transformer.fine_tune` will fine-tune the model with downloaded openai-gpt2 weights.

run `python run_sft_gpt.py` will load fine-tuned gpt and generate instruction response.

run `python run_sft_gpt.py --test` will test fine-tuned gpt on random validation data.


`--device` default is `cuda`. If cuda is not available cpu will be used. 

other cli argumens, please refer to code.
