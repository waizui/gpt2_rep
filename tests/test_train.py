from pathlib import Path
from transformer.train import train


def test_train():
    file = Path(__file__).resolve().parents[1] / "assets" / "the-verdict.txt"
    prompt = "Every effort moves you"


    # file = Path(__file__).resolve().parents[1] / "assets" / "ruozhiba-post-annual.txt"
    # prompt = "如果我"
    train(file, prompt)
