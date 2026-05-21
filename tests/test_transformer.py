from transformer.tokenizer import SimpleTokenizer


def test_preprocess():
    text = "Hello, world. Is this-- a test?"
    tokenizer = SimpleTokenizer()

    preprocessed = tokenizer.preprocess(text)

    assert preprocessed == [
        "Hello",
        ",",
        "world",
        ".",
        "Is",
        "this",
        "--",
        "a",
        "test",
        "?",
    ]


def test_encode():
    tokenizer = SimpleTokenizer()
    tokenizer.load("Hello, world. Is this-- a test?")

    assert tokenizer.encode("Hello, test?") == [4, 0, 7, 3]


def test_decode():
    tokenizer = SimpleTokenizer()
    tokenizer.load("Hello, world. Is this-- a test?")

    assert tokenizer.decode([4, 0, 7, 3]) == "Hello, test?"
