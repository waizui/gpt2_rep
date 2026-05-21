from transformer.tokenizer import PreProcessor


def test_preprocess():
    text = "Hello, world. Is this-- a test?"
    preprocessor = PreProcessor()

    preprocessed = preprocessor.preprocess(text)

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
    preprocessor = PreProcessor()
    preprocessor.load("Hello, world. Is this-- a test?")

    assert preprocessor.encode("Hello, test?") == [0, 1, 8, 9]


def test_decode():
    preprocessor = PreProcessor()
    preprocessor.load("Hello, world. Is this-- a test?")

    assert preprocessor.decode([0, 1, 8, 9]) == "Hello, test?"
