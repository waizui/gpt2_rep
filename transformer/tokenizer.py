from abc import ABC, abstractmethod
import re
import tiktoken


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        pass


class SimpleTokenizer(Tokenizer):
    str_to_int: dict[str, int]
    int_to_str: dict[int, str]

    def __init__(self) -> None:
        self.re_split = re.compile(r'([,.:;?_!"()\']|--|\s)')
        # remove blank space before certain symbols
        self.re_sub = re.compile(r'\s+([,.?!"()\'])')

    def load_from_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            self.load(text)

    def load(self, text):
        p = self.preprocess(text)
        words = sorted(set(p))
        self.str_to_int = {item: i for i, item in enumerate(words)}
        self.int_to_str = {i: item for item, i in self.str_to_int.items()}

    def preprocess(self, raw_text: str) -> list[str]:
        split = self.re_split.split(raw_text)
        preprocessed = [item.strip() for item in split if item.strip()]
        return preprocessed

    def encode(self, text) -> list[int]:
        p = self.preprocess(text)
        ids = [self.str_to_int[item] for item in p]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int_to_str[id] for id in ids])
        # replace with 1st captured group
        text = self.re_sub.sub(r"\1", text)
        return text


class GPTTokenizer(Tokenizer):
    def __init__(self) -> None:
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text) -> list[int]:
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)
