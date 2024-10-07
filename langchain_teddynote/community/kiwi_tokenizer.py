import string
from typing import List, Optional
from kiwipiepy import Kiwi
import nltk


class KiwiBM25Tokenizer:
    def __init__(self, stop_words: Optional[List[str]] = None):
        self._setup_nltk()
        self._stop_words = set(stop_words) if stop_words else set()
        self._punctuation = set(string.punctuation)
        self._tokenizer = self._initialize_tokenizer()

    @staticmethod
    def _initialize_tokenizer() -> Kiwi:
        return Kiwi()

    @staticmethod
    def _tokenize(tokenizer: Kiwi, text: str) -> List[str]:
        return [token.form for token in tokenizer.tokenize(text)]

    @staticmethod
    def _setup_nltk() -> None:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

    def __call__(self, text: str) -> List[str]:
        tokens = self._tokenize(self._tokenizer, text)
        return [
            word.lower()
            for word in tokens
            if word not in self._punctuation and word not in self._stop_words
        ]

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_tokenizer"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokenizer = self._initialize_tokenizer()


class KiwiTokenizer:
    def __init__(self):
        self.kiwi = Kiwi()

    def tokenize(self, text, type="list"):
        if type == "list":
            return [token.form for token in self.kiwi.tokenize(text)]
        else:
            return " ".join([token.form for token in self.kiwi.tokenize(text)])
