import string
import nltk
from typing import List
from kiwipiepy import Kiwi


class KiwiBM25Tokenizer:
    def __init__(
        self,
        stop_words: List[str] = None,
    ):
        self.nltk_setup()
        self._stop_words = list(set(stop_words)) if stop_words else None
        self._punctuation = set(string.punctuation)
        self._tokenizer = Kiwi()

    def initialize_tokenizer(self):
        self._tokenizer = Kiwi()

    @staticmethod
    def korean_tokenize(tokenizer, text: str) -> List[str]:
        return [token.form for token in tokenizer.tokenize(text)]

    @staticmethod
    def nltk_setup() -> None:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

    def __call__(self, text: str) -> List[str]:
        tokens = KiwiBM25Tokenizer.korean_tokenize(self._tokenizer, text)
        tokens = [word.lower() for word in tokens]
        tokens = [word for word in tokens if word not in self._punctuation]
        tokens = [word for word in tokens if word not in self._stop_words]
        return tokens

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_tokenizer"]  # Remove the unpickleable entry
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.initialize_tokenizer()  # Reinitialize the tokenizer
