import re
import random
import unicodedata
from collections import Counter
from typing import List, Dict, Iterable


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


class Vocab:
    def __init__(self, specials=None):
        specials = specials or []
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []
        for sp in specials:
            self.add(sp)

    def add(self, token: str) -> int:
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)
        return self.stoi[token]

    def build(self, tokens: Iterable[str], min_freq: int = 1):
        counter = Counter(tokens)
        for tok, freq in counter.items():
            if freq >= min_freq:
                self.add(tok)

    def encode(self, token: str, unk_token: str = "<UNK>") -> int:
        if token in self.stoi:
            return self.stoi[token]
        return self.stoi.get(unk_token, 0)

    def decode(self, idx: int) -> str:
        return self.itos[idx]

    def __len__(self) -> int:
        return len(self.itos)
