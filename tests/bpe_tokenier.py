from typing import Iterable, Iterator
import regex as re
import pickle

GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, 
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens:list[str] | None = None):
        
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.merges_priority = {merge: i for i, merge in enumerate(self.merges)}
        self.reversed_vocab = {k: v for v, k in vocab.items()}

    @classmethod
    def from_files(cls, 
                   vocab_filepath: str, 
                   merges_filepath: str, 
                   special_tokens: list[str] | None = None):

        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """
        split tokens into gpt2 pattern's pretokens
        apply merges and save its/their ids
        """
        if self.special_tokens:
                sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
                special_set = set(sorted_special_tokens)
                special_pattern = "|".join(re.escape(st) for st in sorted_special_tokens)
                parts = re.split(f"({special_pattern})", text)
                pretokens = []
                for part in parts:
                    if part in special_set:
                        pretokens.append(part)
                    elif part:
                        pretokens.extend(re.findall(GPT2_PATTERN, part))
        else:
            # No special tokens, just apply GPT-2 pattern directly
            pretokens = re.findall(GPT2_PATTERN, text)

        ids = []
        
        for pretoken in pretokens:
            if self.special_tokens and pretoken in self.special_tokens:
                ids.append(self.reversed_vocab[pretoken.encode("utf-8")])
            else:
                # pretoken -> bytes -> merge -> ids
                # bytes list for each word
                pretoken_bytes = [bytes([b]) for b in pretoken.encode("utf-8")]
                merged_tokens = self._merge_bytes(pretoken_bytes)
                for merged_token in merged_tokens:
                    ids.append(self.reversed_vocab[merged_token])

        return ids

    def _merge_bytes(self, pretoken_bytes) -> list[bytes]:

        while len(pretoken_bytes) > 1:
            best_pair = None
            best_idx = None
            best_priority = float('inf')

            for i in range(len(pretoken_bytes) - 1):
                pair = (pretoken_bytes[i], pretoken_bytes[i+1])
                priority = self.merges_priority.get(pair)

                if priority is not None and priority < best_priority:
                    best_pair = pair
                    best_idx = i
                    best_priority = priority

            if best_pair is None:
                break

            pretoken_bytes[best_idx] = pretoken_bytes[best_idx] + pretoken_bytes[best_idx+1]
            pretoken_bytes.pop(best_idx+1)
        
        return pretoken_bytes
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[i] for i in ids)
        return token_bytes.decode("utf-8",errors="replace")
    