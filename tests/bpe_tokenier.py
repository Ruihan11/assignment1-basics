from typing import Iterable, Iterator
import regex as re
import pickle

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        vocab: dict[int: bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        reversered_vocab: dict[bytes: int] 
        merge_priority: dict[tuple[bytes, bytes] : int]
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.reversered_vocab = {v: k for k, v in vocab.items()}
        self.merge_priority = {merge: i for i, merge in enumerate(merges)}
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)
    
    def _merge_bytes(self, tokens_bytes) -> list[int]:
        output = []
        for token_bytes in tokens_bytes:
            # print(f"before : {token_bytes}")
            # stop when each word is fully merged or no more merge in merges can be found
            while(len(token_bytes)>1):
                best_pair = None
                best_idx = None
                best_priority = float('inf')

                # find the most prioritized pair
                for i in range(len(token_bytes)-1):
                    pair = (token_bytes[i], token_bytes[i+1])
                    if pair in self.merges:
                        priority = self.merge_priority.get(pair)
                        if priority < best_priority:
                            best_pair = pair
                            best_idx = i
                            best_priority = priority

                # skip if not found
                if best_pair is None: break

                # merge
                merged = token_bytes[best_idx] + token_bytes[best_idx+1]
                token_bytes = token_bytes[:best_idx] + [merged] + token_bytes[best_idx+2:]

            # print(f"after : {token_bytes}")    
            for i in token_bytes:
                output.append(self.reversered_vocab.get(i))

        return output
    
    def encode(self, text: str) -> list[int]:
        # use gpt2 pattern to divide string
        PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # Handle special tokens first (if any)
        if self.special_tokens:
            # Sort by length descending to match longer tokens first
            # e.g., "<|endoftext|><|endoftext|>" should match before "<|endoftext|>"
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            # Build regex pattern that escapes special chars, with capturing group
            special_pattern = '(' + '|'.join(re.escape(t) for t in sorted_special) + ')'

            output = []
            # Split text by special tokens, keeping the delimiters
            parts = re.split(special_pattern, text)

            for part in parts:
                if not part:
                    continue
                if part in self.special_tokens:
                    # Special token: look up directly in vocab
                    special_bytes = part.encode('utf-8')
                    token_id = self.reversered_vocab.get(special_bytes)
                    if token_id is not None:
                        output.append(token_id)
                else:
                    # Regular text: apply GPT-2 regex + BPE
                    pretokens = re.findall(PAT, part)
                    tokens_bytes = []
                    for pretoken in pretokens:
                        tokens = [bytes([b]) for b in pretoken.encode('utf-8')]
                        tokens_bytes.append(tokens)
                    output.extend(self._merge_bytes(tokens_bytes))
            return output

        # No special tokens: original behavior
        pretokens = re.findall(PAT, text)

        # list[each_word[b'each byte']]
        tokens_bytes = []
        for pretoken in pretokens:
            tokens = [bytes([b]) for b in pretoken.encode('utf-8')]
            tokens_bytes.append(tokens)

        output = self._merge_bytes(tokens_bytes)

        return output

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:

        for line in iterable:
            ids = self.encode(line)
            for id in ids:
                yield id

    def decode(self, ids: list[int]) -> str:
        # Concatenate all raw bytes first, then decode once
        # This properly handles multi-byte UTF-8 sequences split across tokens
        byte_sequence = b""
        for id in ids:
            token_bytes = self.vocab.get(id)
            if token_bytes is not None:
                byte_sequence += token_bytes
        return byte_sequence.decode('utf-8', errors='replace')