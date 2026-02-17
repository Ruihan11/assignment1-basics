import regex as re
import os, time, heapq, pickle
from multiprocessing import Pool, cpu_count
from collections import Counter, defaultdict
from typing import BinaryIO

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = None,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    text = ""
    with open(input_path, 'r') as f:
        text = f.read()

    gpt2_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # Split text by special tokens to prevent them from being merged during BPE
    if special_tokens:
        special_pattern = '|'.join(re.escape(st) for st in special_tokens)
        parts = re.split(f'({special_pattern})', text)
        tokens = []
        for part in parts:
            if part in special_tokens:
                # Skip special tokens - they shouldn't be part of BPE training
                continue
            else:
                # Apply GPT-2 pattern to regular text
                subparts = re.findall(gpt2_pattern, part)
                tokens.extend(subparts)
    else:
        # No special tokens, just apply GPT-2 pattern directly
        tokens = re.findall(gpt2_pattern, text)
    
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    special_token_bytes = [st.encode('utf-8') for st in special_tokens]
    for special_token in special_token_bytes:
        vocab[len(vocab)] = special_token

    words = {}
    for word in tokens:
        word_bytes = tuple(bytes([b]) for b in word.encode('utf-8'))
        # print(word_bytes)
        words[word_bytes] = words.get(word_bytes, 0) + 1

    num_merges = vocab_size - len(vocab)
    merges = []

    for _ in range(num_merges):
        pair_freq = Counter()
        for word, freq in words.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pair_freq[pair] += freq

        # if not pair_freq: break

        best_pair = max(pair_freq, key= lambda p: (pair_freq[p], p))
        merges.append(best_pair)

        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        new_words = {}
        for word, freq in words.items():
            new_word = []
            i = 0
            while i < len(word):
                if (i<len(word)-1 and word[i]==best_pair[0] and word[i+1]==best_pair[1]):
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            # print(new_word)
            new_words[tuple(new_word)] = freq
        words = new_words

    return vocab, merges

 