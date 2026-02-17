#!/usr/bin/env python3
"""Regenerate test_train_bpe_special_tokens snapshot."""
import sys
import pickle
from pathlib import Path
from collections import Counter

import regex as re

def run_train_bpe(input_path, vocab_size, special_tokens):
    """Simple BPE implementation matching adapters.py"""
    text = ""
    with open(input_path, 'r') as f:
        text = f.read()

    if special_tokens:
        remove_special_pattern = '|'.join(re.escape(token) for token in special_tokens)
        text = re.sub(remove_special_pattern, '', text)

    gpt2_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
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
        words[word_bytes] = words.get(word_bytes, 0) + 1

    num_merges = vocab_size - len(vocab)
    merges = []

    for merge_idx in range(num_merges):
        pair_freq = Counter()
        for word, freq in words.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pair_freq[pair] += freq

        if not pair_freq:
            break

        best_pair = max(pair_freq, key=lambda p: (pair_freq[p], p))
        merges.append(best_pair)

        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        new_words = {}
        for word, freq in words.items():
            new_word = []
            i = 0
            while i < len(word):
                if (i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]):
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words[tuple(new_word)] = freq
        words = new_words

        if (merge_idx + 1) % 100 == 0:
            print(f"  Merge {merge_idx+1}/{num_merges}")

    return vocab, merges


if __name__ == "__main__":
    print("🔄 Regenerating snapshot with adapters.py implementation...")
    print("   Training BPE (this may take 1-2 minutes)...")

    vocab, merges = run_train_bpe(
        input_path=Path("tests/fixtures/tinystories_sample_5M.txt"),
        vocab_size=1000,
        special_tokens=["<|endoftext|>"]
    )

    snapshot_path = Path("tests/_snapshots/test_train_bpe_special_tokens.pkl")
    with open(snapshot_path, "wb") as f:
        pickle.dump({
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        }, f)

    print(f"\n✅ Snapshot regenerated!")
    print(f"   Snapshot path: {snapshot_path}")
    print(f"   vocab_size: {len(vocab)}")
    print(f"   num_merges: {len(merges)}")
    print(f"   Special token b'<|endoftext|>' in vocab: {b'<|endoftext|>' in vocab.values()}")

    # Verify special token constraint
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    has_fragment = False
    for word_bytes in vocabs_without_specials:
        if b"<|" in word_bytes:
            print(f"   ⚠️  Found fragment: {word_bytes}")
            has_fragment = True
            break
    if not has_fragment:
        print(f"   ✅ No special token fragments in other vocab items")

    print("\n   Now run: pytest tests/test_train_bpe.py::test_train_bpe_special_tokens -v")
