from collections import defaultdict
import regex as re
import os

GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def load_n_tokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    **kwargs,
) -> list[str]:
    
    text = ""
    with open(input_path, 'r') as f:
        text = f.read()

    # Split text by special tokens to prevent them from being merged during BPE
    if special_tokens:
        special_set =set(special_tokens)
        special_pattern = '|'.join(re.escape(st) for st in special_tokens)
        parts = re.split(f'({special_pattern})', text)
        tokens = []
        for part in parts:
            if part not in special_set:
                tokens.extend(re.findall(GPT2_PATTERN, part))
    else:
        # No special tokens, just apply GPT-2 pattern directly
        tokens = re.findall(GPT2_PATTERN, text)
    
    return tokens

def vocab_init(
    special_tokens: list[str],
    **kwargs,
) -> dict[int, bytes]:
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    
    special_token_bytes = [st.encode('utf-8') for st in special_tokens]
    for special_token in special_token_bytes:
        vocab[len(vocab)] = special_token
    
    return vocab

def words_to_freq(tokens) -> dict[tuple[bytes], int]: 
    words = {}
    for word in tokens:
        word_bytes = tuple(bytes([b]) for b in word.encode('utf-8'))
        words[word_bytes] = words.get(word_bytes, 0) + 1
    return words
 
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # txt to special_tokens divided list[str]
    tokens = load_n_tokenize(input_path, special_tokens)

    # 256 + special tokens
    vocab = vocab_init(special_tokens)

    # {tuple(byte expression of each letter in one word), word freqency}
    words = words_to_freq(tokens)

    num_merges = vocab_size - len(vocab)
    merges = []

    # init pair-and-freq
    pair_freq = defaultdict(int)
    for word, freq in words.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_freq[pair] += freq
    
    for _ in range(num_merges):
        if len(vocab) == 322:
            breakpoint()

        if not pair_freq:
            break

        best_pair = max(pair_freq, key= lambda p: (pair_freq[p], p))
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        new_words = {}
        for word, freq in words.items():
            if not any(word[i] == best_pair[0] and word[i+1] == best_pair[1] for i in range(len(word)-1)):
                new_words[word] = freq
                continue

            new_word = []
            i = 0
            while i < len(word):
                if (i<len(word)-1 and word[i]==best_pair[0] and word[i+1]==best_pair[1]):
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            new_words[new_word] = words.get(new_word, 0) + freq

            for j, token in enumerate(new_word):
                if token!=new_token:
                    continue

                """
                (x h e l l o) -> (x he l l o)
                remove (x,h) (e,l) (he) 
                add (x,he) (he,l) 
                """
                pair_freq[best_pair] -=freq

                if j > 0: # have left neighbor
                    old_left = best_pair[1] if new_word[j-1]==new_token else new_word[j-1]
                    pair_freq[old_left, best_pair[0]] -= freq
                    pair_freq[new_word[j-1], new_token] += freq
                    
                if j < len(new_word)-1 and new_word[j+1]!=new_token: # have right neighbor
                    pair_freq[best_pair[1], new_word[j+1]] -= freq
                    pair_freq[new_token, new_word[j+1]] += freq

        pair_freq = defaultdict(int, {p: f for p, f in pair_freq.items() if f > 0})
        words = new_words

    return vocab, merges



def main():
    vocab, merge = run_train_bpe(
        input_path="/Users/ryanli/Documents/repos/assignment1-basics/cs336_basics/data/tinystories_sample_5M.txt",
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
        num_processes=10,
    )

if __name__ == '__main__':
    main()
