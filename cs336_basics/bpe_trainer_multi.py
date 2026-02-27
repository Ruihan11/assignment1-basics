from collections import defaultdict
from typing import BinaryIO
from multiprocessing import Pool, cpu_count
import regex as re
import os

GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    Finds boundaries that align with special tokens to avoid splitting them.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # Calculate initial chunk boundaries (evenly spaced)
    chunk_size = file_size // num_chunks
    chunk_boundaries = [i * chunk_size for i in range(num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4KB at a time to find special tokens

    # Adjust each boundary to align with a special token
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            # If EOF, set boundary at end of file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Remove duplicate boundaries
    return sorted(set(chunk_boundaries))

def process_chunk(args):

    input_path, start, end, special_tokens = args
    text = ""
    with open(input_path, 'rb') as f:
        
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    
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

    chunk = {}
    for word in tokens:
        word_bytes = tuple(bytes([b]) for b in word.encode('utf-8'))
        chunk[word_bytes] = chunk.get(word_bytes, 0) + 1
    
    return chunk

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
 
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes = cpu_count(),
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    words = {}

    # imap_unordered does not out a full chunk like map does, it yield chunks every iteration
    with Pool(num_processes) as pool:
        chunks = pool.imap_unordered(process_chunk, chunk_args)
        for chunk in chunks:
            for k, v in chunk.items():
                words[k] = words.get(k, 0) + v

    # init pairs to word, optimize finding if word have best pair
    pair_to_words : dict[tuple[bytes, bytes], set[tuple[bytes]]] = defaultdict(set)
    for word, _ in words.items():
        for i in range(len(word)-1):
            pair_to_words[(word[i], word[i+1])].add(word)

    # 256 + special tokens
    vocab = vocab_init(special_tokens)

    num_merges = vocab_size - len(vocab)
    merges = []

    # init pair-and-freq
    pair_freq = defaultdict(int)
    for word, freq in words.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_freq[pair] += freq
    
    for _ in range(num_merges):    
        if not pair_freq:
            break

        best_pair = max(pair_freq, key= lambda p: (pair_freq[p], p))
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        affected_words = list(pair_to_words[best_pair[0], best_pair[1]])

        for word in affected_words:
            """
            for each affected word - word containing the best pair,
            merge the best pair into a new word
            remove the affected word from the dict
            renew pair to words
            renew pair to freq
            """

            freq = words[word]
            new_word = []
            i = 0

            while i < len(word):
                if (i<len(word)-1 and word[i]==best_pair[0] and word[i+1]==best_pair[1]):
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            # edit the words directly, remove old word pairs, add new's 
            new_word = tuple(new_word)
            del words[word]
            words[new_word] = freq

            for i in range(len(word)-1):
                pair_to_words[word[i], word[i+1]].discard(word)

            for i in range(len(new_word)-1):
                pair_to_words[new_word[i], new_word[i+1]].add(new_word)

            for j, token in enumerate(new_word):
                """
                (x h e l l o) -> (x he l l o)
                remove (x,h) (e,l) (he) 
                add (x,he) (he,l) 
                """
                if token!=new_token:
                    continue

                pair_freq[best_pair] -=freq

                if j > 0: # have left neighbor
                    old_left = best_pair[1] if new_word[j-1]==new_token else new_word[j-1]
                    pair_freq[old_left, best_pair[0]] -= freq
                    pair_freq[new_word[j-1], new_token] += freq
                    
                if j < len(new_word)-1 and new_word[j+1]!=new_token: # have right neighbor
                    pair_freq[best_pair[1], new_word[j+1]] -= freq
                    pair_freq[new_token, new_word[j+1]] += freq

        pair_freq = defaultdict(int, {p: f for p, f in pair_freq.items() if f > 0})

    return vocab, merges

def main():
    vocab, merge = run_train_bpe(
        input_path="/Users/ryanli/Documents/repos/assignment1-basics/cs336_basics/data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        num_processes=10,
    )

if __name__ == '__main__':
    main()