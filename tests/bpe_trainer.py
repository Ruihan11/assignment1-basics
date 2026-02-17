import regex as re
import os
import time
import heapq
from multiprocessing import Pool, cpu_count
from collections import Counter, defaultdict
from typing import BinaryIO

class ReversedPair:
    """Wrapper for pair tuples that reverses comparison order.

    Used in min-heap to achieve max-behavior for lexicographic tie-breaking:
    when frequencies are equal, we want the lexicographically LARGEST pair,
    but heapq is a min-heap. This wrapper reverses __lt__ so smaller wrapper
    means larger pair.
    """
    __slots__ = ('pair',)

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair  # Reversed for max-heap behavior

    def __eq__(self, other):
        return self.pair == other.pair

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
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
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
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


def pretokenize_chunk(args):
    """
    Pre-tokenize a chunk of the input file using GPT-2 tokenization pattern.
    Returns a Counter of word frequencies for this chunk.
    """
    input_path, start, end, special_tokens = args

    # GPT-2 tokenization pattern - splits text into meaningful units:
    # - Contractions ('s, 't, 're, 've, 'm, 'll, 'd)
    # - Words (with optional leading space)
    # - Numbers (with optional leading space)
    # - Punctuation/symbols (with optional leading space)
    # - Whitespace sequences
    gpt2_pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # Read the chunk from file
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # Split text by special tokens to prevent them from being merged during BPE
    if special_tokens:
        special_pattern = '|'.join(re.escape(st) for st in special_tokens)
        parts = re.split(f'({special_pattern})', chunk)

        words = []
        for part in parts:
            if part in special_tokens:
                # Skip special tokens - they shouldn't be part of BPE training
                continue
            else:
                # Apply GPT-2 pattern to regular text
                subparts = gpt2_pattern.findall(part)
                for subpart in subparts:
                    if subpart:
                        words.append(subpart)
    else:
        # No special tokens, just apply GPT-2 pattern directly
        words = gpt2_pattern.findall(chunk)

    return Counter(words)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = None,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE (Byte Pair Encoding) tokenizer on the input file.
    
    Args:
        input_path: Path to the training text file
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to preserve (e.g., ["<|endoftext|>"])
        num_processes: Number of parallel processes for pre-tokenization
    
    Returns:
        vocab: Dictionary mapping token IDs to byte sequences
        merges: List of merge operations as (token1, token2) tuples
    """
    total_start = time.time()

    if num_processes is None:
        num_processes = cpu_count()

    # Initialize vocabulary with all single byte values (0-255)
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    # Add special tokens to vocabulary
    special_token_bytes = [st.encode('utf-8') for st in special_tokens]
    for special_token in special_token_bytes:
        vocab[len(vocab)] = special_token

    # Split file into chunks for parallel processing
    # Use many more chunks than processes for better load balancing and memory efficiency
    # Smaller chunks = lower peak memory per process + faster individual processing
    t0 = time.time()
    num_chunks = num_processes * 16  # ~64 chunks for 4 processes = ~170MB each for 11GB file
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, special_token_bytes[0])
    chunk_time = time.time() - t0

    # Create tasks for multiprocessing
    tasks = [(input_path, start, end, special_tokens)
             for start, end in zip(boundaries[:-1], boundaries[1:])]

    # Pre-tokenize chunks in parallel with progress tracking
    t0 = time.time()
    with Pool(num_processes, maxtasksperchild=4) as pool:
        print(f"Pre-tokenizing {len(tasks)} chunks across {num_processes} processes...")
        chunk_word_counts = []
        # Use imap_unordered for ~30% speedup (no ordering overhead)
        for i, result in enumerate(pool.imap_unordered(pretokenize_chunk, tasks, chunksize=1)):
            chunk_word_counts.append(result)
            # Show progress every 10% or for small task counts
            if len(tasks) <= 10 or (i + 1) % max(1, len(tasks) // 10) == 0 or i == len(tasks) - 1:
                pct = (i + 1) / len(tasks) * 100
                print(f"  Chunks: {i+1}/{len(tasks)} ({pct:.0f}%)")
    tokenize_time = time.time() - t0

    # Merge word counts from all chunks
    t0 = time.time()
    words_counter = Counter()
    for chunk_count in chunk_word_counts:
        words_counter.update(chunk_count)
    merge_time = time.time() - t0

    print(f"  Chunking: {chunk_time*1000:.1f}ms | Tokenize: {tokenize_time*1000:.1f}ms | Merge: {merge_time*1000:.1f}ms")

    # Convert words to tuples of single-byte tokens for BPE processing
    word_freq = {}
    for word, freq in words_counter.items():
        word_bytes = tuple(bytes([b]) for b in word.encode('utf-8'))
        if word_bytes:
            word_freq[word_bytes] = freq

    # Build initial pair frequency and pair-to-words index (pre-indexing)
    pair_freq = defaultdict(int)
    pair_to_words = defaultdict(set)

    for word, freq in word_freq.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freq[pair] += freq
            pair_to_words[pair].add(word)

    # Build max-heap (use negative freq and ReversedPair for correct tie-breaking)
    heap = [(-freq, ReversedPair(pair)) for pair, freq in pair_freq.items()]
    heapq.heapify(heap)

    # BPE merge loop
    merges = []
    num_merges = vocab_size - len(vocab)
    iter_times = []

    for merge_idx in range(num_merges):
        iter_start = time.time()

        # Find the most frequent pair using heap with lazy deletion
        while heap:
            neg_freq, reversed_pair = heapq.heappop(heap)
            best_pair = reversed_pair.pair
            actual_freq = pair_freq.get(best_pair, 0)
            # Check if this heap entry is still valid (not stale)
            if actual_freq == -neg_freq and actual_freq > 0:
                break
            # Stale entry - frequency changed, skip it
        else:
            break  # Heap exhausted, no more pairs

        # Record the merge operation
        merges.append(best_pair)

        # Create new token by concatenating the pair
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        # Get all words affected by this merge (only these need updating)
        affected_words = list(pair_to_words.pop(best_pair, set()))
        del pair_freq[best_pair]

        # Track pairs that need heap updates
        pairs_to_update = set()

        # Process only affected words (key optimization!)
        for old_word in affected_words:
            if old_word not in word_freq:
                continue

            freq = word_freq[old_word]

            # Remove old pair contributions from this word
            for i in range(len(old_word) - 1):
                pair = (old_word[i], old_word[i + 1])
                if pair in pair_freq:
                    pair_freq[pair] -= freq
                    if pair_freq[pair] <= 0:
                        del pair_freq[pair]
                    else:
                        pairs_to_update.add(pair)
                if pair in pair_to_words:
                    pair_to_words[pair].discard(old_word)

            # Build new word with merge applied
            new_word = []
            i = 0
            while i < len(old_word):
                if (i < len(old_word) - 1 and
                    old_word[i] == best_pair[0] and
                    old_word[i + 1] == best_pair[1]):
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(old_word[i])
                    i += 1

            new_word = tuple(new_word)

            # Update word_freq
            del word_freq[old_word]
            word_freq[new_word] = word_freq.get(new_word, 0) + freq

            # Add new pair contributions
            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                pair_freq[pair] = pair_freq.get(pair, 0) + freq
                pair_to_words[pair].add(new_word)
                pairs_to_update.add(pair)

        # Push updated pairs to heap
        for pair in pairs_to_update:
            if pair in pair_freq and pair_freq[pair] > 0:
                heapq.heappush(heap, (-pair_freq[pair], ReversedPair(pair)))

        # Track iteration time for progress reporting
        iter_time = time.time() - iter_start
        iter_times.append(iter_time)

        # Print progress every 100 merges
        if (merge_idx + 1) % 100 == 0 or merge_idx == 0:
            avg = sum(iter_times) / len(iter_times)
            eta = avg * (num_merges - merge_idx - 1)
            pct = ((merge_idx + 1) / num_merges) * 100

            print(f"Merge {merge_idx+1:4d}/{num_merges} ({pct:5.1f}%) | "
                  f"Time: {iter_time*1000:5.2f}ms | Avg: {avg*1000:5.2f}ms | "
                  f"ETA: {int(eta//60):2d}m {int(eta%60):2d}s")

    total_time = time.time() - total_start
    print(f"\nTotal time: {int(total_time//60):2d}m {total_time%60:.2f}s")

    return vocab, merges