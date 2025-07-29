# chunking.py

import numpy as np
import pandas as pd
from typing import List
from .utils import count_words
from .constants import DEFAULT_THRESHOLD, DEFAULT_OVERLAP

def needBreak(word_len: int, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """
    Decide whether text should be broken into chunks.

    Args:
        word_len: Number of words in the input text.
        threshold: Maximum chunk size.

    Returns:
        Whether or not to split text.
    """
    goBreak = (word_len > threshold)
    return goBreak

def numBatches(word_len: int, threshold: float = DEFAULT_THRESHOLD, overlap_min: int = DEFAULT_OVERLAP) -> int:
    """
    Determine how many chunks a text should be broken into.

    Args:
        word_len: Total number of words in the text.
        threshold: Chunk size.
        overlap_min: Number of overlapping words between chunks.

    Returns:
        Number of pieces to break the text into.
    """
    L, m, o = word_len, threshold, overlap_min
    n = np.ceil((L - o) / (m - o)).astype(int)  # sliding window formula
    return np.max([n, 1])  # always return at least one chunk

def breakIndices(word_len: int, threshold: float = DEFAULT_THRESHOLD, overlap_min: int = DEFAULT_OVERLAP, verbose: bool = False) -> np.ndarray:
    """
    Determine start and end word indices for each chunk.

    Args:
        word_len: Number of words in the text.
        threshold: Max words per chunk.
        overlap_min: Overlap between chunks.
        verbose: Whether to print debug info.

    Returns:
        Array of [start, end] index pairs for each chunk.
    """
    L, m, o = word_len, threshold, overlap_min
    Need = needBreak(word_len, threshold=threshold)

    if Need:
        n = numBatches(L, m, o)

        indices = [[0, m]]  # first chunk
        for i in range(1, n - 1):
            # next chunk: start = previous end - overlap
            indices.append([indices[-1][1] - o, indices[-1][1] - o + m])
        # final chunk up to the end of text
        indices.append([indices[-1][1] - o, L])
    else:
        indices = [[0, L]]

    if verbose:
        print(f"[breakIndices] Total words: {L}, Chunks: {len(indices)}")

    return np.array(indices).astype(int)

def break_chunks(text: str, threshold: float = DEFAULT_THRESHOLD, overlap_min: int = DEFAULT_OVERLAP) -> List[str]:
    """
    Split a single text into overlapping word chunks.

    Args:
        text: Full input text.
        threshold: Max words per chunk.
        overlap_min: Number of overlapping words.

    Returns:
        List of chunked text segments.
    """
    m, o = threshold, overlap_min
    L = count_words(text)
    indices = breakIndices(L, threshold=threshold, overlap_min=overlap_min)
    words = text.split()
    chunks = []

    for start, end in indices:
        subtext = ' '.join(words[start:end])
        chunks.append(subtext)

    return chunks

def coding_alt(corpus: pd.Series, threshold: float = DEFAULT_THRESHOLD, overlap_min: int = DEFAULT_OVERLAP) -> List[int]:
    """
    Calculate how many chunks each document will be split into.

    Args:
        corpus: A Series of input texts.
        threshold: Max words per chunk.
        overlap_min: Word overlap between chunks.

    Returns:
        A list containing number of chunks per input text.
    """
    num_batches = [
        numBatches(count_words(corpus[i]), threshold=threshold, overlap_min=overlap_min)
        for i in range(len(corpus))
    ]
    return num_batches

def coding(code_alt: List[int]) -> List[int]:
    """
    Create a document index map corresponding to each chunk.

    Args:
        code_alt: List of number of chunks per document.

    Returns:
        A flat list mapping each chunk to its original document index.
    """
    code = []
    for i in range(len(code_alt)):
        num = code_alt[i]
        code += [i] * num
    return code
