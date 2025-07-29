# test_chunking.py

from ner_chunk_stitch_pipeline.chunking import (
    needBreak,
    numBatches,
    breakIndices,
    break_chunks,
    coding_alt,
    coding
)
from ner_chunk_stitch_pipeline.utils import count_words
import pandas as pd
import numpy as np

def test_needBreak():
    assert needBreak(50, threshold=100) is False
    assert needBreak(150, threshold=100) is True

def test_numBatches():
    assert numBatches(150, threshold=100, overlap_min=40) == 2
    assert numBatches(300, threshold=112, overlap_min=40) >= 2

def test_breakIndices():
    # Should return indices like [[0,100],[60,160],[120,150]]
    out = breakIndices(150, threshold=100, overlap_min=40)
    assert out.shape[1] == 2
    assert out[0][0] == 0
    assert out[-1][1] == 150

def test_break_chunks():
    text = " ".join([f"word{i}" for i in range(150)])
    chunks = break_chunks(text, threshold=100, overlap_min=40)
    assert len(chunks) >= 2
    assert all(isinstance(c, str) for c in chunks)

def test_coding_alt_and_coding():
    corpus = pd.Series([
        "This is a test document with a small number of words.",
        "This one is longer " + " ".join(["word"] * 200)
    ])
    alt = coding_alt(corpus, threshold=50, overlap_min=20)
    expanded = coding(alt)
    assert sum(alt) == len(expanded)
    assert isinstance(alt, list)
    assert all(isinstance(i, int) for i in alt)

if __name__ == "__main__":
    test_needBreak()
    test_numBatches()
    test_breakIndices()
    test_break_chunks()
    test_coding_alt_and_coding()
    print("All chunking tests passed.")
