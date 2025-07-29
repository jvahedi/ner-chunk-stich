# test_core.py

import pandas as pd
from ner_chunk_stitch_pipeline.core import preprocess_break, postprocess_stitch
from ner_chunk_stitch_pipeline.utils import count_words

def test_preprocess_break():
    corpus = pd.Series([
        " ".join(["alpha"] * 120),  # needs to be split
        "short text"               # stays whole
    ])
    chunks, code = preprocess_break(corpus, threshold=50, overlap_min=20)
    assert isinstance(chunks, list)
    assert isinstance(code, list)
    assert sum(code) == len(chunks)
    assert len(code) == len(corpus)

def test_postprocess_stitch_roundtrip():
    # Simulate model output
    predict_broken = []
    for i in range(2):
        # Two "chunks" for a document
        chunk1 = [{"token": f"w{i}_{j}", "label": "O"} for j in range(60)]
        chunk2 = [{"token": f"w{i}_{j}", "label": "O"} for j in range(30, 90)]
        predict_broken.extend([chunk1, chunk2])

    code_alt = [2]  # One document with 2 chunks
    stitched = postprocess_stitch(predict_broken, code_alt, threshold=60, overlap_min=30)

    assert isinstance(stitched, list)
    assert isinstance(stitched[0], list)
    assert len(stitched) == 1
    assert stitched[0][0]["token"].startswith("w0_")

if __name__ == "__main__":
    test_preprocess_break()
    test_postprocess_stitch_roundtrip()
    print("All core tests passed.")
