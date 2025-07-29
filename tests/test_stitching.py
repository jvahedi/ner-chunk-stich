# test_stitching.py

from ner_chunk_stitch_pipeline.stitching import dominant_express, stitch

def test_dominant_express():
    # Test overlap resolution with 'O' and non-'O' labels
    overlap1 = [{"token": "John", "label": "B-PER"}, {"token": "Doe", "label": "B-PER"}]
    overlap2 = [{"token": "John", "label": "O"}, {"token": "Doe", "label": "B-PER"}]

    # overlap1 dominates unless it contains 'O'
    resolved = dominant_express(overlap1, overlap2)

    assert resolved[0]["label"] == "B-PER"  # 'O' from overlap2 should not override 'B-PER'
    assert resolved[1]["label"] == "B-PER"  # same labels, so no conflict
    
    overlap1 = [{"token": "X", "label": "O"}]
    overlap2 = [{"token": "X", "label": "O"}]
    resolved = dominant_express(overlap1, overlap2)
    assert resolved[0]["label"] == "O"

    overlap1 = [{"token": "Paris", "label": "B-LOC"}]
    overlap2 = [{"token": "Paris", "label": "B-ORG"}]
    resolved = dominant_express(overlap1, overlap2)
    assert resolved[0]["label"] == "B-LOC"  # should keep overlap1

def test_stitch_two_chunks():
    # Simulate two overlapping chunks of tagged tokens
    chunk1 = [{"token": f"word{i}", "label": "O"} for i in range(72)]
    chunk2 = [{"token": f"word{i}", "label": "O"} for i in range(40, 112)]

    stitched = stitch([chunk1, chunk2], threshold=72, overlap_min=32)
    assert len(stitched) == 112  # Should cover the full word0 to word111
    assert stitched[0]["token"] == "word0"
    assert stitched[-1]["token"] == "word111"

def test_stitch_single_chunk():
    # Should return same chunk if only one
    chunk = [{"token": f"w{i}", "label": "O"} for i in range(80)]
    out = stitch([chunk], threshold=80, overlap_min=32)
    assert out == chunk

if __name__ == "__main__":
    test_dominant_express()
    test_stitch_two_chunks()
    test_stitch_single_chunk()
    print("All stitching tests passed.")
