# core.py

from typing import List, Tuple, Dict
import pandas as pd
from .chunking import break_chunks, coding_alt
from .stitching import stitch
from .utils import count_words
from .constants import DEFAULT_THRESHOLD, DEFAULT_OVERLAP

def preprocess_break(
    corpus: pd.Series,
    threshold: float = DEFAULT_THRESHOLD,
    overlap_min: int = DEFAULT_OVERLAP
) -> Tuple[List[str], List[int]]:
    """
    Break full texts in corpus into overlapping word chunks.

    Args:
        corpus: Series of full texts.
        threshold: Max words per chunk.
        overlap_min: Word overlap between chunks.

    Returns:
        corpus_broken: List of chunked text strings.
        code_alt: Number of chunks generated per original document.
    """
    threshold, overlap_min = float(threshold), float(overlap_min)
    corpus_broken = []

    for j in range(len(corpus)):
        text = corpus[j]
        text_chunks = break_chunks(text, threshold=threshold, overlap_min=overlap_min)
        corpus_broken += text_chunks

    code_alt = coding_alt(corpus, threshold=threshold, overlap_min=overlap_min)
    return corpus_broken, code_alt

def postprocess_stitch(
    predict_broken: List[List[Dict[str, str]]],
    code_alt: List[int],
    threshold: float = DEFAULT_THRESHOLD,
    overlap_min: int = DEFAULT_OVERLAP
) -> List[List[Dict[str, str]]]:
    """
    Stitch tagged text pieces back together for each original document.

    Args:
        predict_broken: Flat list of predicted tokens (split into chunks).
        code_alt: List of number of chunks per document.
        threshold: Max words per chunk.
        overlap_min: Word overlap between chunks.

    Returns:
        predict: List of stitched predictions for each document.
    """
    threshold = int(threshold)
    overlap_min = int(overlap_min)
    mx = len(code_alt)
    predict = []

    index_s = 0  # start index in flattened predictions
    for i in range(mx):
        index_e = index_s + code_alt[i]  # end index for this doc
        parts_i = predict_broken[index_s:index_e]  # pull out just this document's chunks
        stitched_i = stitch(parts_i, threshold, overlap_min)  # stitch them
        predict.append(stitched_i)
        index_s = index_e  # advance for next doc

    return predict

def entity_extractor(
    model,
    corpus: pd.Series,
    threshold: float = DEFAULT_THRESHOLD,
    overlap_min: int = DEFAULT_OVERLAP,
    verbose: bool = False
) -> List[List[Dict[str, str]]]:
    """
    Full pipeline: chunk, run NER prediction, and stitch output.

    Args:
        model: A SimpleTransformers NER model object.
        corpus: Series of text documents.
        threshold: Max chunk size (in words).
        overlap_min: Overlap between chunks (in words).
        verbose: Whether to show status display.

    Returns:
        List of tagged token dicts for each original document.
    """
    if verbose:
        from IPython.display import clear_output
        print("Running NER on corpus...")
        clear_output(wait=True)

    corpus.reset_index(drop=True, inplace=True)

    # Step 1: Chunk the documents
    corpus_broken, code = preprocess_break(corpus, threshold=threshold, overlap_min=overlap_min)

    # Step 2: Predict tags for each chunk
    predict_broken, _ = model.predict(corpus_broken)

    # Step 3: Stitch predictions back per document
    predict = postprocess_stitch(predict_broken, code, threshold=threshold, overlap_min=overlap_min)

    return predict
