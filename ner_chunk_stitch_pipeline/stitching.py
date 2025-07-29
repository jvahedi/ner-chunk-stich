# stitching.py

from typing import List, Dict
from .constants import DEFAULT_THRESHOLD, DEFAULT_OVERLAP

def dominant_express(
    overlap1: List[Dict[str, str]],
    overlap2: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Deduplicate overlapping NER predictions by prioritizing non-'O' tags.

    Inputs:
        overlap1: List of token-tag dictionaries from earlier chunk
        overlap2: List from later chunk (same span)

    Returns:
        List of unified dictionaries, choosing dominant tag for overlap.
    """
    overlap = overlap1  # default to left side (earlier chunk)
    for k in range(len(overlap1)):
        if overlap1[k] != overlap2[k]:
            val1 = list(overlap1[k].values())[0]
            val2 = list(overlap2[k].values())[0]
            if val1 == 'O':
                # Prefer non-'O' if other side provides one
                overlap[k] = overlap2[k]
            else:
                # If both have entities, or left is dominant, keep as-is
                pass
    return overlap

def stitch(
    broken: List[List[Dict[str, str]]],
    threshold: float = DEFAULT_THRESHOLD,
    overlap_min: int = DEFAULT_OVERLAP
) -> List[Dict[str, str]]:
    """
    Unify tagged pieces of original text into one document.

    Inputs:
        broken: List of tagged chunks (each chunk is a list of dicts)
        threshold: Max chunk length in words
        overlap_min: Number of words duplicated at chunk boundaries

    Returns:
        Single stitched list of token-tag dictionaries
    """
    m, o = int(threshold), int(overlap_min)
    l = len(broken)

    if l == 1:
        return broken[0]

    out = []  # final stitched output

    # Initial section from first chunk (excluding overlap)
    start = broken[0][:m - o]
    out += start

    # Stitch mid overlaps and middle pieces
    for j in range(l - 1):
        overlap_A = broken[j][m - o:]       # trailing overlap from previous
        overlap_B = broken[j + 1][:o]       # leading overlap from next
        mid_B     = broken[j + 1][o:m - o]  # middle section of next chunk

        overlap = dominant_express(overlap_A, overlap_B)
        out += overlap + mid_B

    # Final section from last chunk
    end = broken[-1][m - o:]
    out += end

    return out
