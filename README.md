# NER Chunk-Stitch Pipeline

This package implements a robust, modular pipeline for running Named Entity Recognition (NER) on long texts using overlapping word-based chunks. It is built on top of `SimpleTransformers` and designed to work with transformer-based models (e.g., BERT).

---

## 🔧 Features

* Splits long documents into overlapping word chunks
* Applies NER tagging to each chunk
* Stitches predictions back into the original document structure
* Designed to avoid entity fragmentation at chunk boundaries
* Clear function-level modularity for easy customization and reuse

---

## 🚀 Quickstart

```python
from ner_chunk_stitch_pipeline import load_ner_model, entity_extractor
import pandas as pd

# Load model (from SimpleTransformers format directory)
model = load_ner_model("outputs/", use_cuda=True)

# Sample corpus
corpus = pd.Series([
    "John Smith lives in New York City and works for NASA.",
    "The Eiffel Tower is located in Paris, France."
])

# Run extraction
predictions = entity_extractor(model, corpus)

# Each prediction is a list of tagged tokens (dicts)
for doc_preds in predictions:
    print(doc_preds)
```

---

## 📦 Functions Overview

### Core

* `entity_extractor(model, corpus, ...)` — Run full pipeline on a corpus
* `load_ner_model(...)` — Load a pretrained model

### Chunking

* `break_chunks(text, ...)` — Break text into overlapping word chunks
* `breakIndices(...)` — Compute chunk index ranges
* `count_words(text)` — Count whitespace-separated words
* `numBatches(...)` — Compute number of chunks per document

### Stitching

* `stitch(...)` — Merge predictions across chunks
* `dominant_express(...)` — Resolve overlapping tag conflicts

### Metadata

* `coding_alt(...)` — Get chunk count per original document
* `coding(...)` — Expand chunk count into document index map
* `preprocess_break(...)` — Break full corpus and return chunk metadata
* `postprocess_stitch(...)` — Reassemble chunk-wise predictions

---

## 📌 Assumptions

* Tokenization is word-based (via `.split()`), not tokenizer-driven
* Chunk length and overlap are specified in word count
* Model input/output must follow SimpleTransformers NER format

---

## 🔄 Recommended Parameters

* `threshold=112` (default): max words per chunk
* `overlap_min=40` (default): number of overlapping words between chunks

---

## 🧪 Future Improvements

* Add support for tokenizer-aware chunking
* Convert to HuggingFace pipeline compatibility
* Add character-offset stitching

---

## 📂 Directory Suggestions

You can organize the module in a directory structure like:

```
ner_chunk_stitch_pipeline/
├── __init__.py
├── core.py
├── chunking.py
├── stitching.py
├── utils.py
├── model.py
├── README.md
```

---

## 👤 Author Notes

This was originally developed for RAND research and personal use.
