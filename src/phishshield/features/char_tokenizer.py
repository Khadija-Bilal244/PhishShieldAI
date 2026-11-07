"""Character-level URL tokenizer (numpy only).

Used by (1) the distilled CharCNN student, which is what the API serves through ONNX,
and (2) the `debug_tiny` DistilBERT encoder that runs offline without downloading a
pretrained vocabulary.
"""

from __future__ import annotations

import string

import numpy as np

PAD, UNK = 0, 1
_CHARS = string.ascii_lowercase + string.digits + string.punctuation
CHAR2ID = {c: i + 2 for i, c in enumerate(_CHARS)}
VOCAB_SIZE = len(CHAR2ID) + 2


def encode(url: str, max_len: int) -> np.ndarray:
    ids = [CHAR2ID.get(c, UNK) for c in str(url).lower()[:max_len]]
    out = np.zeros(max_len, dtype=np.int64)
    out[: len(ids)] = ids
    return out


def encode_batch(urls, max_len: int) -> np.ndarray:
    return np.stack([encode(u, max_len) for u in urls]) if len(urls) else np.zeros((0, max_len), np.int64)
