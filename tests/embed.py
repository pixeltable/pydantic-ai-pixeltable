"""Hash embedding used by catalog tests. Not a semantic model."""

from __future__ import annotations

import hashlib

import numpy as np
import pixeltable as pxt

DIM = 8


@pxt.udf
def tiny_embed(text: str) -> pxt.Array[(8,), pxt.Float]:
    digest = hashlib.sha256(text.encode()).digest()
    values = np.array([(digest[i % 32] / 127.5) - 1.0 for i in range(DIM)], dtype=np.float32)
    norm = float(np.linalg.norm(values))
    return values / norm if norm else values
