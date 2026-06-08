"""Memmap batch loader for tokenized .bin files.

Loader pattern adapted from karpathy/nanoGPT's train.py
"""

import os
import numpy as np


def get_batch(split, block_size, batch_size, data_dir, rng):
    """Sample `batch_size` random offsets from <data_dir>/<split>.bin.

    Returns (idx, targets), both shape (batch_size, block_size), dtype int64.
    `targets` is `idx` shifted forward by one token (next-token prediction).
    """
    path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = rng.integers(0, len(data) - block_size, size=(batch_size,))
    idx = np.stack([np.asarray(data[i : i + block_size], dtype=np.int64) for i in ix])
    targets = np.stack([np.asarray(data[i + 1 : i + block_size + 1], dtype=np.int64) for i in ix])
    return idx, targets