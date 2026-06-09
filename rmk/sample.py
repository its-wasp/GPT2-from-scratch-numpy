"""Autoregressive sampling: temperature + optional top-k filtering."""

import numpy as np

from rmk.backend import xp


def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """Append `max_new_tokens` autoregressively-sampled tokens to `idx`.

    idx: int ndarray of shape (B, T) starting context.
    Returns ndarray of shape (B, T + max_new_tokens).
    """
    model.eval()
    block_size = model.config.block_size

    for _ in range(max_new_tokens):
        # crop to the model's context window
        idx_cond = idx if idx.shape[1] <= block_size else idx[:, -block_size:]
        logits_t, _ = model(xp.asarray(idx_cond))
        logits = logits_t.data[:, -1, :]  # (B, V)  only the last position matters (because next token)

        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None:
            # mask everything below the top-k threshold to -1e9 before softmax,
            # so the kept probabilities sum to 1 after normalization
            kth = xp.sort(logits, axis=-1)[:, -top_k : -top_k + 1]
            logits = xp.where(logits < kth, -1e9, logits)

        # numerically stable softmax
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probs = xp.exp(shifted) / xp.exp(shifted).sum(axis=-1, keepdims=True)
        probs_np = np.asarray(probs)  # numpy.random.choice expects numpy

        next_tokens = np.array(
            [np.random.choice(probs_np.shape[-1], p=probs_np[b]) for b in range(probs_np.shape[0])],
            dtype=np.int64,
        ).reshape(-1, 1)
        idx = np.concatenate([idx, next_tokens], axis=1)

    model.train()
    return idx
