"""Stream OpenWebText, BPE-tokenize with GPT-2 vocab, write train.bin / val.bin.

Adapted from karpathy/nanoGPT::data/openwebtext/prepare.py. Simplified: uses
streaming + a target token count instead of multiprocess `dataset.map(...)`.

Defaults to ~500M tokens (right scale for an 11M-param model).

"""

import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "openwebtext")
TARGET_TOKENS = 500_000_000  # ~500M tokens; tune for the actual training run
VAL_FRACTION = 0.005  # ~0.5% held out for validation


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    dataset = load_dataset("openwebtext", split="train", streaming=True)

    all_tokens = []
    pbar = tqdm(total=TARGET_TOKENS, unit="tok", unit_scale=True, desc="tokenizing OWT")
    for example in dataset:
        ids = enc.encode_ordinary(example["text"])
        ids.append(eot)  # document separator
        all_tokens.extend(ids)
        pbar.update(len(ids))
        if len(all_tokens) >= TARGET_TOKENS:
            break
    pbar.close()

    n = len(all_tokens)
    val_size = int(n * VAL_FRACTION)
    val_tokens = all_tokens[:val_size]
    train_tokens = all_tokens[val_size:]

    print(f"train: {len(train_tokens):,} tokens   val: {len(val_tokens):,} tokens")
    np.array(train_tokens, dtype=np.uint16).tofile(os.path.join(DATA_DIR, "train.bin"))
    np.array(val_tokens, dtype=np.uint16).tofile(os.path.join(DATA_DIR, "val.bin"))
    print(f"wrote {DATA_DIR}/train.bin and val.bin")


if __name__ == "__main__":
    main()
