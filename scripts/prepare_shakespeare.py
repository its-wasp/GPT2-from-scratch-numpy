"""Download TinyShakespeare, BPE-tokenize with GPT-2 vocab, write train.bin / val.bin.

Adapted from karpathy/nanoGPT::data/shakespeare/prepare.py.

"""

import os
import urllib.request

import numpy as np
import tiktoken

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "shakespeare")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_path = os.path.join(DATA_DIR, "input.txt")

    if not os.path.exists(raw_path):
        print(f"downloading TinyShakespeare from {URL}")
        text = urllib.request.urlopen(URL).read().decode("utf-8")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        with open(raw_path, encoding="utf-8") as f:
            text = f.read()

    n = len(text)
    train_text = text[: int(n * 0.9)]
    val_text = text[int(n * 0.9) :]

    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_text)
    val_ids = enc.encode_ordinary(val_text)
    print(f"train: {len(train_ids):,} tokens   val: {len(val_ids):,} tokens")

    np.array(train_ids, dtype=np.uint16).tofile(os.path.join(DATA_DIR, "train.bin"))
    np.array(val_ids, dtype=np.uint16).tofile(os.path.join(DATA_DIR, "val.bin"))
    print(f"wrote {DATA_DIR}/train.bin and val.bin")


if __name__ == "__main__":
    main()
