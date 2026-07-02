"""Compute BPE-level perplexity of a checkpoint on WikiText-103 val or OWT held-out val."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set fp32 BEFORE any Tensor is constructed (training was fp32; ckpt expects it).
from rmk import backend

backend.DTYPE = backend.xp.float32

from rmk.eval import compute_perplexity
from rmk.model import GPT, GPTConfig
from rmk.dca import DCAGPT
from rmk.train import load_checkpoint


DATA_PATHS = {
    "wikitext": "data/wikitext-103/val.bin",
    "owt": "data/openwebtext/val.bin",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to .npz checkpoint")
    ap.add_argument("--dataset", choices=list(DATA_PATHS), required=True)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--out", default=None,
                    help="output JSON path (default: <ckpt_dir>/eval_<dataset>.json)")
    args = ap.parse_args()

    # Reconstruct the model from the run's config.json (sibling of the ckpt)
    ckpt_dir = os.path.dirname(args.ckpt)
    cfg_path = os.path.join(ckpt_dir, "config.json")
    with open(cfg_path) as f:
        full_cfg = json.load(f)
    model_cfg = GPTConfig(**full_cfg["model"])
    arch = full_cfg.get("arch", "GPT")
    print(f"model config: {model_cfg}  arch: {arch}")
    model = DCAGPT(model_cfg) if arch == "DCAGPT" else GPT(model_cfg)
    step = load_checkpoint(args.ckpt, model)
    print(f"loaded ckpt step {step}")

    # Load eval tokens (uint16 memmap -> int64 ndarray)
    bin_path = DATA_PATHS[args.dataset]
    tokens = np.asarray(np.memmap(bin_path, dtype=np.uint16, mode="r"), dtype=np.int64)
    print(f"{args.dataset}: {len(tokens):,} tokens")

    # Sliding-window perplexity
    ppl = compute_perplexity(model, tokens, block_size=args.block_size)
    val_loss = float(np.log(ppl))
    print(f"BPE-level perplexity: {ppl:.2f}  (val_loss = {val_loss:.4f})")

    # Persist
    out_path = args.out or os.path.join(ckpt_dir, f"eval_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "step": step,
            "dataset": args.dataset,
            "n_tokens": len(tokens),
            "block_size": args.block_size,
            "stride": args.block_size,
            "val_loss": val_loss,
            "ppl_bpe": ppl,
        }, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
