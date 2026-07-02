"""Train a DeepCrossAttention GPT on the same OWT slice/config as train_owt.py (fp32, GPU)."""

import argparse
import datetime
import json
import os
import sys
from dataclasses import asdict

import numpy as np
import tiktoken
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# set fp32 before any Tensor is built (same memory reason as train_owt.py)
from rmk import backend

backend.DTYPE = backend.xp.float32

from rmk.data import get_batch
from rmk.model import GPTConfig
from rmk.dca import DCAGPT
from rmk.optim import AdamW
from rmk.sample import generate
from rmk.train import TrainConfig, train


def make_batch_fn(split, cfg, rng):
    def fn():
        return get_batch(split, cfg.block_size, cfg.batch_size, cfg.data_dir, rng)
    return fn


def plot_metrics(metrics, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    tr = np.array(metrics["train_loss"])
    va = np.array(metrics["val_loss"])
    ax.plot(tr[:, 0], tr[:, 1], alpha=0.4, label="train")
    if len(va):
        ax.plot(va[:, 0], va[:, 1], marker="o", label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    gn = np.array(metrics["grad_norm"])
    ax.plot(gn[:, 0], gn[:, 1])
    ax.set_xlabel("step")
    ax.set_ylabel("pre-clip global grad norm")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "grad_norm.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    lr = np.array(metrics["lr"])
    ax.plot(lr[:, 0], lr[:, 1])
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lr.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, required=True, help="total training steps")
    ap.add_argument(
        "--name",
        type=str,
        default=None,
        help="run subdirectory under runs/owt/ (defaults to YYYYMMDD-HHMMSS)",
    )
    args = ap.parse_args()

    run_name = args.name or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("runs", "owt", run_name)

    # identical to train_owt.py so the only variable vs baseline is the architecture
    cfg = TrainConfig(
        data_dir="data/openwebtext",
        block_size=256,
        batch_size=4,
        max_lr=6e-4,
        min_lr=6e-5,
        warmup_steps=200,
        max_steps=args.max_steps,
        weight_decay=0.1,
        grad_clip=1.0,
        eval_interval=500,
        eval_iters=20,
        log_interval=10,
        checkpoint_interval=2000,
        out_dir=out_dir,
        seed=0,
    )

    np.random.seed(cfg.seed)
    rng_train = np.random.default_rng(cfg.seed)
    rng_val = np.random.default_rng(cfg.seed + 1)

    model_cfg = GPTConfig(
        block_size=cfg.block_size,
        vocab_size=50257,
        n_layer=6,
        n_head=6,
        n_embd=192,
        dropout=0.0,
    )
    model = DCAGPT(model_cfg)
    n_params = sum(p.data.size for p in model.parameters())
    print(f"run name: {run_name}  ->  {out_dir}/")
    print(f"backend: {backend.name}  dtype: {backend.DTYPE}  model: DCAGPT")
    print(f"model: {n_params:,} params ({n_params/1e6:.2f}M)")

    optimizer = AdamW(
        model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump({"model": asdict(model_cfg), "train": asdict(cfg), "arch": "DCAGPT"}, f, indent=2)

    get_train_batch = make_batch_fn("train", cfg, rng_train)
    get_val_batch = make_batch_fn("val", cfg, rng_val)

    metrics = train(model, optimizer, get_train_batch, get_val_batch, cfg)
    plot_metrics(metrics, cfg.out_dir)
    print(f"saved plots to {cfg.out_dir}/")

    enc = tiktoken.get_encoding("gpt2")
    prompt = "The "
    prompt_ids = np.array([enc.encode_ordinary(prompt)], dtype=np.int64)
    out_ids = generate(model, prompt_ids, max_new_tokens=200, temperature=0.8, top_k=40)
    sample_text = enc.decode(out_ids[0].tolist())
    samples_path = os.path.join(cfg.out_dir, "samples.txt")
    with open(samples_path, "w", encoding="utf-8") as f:
        f.write(sample_text)
    print(f"sample written to {samples_path}")
    print("--- sample (first 500 chars, ascii-safe) ---")
    print(sample_text[:500].encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
