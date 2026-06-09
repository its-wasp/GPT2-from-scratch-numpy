"""Train a small GPT on TinyShakespeare and produce loss curves + sample text."""

import json
import os
import sys

import numpy as np
import tiktoken
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rmk.backend import xp
from rmk.data import get_batch
from rmk.model import GPT, GPTConfig
from rmk.optim import AdamW
from rmk.sample import generate
from rmk.train import TrainConfig, train


def make_batch_fn(split, cfg, rng):
    def fn():
        return get_batch(split, cfg.block_size, cfg.batch_size, cfg.data_dir, rng)
    return fn


def plot_metrics(metrics, out_dir):
    # loss curve: train (every step) + val (every eval_interval)
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

    # grad norm
    fig, ax = plt.subplots(figsize=(8, 4))
    gn = np.array(metrics["grad_norm"])
    ax.plot(gn[:, 0], gn[:, 1])
    ax.set_xlabel("step")
    ax.set_ylabel("pre-clip global grad norm")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "grad_norm.png"), dpi=120)
    plt.close(fig)

    # LR schedule
    fig, ax = plt.subplots(figsize=(8, 4))
    lr = np.array(metrics["lr"])
    ax.plot(lr[:, 0], lr[:, 1])
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lr.png"), dpi=120)
    plt.close(fig)


def main():

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=2000, help="total training steps")
    args = ap.parse_args()
    cfg = TrainConfig(max_steps=args.max_steps)

    np.random.seed(cfg.seed)
    rng_train = np.random.default_rng(cfg.seed)
    rng_val = np.random.default_rng(cfg.seed + 1)

    model_cfg = GPTConfig(
        block_size=cfg.block_size,
        vocab_size=50257,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
    )
    model = GPT(model_cfg)
    n_params = sum(p.data.size for p in model.parameters())
    print(f"model: {n_params:,} params ({n_params/1e6:.2f}M)")

    optimizer = AdamW(
        model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay
    )

    get_train_batch = make_batch_fn("train", cfg, rng_train)
    get_val_batch = make_batch_fn("val", cfg, rng_val)

    metrics = train(model, optimizer, get_train_batch, get_val_batch, cfg)
    plot_metrics(metrics, cfg.out_dir)
    print(f"saved plots to {cfg.out_dir}/")

    # generate a sample continuation from a Shakespearean prompt
    enc = tiktoken.get_encoding("gpt2")
    prompt = "ROMEO: "
    prompt_ids = np.array([enc.encode_ordinary(prompt)], dtype=np.int64)
    out_ids = generate(model, prompt_ids, max_new_tokens=200, temperature=0.8, top_k=40)
    sample_text = enc.decode(out_ids[0].tolist())
    samples_path = os.path.join(cfg.out_dir, "samples.txt")
    with open(samples_path, "w", encoding="utf-8") as f:
        f.write(sample_text)
    print(f"sample written to {samples_path}")
    print("--- sample ---")
    print(sample_text)


if __name__ == "__main__":
    main()
