"""Training loop, periodic eval, checkpoint save/load."""

import json
import os
from dataclasses import dataclass

import numpy as np

from rmk.backend import xp
from rmk.optim import cosine_with_warmup, clip_grad_norm


@dataclass
class TrainConfig:
    data_dir: str = "data/shakespeare"
    block_size: int = 128
    batch_size: int = 4
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    max_steps: int = 2000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_iters: int = 20
    log_interval: int = 10
    checkpoint_interval: int = 500
    out_dir: str = "runs/shakespeare"
    seed: int = 0


def estimate_loss(model, get_batch_fn, n_iters):
    """Run the model in eval mode for n_iters batches; return mean loss."""
    model.eval()
    losses = []
    for _ in range(n_iters):
        idx, tgt = get_batch_fn()
        _, loss = model(xp.asarray(idx), xp.asarray(tgt))
        losses.append(float(loss.data))
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path, model, step):
    np.savez(path, step=step, **model.state_dict())


def load_checkpoint(path, model):
    data = np.load(path)
    sd = {k: data[k] for k in data.files if k != "step"}
    model.load_state_dict(sd)
    return int(data["step"])


def train(model, optimizer, get_train_batch, get_val_batch, cfg):
    """Run cfg.max_steps of (forward, backward, clip, step). Returns metrics dict."""
    metrics = {"train_loss": [], "val_loss": [], "grad_norm": [], "lr": []}
    os.makedirs(cfg.out_dir, exist_ok=True)

    for step in range(cfg.max_steps):
        lr = cosine_with_warmup(step, cfg.warmup_steps, cfg.max_steps, cfg.max_lr, cfg.min_lr)
        optimizer.lr = lr

        if step % cfg.eval_interval == 0:
            val_loss = estimate_loss(model, get_val_batch, cfg.eval_iters)
            metrics["val_loss"].append((step, val_loss))
            print(f"step {step:5d}  lr={lr:.2e}  val_loss={val_loss:.4f}")

        idx, tgt = get_train_batch()
        optimizer.zero_grad()
        _, loss = model(xp.asarray(idx), xp.asarray(tgt))
        loss.backward()
        gn = clip_grad_norm(model.parameters(), cfg.grad_clip)
        optimizer.step()

        train_loss = float(loss.data)
        metrics["train_loss"].append((step, train_loss))
        metrics["grad_norm"].append((step, gn))
        metrics["lr"].append((step, lr))

        if step % cfg.log_interval == 0:
            print(f"  step {step:5d}  train_loss={train_loss:.4f}  grad_norm={gn:.3f}")

        if step > 0 and step % cfg.checkpoint_interval == 0:
            save_checkpoint(os.path.join(cfg.out_dir, f"ckpt_step{step}.npz"), model, step)

    save_checkpoint(os.path.join(cfg.out_dir, "ckpt_final.npz"), model, cfg.max_steps)
    with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    return metrics
