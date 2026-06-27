"""
Cross-entropy training loop for TinyStories.
"""

import math
import os
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.transformer import Transformer


@dataclass
class TrainConfig:
    lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_steps: int = 1000
    max_steps: int = 100_000
    batch_size: int = 32
    grad_accum_steps: int = 4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    distill_alpha: float = 1.0

    log_interval: int = 1
    save_interval: int = 2000
    keep_last_n: int = 3
    plot_interval: int = 100
    save_dir: str = "checkpoints"

    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = False


def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def _init_history(resume_history: dict | None) -> dict[str, list]:
    history = resume_history or {}
    for key in ("step", "loss", "lr", "tok_per_sec"):
        history.setdefault(key, [])
    return history


def train(
    model: Transformer,
    train_loader: DataLoader,
    cfg: TrainConfig,
    resume_step: int = 0,
    resume_optimizer: dict | None = None,
    resume_history: dict | None = None,
):
    device = cfg.device
    dtype = getattr(torch, cfg.dtype)
    model = model.to(device)

    if cfg.compile:
        model = torch.compile(model)

    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if p.dim() < 2 or "emb" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=cfg.betas,
    )

    if resume_optimizer is not None:
        optimizer.load_state_dict(resume_optimizer)
        print(f"resumed optimizer state ({len(resume_optimizer.get('state', {}))} param slots)")

    print("training objective: cross entropy")
    os.makedirs(cfg.save_dir, exist_ok=True)

    use_scaler = dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    data_iter = iter(train_loader)
    t0 = time.time()
    history = _init_history(resume_history)

    start_step = resume_step + 1
    for step in range(start_step, cfg.max_steps + 1):
        model.train()
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)

        for _ in range(cfg.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=dtype):
                loss = model(input_ids, labels=labels)["loss"]
                loss = loss / cfg.grad_accum_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            with torch.no_grad():
                loss_sum += loss.detach()

        if use_scaler:
            scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            tok_per_sec = (
                cfg.batch_size * cfg.grad_accum_steps * input_ids.size(1) * cfg.log_interval / dt
            )
            loss_val = loss_sum.item()
            print(
                f"step {step:>6d} | loss {loss_val:.4f} | "
                f"ppl {math.exp(min(loss_val, 20)):.2f} | "
                f"lr {lr:.2e} | tok/s {tok_per_sec:,.0f} | dt {dt:.1f}s"
            )
            history["step"].append(step)
            history["loss"].append(loss_val)
            history["lr"].append(lr)
            history["tok_per_sec"].append(tok_per_sec)
            t0 = time.time()

        if step % cfg.plot_interval == 0:
            _save_plot(history, cfg.save_dir)

        if step % cfg.save_interval == 0:
            _save_checkpoint(model, optimizer, history, cfg, step)

    torch.save(
        {
            "step": cfg.max_steps,
            "model": model.state_dict(),
            "config": model.cfg,
        },
        os.path.join(cfg.save_dir, "final.pt"),
    )


def _save_plot(history: dict[str, list], save_dir: str) -> None:
    steps = history["step"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(steps, history["loss"], linewidth=0.8)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("step")

    axes[1].plot(steps, history["lr"], linewidth=0.8, color="orange")
    axes[1].set_title("Learning Rate")
    axes[1].set_xlabel("step")

    axes[2].plot(steps, history["tok_per_sec"], linewidth=0.8, color="green")
    axes[2].set_title("Tokens/sec")
    axes[2].set_xlabel("step")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def _save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    history: dict[str, list],
    cfg: TrainConfig,
    step: int,
) -> None:
    ckpt_path = os.path.join(cfg.save_dir, f"step_{step:07d}.pt")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": model.cfg,
            "history": history,
        },
        ckpt_path,
    )
    print(f"saved checkpoint: {ckpt_path}")

    if cfg.keep_last_n > 0:
        import glob

        ckpts = sorted(glob.glob(os.path.join(cfg.save_dir, "step_*.pt")))
        for old in ckpts[:-cfg.keep_last_n]:
            os.remove(old)
            print(f"pruned old checkpoint: {old}")
