"""
Entry point for TinyStories training with the ternary transformer.

Usage:
    python train.py
    python train.py --resume checkpoints/step_0010000.pt
"""

import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader

from src.data.dataset import TextDataset, build_tokenizer, collate_fn
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.training.trainer import TrainConfig, train


def main():
    parser = argparse.ArgumentParser(description="Train ternary LM on TinyStories")

    # Data
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text-key", type=str, default="text")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-Small-Instruct-2409")
    parser.add_argument("--data-cache-dir", type=str, default="data/cache")

    # Model
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=2304)
    parser.add_argument("--max-seq-len", type=int, default=1024)

    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=1000)

    # Infra
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    print(f"loading tokenizer: {args.tokenizer}")
    tokenizer = build_tokenizer(args.tokenizer)

    model_cfg = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        vocab_size=len(tokenizer),
        max_seq_len=args.max_seq_len,
    )
    print(f"model: {model_cfg.param_count / 1e6:.1f}M params")
    print(f"ternary size: {model_cfg.ternary_size_mb:.1f} MB")

    model = Transformer(model_cfg)

    resume_step = 0
    resume_optimizer = None
    resume_history = None
    if args.resume:
        print(f"resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state_dict)
        resume_step = ckpt.get("step", 0)
        resume_optimizer = ckpt.get("optimizer")
        resume_history = ckpt.get("history")
        print(
            f"resuming from step {resume_step} "
            f"(optimizer={'yes' if resume_optimizer else 'no'}, "
            f"history={'yes' if resume_history else 'no'})"
        )

    print(f"loading dataset: {args.dataset}")
    dataset = TextDataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        split=args.split,
        subset=args.subset,
        text_key=args.text_key,
        max_samples=args.max_samples,
        cache_dir=args.data_cache_dir,
    )
    print(f"dataset: {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_id=tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    train_cfg = TrainConfig(
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        distill_alpha=1.0,
        device=args.device,
        dtype=args.dtype,
        compile=args.compile,
        save_dir=args.save_dir,
    )

    train(
        model,
        loader,
        train_cfg,
        resume_step=resume_step,
        resume_optimizer=resume_optimizer,
        resume_history=resume_history,
    )


if __name__ == "__main__":
    main()
