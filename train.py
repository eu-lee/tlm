"""
Entry point for training the 10k ternary transformer.

Usage:
    # Train on TinyStories (default)
    python train.py

    # Train on chat data
    python train.py --mode chat

    # Train on chat data with existing local caches
    python train.py --mode chat --data-cache-dir data_chat/cache --logit-cache-dir data_chat/logit_cache

    # Resume from checkpoint
    python train.py --resume checkpoints/step_0010000.pt

    # Precompute teacher logits (one-time, then train without teacher)
    python train.py --precompute-logits --precompute-batch-size 8
    python train.py  # auto-detects cache → skips teacher forward (~58% faster)
"""

import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import ChatDataset, TextDataset, build_tokenizer, collate_fn
from src.data.logit_cache import LogitCache
from src.training.trainer import TrainConfig, load_teacher, train


def main():
    parser = argparse.ArgumentParser(description="Train 10k ternary LM")

    # Data
    parser.add_argument("--mode", type=str, default="text", choices=["text", "chat"],
                        help="Dataset mode: 'text' for plain LM (TinyStories), 'chat' for conversations")
    parser.add_argument("--dataset", type=str, default=None,
                        help="HF dataset name (defaults: text=roneneldan/TinyStories, chat=HuggingFaceH4/ultrachat_200k)")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default=None,
                        help="Dataset split (defaults: text=train, chat=train_sft)")
    parser.add_argument("--text-key", type=str, default="text")
    parser.add_argument("--conv-key", type=str, default="messages")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-Small-Instruct-2409")
    parser.add_argument("--data-cache-dir", type=str, default="data/cache",
                        help="Directory for tokenized dataset caches")

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

    # Distillation
    parser.add_argument("--teacher", type=str, default="mistralai/Mistral-Small-Instruct-2409")
    parser.add_argument("--distill-alpha", type=float, default=None,
                        help="CE vs KD weight (1.0=pure CE, no teacher). Defaults: text=1.0, chat=0.3")
    parser.add_argument("--distill-temp", type=float, default=3.0)

    # Logit caching
    parser.add_argument("--precompute-logits", action="store_true",
                        help="Precompute teacher logits to cache, then exit")
    parser.add_argument("--logit-cache-dir", type=str, default="data/logit_cache")
    parser.add_argument("--logit-cache-k", type=int, default=64,
                        help="Number of top-k teacher logits to cache per position")
    parser.add_argument("--precompute-batch-size", type=int, default=8)
    parser.add_argument("--teacher-quantization", type=str, default=None,
                        choices=["int8", "nf4"],
                        help="Quantize teacher for precomputation (nf4 recommended for speed)")

    # Infra
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    # Tokenizer
    print(f"loading tokenizer: {args.tokenizer}")
    tokenizer = build_tokenizer(args.tokenizer)

    # Model
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
        # torch.compile adds "_orig_mod." prefix to state dict keys
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state_dict)
        resume_step = ckpt.get("step", 0)
        resume_optimizer = ckpt.get("optimizer")
        resume_history = ckpt.get("history")
        print(f"resuming from step {resume_step} "
              f"(optimizer={'yes' if resume_optimizer else 'no'}, "
              f"history={'yes' if resume_history else 'no'})")

    # Dataset — apply mode-specific defaults
    DEFAULTS = {
        "text": {"dataset": "roneneldan/TinyStories", "split": "train", "distill_alpha": 1.0},
        "chat": {"dataset": "HuggingFaceH4/ultrachat_200k", "split": "train_sft", "distill_alpha": 0.3},
    }
    if args.dataset is None:
        args.dataset = DEFAULTS[args.mode]["dataset"]
    if args.split is None:
        args.split = DEFAULTS[args.mode]["split"]
    if args.distill_alpha is None:
        args.distill_alpha = DEFAULTS[args.mode]["distill_alpha"]

    print(f"loading dataset: {args.dataset} (mode={args.mode})")
    if args.mode == "text":
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
    else:
        dataset = ChatDataset(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            split=args.split,
            subset=args.subset,
            conversation_key=args.conv_key,
            max_samples=args.max_samples,
            cache_dir=args.data_cache_dir,
        )
    print(f"dataset: {len(dataset)} samples")

    # Precompute teacher logits to disk and exit (no training).
    if args.precompute_logits:
        dtype = getattr(torch, args.dtype)
        teacher = load_teacher(
            args.teacher, args.device, vocab_size=len(tokenizer),
            quantization=args.teacher_quantization,
        )
        LogitCache.precompute(
            teacher=teacher,
            dataset=dataset,
            cache_dir=args.logit_cache_dir,
            pad_id=tokenizer.pad_token_id,
            top_k=args.logit_cache_k,
            batch_size=args.precompute_batch_size,
            device=args.device,
            dtype=dtype,
            teacher_model=args.teacher,
        )
        return

    # num_workers=0: ChatDataset preloads everything into RAM in __init__,
    # so worker processes add spawn overhead without any win — and on Windows
    # that overhead has been observed to stall training for minutes per step.
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_id=tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # Use logit cache if it exists on disk
    cache_dir = args.logit_cache_dir
    logit_cache_dir = cache_dir if LogitCache(cache_dir).exists() else None
    if logit_cache_dir:
        print(f"logit cache found at {logit_cache_dir} — teacher forward will be skipped")

    # Train
    train_cfg = TrainConfig(
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        teacher_model=args.teacher,
        distill_alpha=args.distill_alpha,
        distill_temp=args.distill_temp,
        logit_cache_dir=logit_cache_dir,
        logit_cache_k=args.logit_cache_k,
        device=args.device,
        dtype=args.dtype,
        compile=args.compile,
        save_dir=args.save_dir,
    )

    train(
        model, loader, train_cfg,
        resume_step=resume_step,
        resume_optimizer=resume_optimizer,
        resume_history=resume_history,
    )


if __name__ == "__main__":
    main()
