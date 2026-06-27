"""
Load a checkpoint and generate text autoregressively.

Usage:
    python generate.py --ckpt checkpoints/step_0028000.pt --prompt "Hello!"
    python generate.py --ckpt checkpoints/step_0028000.pt  # interactive REPL
"""

import argparse
import torch

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import build_tokenizer


@torch.no_grad()
def generate(
    model: Transformer,
    tokenizer,
    prompt_ids: torch.Tensor,            # (1, S)
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cuda",
    eos_id: int | None = None,
) -> torch.Tensor:
    model.eval()
    ids = prompt_ids.to(device)
    max_seq_len = model.cfg.max_seq_len

    for _ in range(max_new_tokens):
        # Truncate context to fit RoPE buffer
        ctx = ids[:, -max_seq_len:]
        logits = model(ctx)["logits"][:, -1, :]  # (1, V)

        if temperature <= 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                mask = cum_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits[mask] = -float("inf")
                logits = torch.full_like(logits, -float("inf")).scatter(
                    1, sorted_idx, sorted_logits
                )
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        ids = torch.cat([ids, next_id], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break

    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default="mistralai/Mistral-Small-Instruct-2409")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--raw", action="store_true",
                   help="Skip chat template; feed --prompt as raw text (completion mode)")
    args = p.parse_args()

    print(f"loading tokenizer: {args.tokenizer}")
    tokenizer = build_tokenizer(args.tokenizer)

    print(f"loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    cfg: ModelConfig = ckpt["config"]
    model = Transformer(cfg)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    step = ckpt.get("step", "?")
    print(f"model: {cfg.d_model}d, {cfg.n_layers}L, step {step}")

    def run(user_prompt: str):
        if args.raw:
            text = user_prompt
        else:
            messages = [{"role": "user", "content": user_prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        input_ids = tokenizer(text, return_tensors="pt").input_ids

        out_ids = generate(
            model, tokenizer, input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
            eos_id=tokenizer.eos_token_id,
        )
        new_ids = out_ids[0, input_ids.size(1):]
        completion = tokenizer.decode(new_ids, skip_special_tokens=True)
        print(completion)
        print("-" * 60)

    if args.prompt is not None:
        run(args.prompt)
    else:
        print("interactive mode — ctrl+C to quit")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                continue
            run(q)


if __name__ == "__main__":
    main()
