from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from common import load_json
from gpt import GPT, GPTConfig


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = float("-inf"),
) -> torch.Tensor:
    """Filter logits with top-k and nucleus sampling."""

    filtered = logits.clone()
    if top_k > 0:
        top_k = min(top_k, filtered.size(-1))
        threshold = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, filter_value)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )
        filtered = filtered.masked_fill(indices_to_remove, filter_value)

    return filtered


@torch.no_grad()
def generate_tokens(
    model: GPT,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    model.eval()
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    generated = input_ids
    for _ in range(max_new_tokens):
        context = generated[:, -model.config.block_size :]
        logits = model(context).logits[:, -1, :] / temperature
        filtered = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(filtered, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
        if eos_token_id is not None and torch.all(next_token.eq(eos_token_id)):
            break
    return generated


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> GPT:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = GPTConfig(**checkpoint["model_config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained GPT checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--tokenizer", default="roneneldan/TinyStories-33M")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=args.local_files_only
    )
    model = load_checkpoint(args.checkpoint, device)

    expected_config = load_json(args.model_config)
    actual_config = model.config.to_dict()
    mismatched = {
        key: (expected_value, actual_config.get(key))
        for key, expected_value in expected_config.items()
        if actual_config.get(key) != expected_value
    }
    if mismatched:
        raise ValueError(f"Checkpoint does not match --model-config: {mismatched}")

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    output_ids = generate_tokens(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
