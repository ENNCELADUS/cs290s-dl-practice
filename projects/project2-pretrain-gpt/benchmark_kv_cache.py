from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from common import configure_logging, ensure_dir, load_json, load_yaml, save_json, set_seed
from generate import generate_tokens, generate_tokens_cached, load_checkpoint

LOGGER = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "Once upon a time",
    "Lily found a tiny red door under the old tree",
    "Tom wanted to help his friend learn how to share",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GPT generation with and without KV cache."
    )
    parser.add_argument("--config", help="Path to a KV cache benchmark YAML config.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--model-config", default="configs/models/medium.json")
    parser.add_argument("--tokenizer", default="roneneldan/TinyStories-33M")
    parser.add_argument("--output-dir", default="outputs/kv_cache_benchmark")
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_benchmark_config(args: argparse.Namespace) -> dict[str, object]:
    if args.config is None:
        if args.checkpoint is None:
            raise ValueError("Either --config or --checkpoint is required.")
        return {
            "checkpoint": args.checkpoint,
            "model_config": args.model_config,
            "tokenizer_name_or_path": args.tokenizer,
            "output_dir": args.output_dir,
            "prompts": args.prompts or DEFAULT_PROMPTS,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "warmup_iters": args.warmup_iters,
            "measure_iters": args.measure_iters,
            "seed": args.seed,
            "tokenizer_local_files_only": args.local_files_only,
        }

    config = load_yaml(args.config)
    checkpoint = config.get("checkpoint") or config.get("checkpoint_path")
    if checkpoint is None:
        raise ValueError("KV benchmark config must set checkpoint or checkpoint_path.")

    return {
        "checkpoint": checkpoint,
        "model_config": config.get("model_config", args.model_config),
        "tokenizer_name_or_path": config.get(
            "tokenizer_name_or_path",
            config.get("tokenizer", args.tokenizer),
        ),
        "output_dir": config.get("benchmark_output_dir", config.get("output_dir", args.output_dir)),
        "prompts": config.get("benchmark_prompts", config.get("prompts", DEFAULT_PROMPTS)),
        "max_new_tokens": config.get(
            "benchmark_max_new_tokens",
            config.get("sample_max_new_tokens", args.max_new_tokens),
        ),
        "temperature": config.get(
            "benchmark_temperature",
            config.get("sample_temperature", args.temperature),
        ),
        "top_k": config.get("benchmark_top_k", config.get("sample_top_k", args.top_k)),
        "top_p": config.get("benchmark_top_p", config.get("sample_top_p", args.top_p)),
        "warmup_iters": config.get("benchmark_warmup_iters", args.warmup_iters),
        "measure_iters": config.get("benchmark_measure_iters", args.measure_iters),
        "seed": config.get("seed", args.seed),
        "tokenizer_local_files_only": config.get(
            "tokenizer_local_files_only",
            args.local_files_only,
        ),
    }


def synchronize_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def run_generation(
    mode: str,
    model,
    input_ids: torch.Tensor,
    config: dict[str, object],
    eos_token_id: int | None,
) -> torch.Tensor:
    generator = generate_tokens_cached if mode == "kv_cache" else generate_tokens
    return generator(
        model=model,
        input_ids=input_ids,
        max_new_tokens=int(config["max_new_tokens"]),
        temperature=float(config["temperature"]),
        top_k=int(config["top_k"]),
        top_p=float(config["top_p"]),
        eos_token_id=eos_token_id,
    )


def benchmark_mode(
    mode: str,
    model,
    tokenizer,
    prompts: list[str],
    config: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    encoded_prompts = [
        tokenizer(prompt, return_tensors="pt").input_ids.to(device) for prompt in prompts
    ]

    for warmup_idx in range(int(config["warmup_iters"])):
        for input_ids in encoded_prompts:
            set_seed(int(config["seed"]) + warmup_idx)
            run_generation(mode, model, input_ids, config, tokenizer.eos_token_id)
    synchronize_if_cuda()

    measured_runs = []
    sample_texts: list[str] = []
    total_seconds = 0.0
    total_new_tokens = 0
    max_peak_memory_gib = 0.0

    for measure_idx in range(int(config["measure_iters"])):
        for prompt_idx, input_ids in enumerate(encoded_prompts):
            set_seed(int(config["seed"]) + measure_idx)
            reset_peak_memory_if_cuda()
            synchronize_if_cuda()
            start_time = time.perf_counter()
            output_ids = run_generation(mode, model, input_ids, config, tokenizer.eos_token_id)
            synchronize_if_cuda()
            elapsed_seconds = time.perf_counter() - start_time
            new_tokens = output_ids.size(1) - input_ids.size(1)
            memory_gib = peak_memory_gib()

            total_seconds += elapsed_seconds
            total_new_tokens += new_tokens
            max_peak_memory_gib = max(max_peak_memory_gib, memory_gib)
            measured_runs.append(
                {
                    "prompt": prompts[prompt_idx],
                    "elapsed_seconds": elapsed_seconds,
                    "new_tokens": new_tokens,
                    "seconds_per_token": elapsed_seconds / new_tokens
                    if new_tokens
                    else float("nan"),
                    "tokens_per_second": new_tokens / elapsed_seconds
                    if elapsed_seconds > 0
                    else float("nan"),
                    "peak_cuda_memory_gib": memory_gib,
                }
            )
            if measure_idx == 0:
                sample_texts.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))

    return {
        "mode": mode,
        "total_seconds": total_seconds,
        "total_new_tokens": total_new_tokens,
        "seconds_per_token": total_seconds / total_new_tokens if total_new_tokens else float("nan"),
        "tokens_per_second": total_new_tokens / total_seconds
        if total_seconds > 0
        else float("nan"),
        "peak_cuda_memory_gib": max_peak_memory_gib,
        "runs": measured_runs,
        "samples": sample_texts,
    }


def build_markdown(results: dict[str, object]) -> str:
    rows = [
        "| Mode | Total time (s) | New tokens | Seconds/token | Tokens/sec | "
        "Peak CUDA memory (GiB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    modes = results["modes"]
    if not isinstance(modes, list):
        raise TypeError("Expected modes to be a list.")
    for mode_result in modes:
        if not isinstance(mode_result, dict):
            raise TypeError("Expected each mode result to be a dict.")
        rows.append(
            "| {mode} | {total_seconds:.3f} | {total_new_tokens} | {seconds_per_token:.5f} | "
            "{tokens_per_second:.2f} | {peak_cuda_memory_gib:.3f} |".format(**mode_result)
        )

    lines = [
        "# KV Cache Benchmark",
        "",
        *rows,
        "",
        "Generation settings: `max_new_tokens={max_new_tokens}`, `temperature={temperature}`, "
        "`top_k={top_k}`, `top_p={top_p}`.".format(**results["generation_settings"]),
        "",
    ]
    for mode_result in modes:
        if not isinstance(mode_result, dict):
            continue
        lines.append(f"## Samples: {mode_result['mode']}")
        samples = mode_result.get("samples", [])
        if isinstance(samples, list):
            for sample_idx, sample in enumerate(samples, start=1):
                lines.append("")
                lines.append(f"### Prompt {sample_idx}")
                lines.append("")
                lines.append(str(sample))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_benchmark_config(args)
    if int(config["measure_iters"]) < 1:
        raise ValueError("--measure-iters must be at least 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        str(config["tokenizer_name_or_path"]),
        local_files_only=bool(config["tokenizer_local_files_only"]),
    )
    model = load_checkpoint(str(config["checkpoint"]), device)

    expected_config = load_json(str(config["model_config"]))
    actual_config = model.config.to_dict()
    mismatched = {
        key: (expected_value, actual_config.get(key))
        for key, expected_value in expected_config.items()
        if actual_config.get(key) != expected_value
    }
    if mismatched:
        raise ValueError(f"Checkpoint does not match --model-config: {mismatched}")

    prompts = [str(prompt) for prompt in config["prompts"]]
    modes = [
        benchmark_mode("no_cache", model, tokenizer, prompts, config, device),
        benchmark_mode("kv_cache", model, tokenizer, prompts, config, device),
    ]
    results = {
        "checkpoint": str(Path(str(config["checkpoint"]))),
        "device": str(device),
        "prompts": prompts,
        "generation_settings": {
            "max_new_tokens": int(config["max_new_tokens"]),
            "temperature": float(config["temperature"]),
            "top_k": int(config["top_k"]),
            "top_p": float(config["top_p"]),
            "warmup_iters": int(config["warmup_iters"]),
            "measure_iters": int(config["measure_iters"]),
            "seed": int(config["seed"]),
        },
        "modes": modes,
    }

    output_dir = ensure_dir(str(config["output_dir"]))
    save_json(results, output_dir / "kv_cache_benchmark.json")
    markdown_path = output_dir / "kv_cache_benchmark.md"
    markdown_path.write_text(build_markdown(results), encoding="utf-8")
    LOGGER.info("Wrote %s and %s", output_dir / "kv_cache_benchmark.json", markdown_path)


if __name__ == "__main__":
    main()
