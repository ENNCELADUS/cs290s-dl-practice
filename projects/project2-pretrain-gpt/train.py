from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import cast

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from common import (
    configure_logging,
    count_trainable_parameters,
    ensure_dir,
    format_metrics,
    load_json,
    load_yaml,
    save_json,
    set_seed,
)
from dataset import DatasetBuildConfig, build_token_block_datasets
from generate import generate_tokens
from gpt import GPT, GPTConfig

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a from-scratch GPT on TinyStories.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    return parser.parse_args()


def create_accelerator(exp_config: dict) -> Accelerator:
    backend = exp_config.get("distributed_backend") or os.environ.get("PROJECT2_DDP_BACKEND")
    kwargs_handlers = []
    if backend:
        kwargs_handlers.append(InitProcessGroupKwargs(backend=backend))

    return Accelerator(
        gradient_accumulation_steps=exp_config["gradient_accumulation_steps"],
        log_with="tensorboard",
        project_dir=exp_config["output_dir"],
        kwargs_handlers=kwargs_handlers,
    )


def build_dataloaders(exp_config: dict, tokenizer) -> tuple[DataLoader, DataLoader]:
    data_config = DatasetBuildConfig(
        dataset_name=exp_config["dataset_name"],
        dataset_config_name=exp_config.get("dataset_config_name"),
        block_size=exp_config["block_size"],
        max_train_samples=exp_config.get("max_train_samples"),
        max_validation_samples=exp_config.get("max_validation_samples"),
        num_preprocessing_workers=exp_config.get("num_preprocessing_workers", 1),
        local_files_only=exp_config.get("dataset_local_files_only", False),
    )
    train_dataset, validation_dataset = build_token_block_datasets(data_config, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=exp_config["per_device_train_batch_size"],
        shuffle=True,
        num_workers=exp_config.get("dataloader_num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=exp_config["per_device_eval_batch_size"],
        shuffle=False,
        num_workers=exp_config.get("dataloader_num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, validation_loader


def reset_cuda_peak_memory_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def collect_cuda_memory_metrics(accelerator: Accelerator) -> dict[str, float]:
    if torch.cuda.is_available():
        local_memory = torch.tensor(
            [
                torch.cuda.max_memory_allocated() / (1024**3),
                torch.cuda.max_memory_reserved() / (1024**3),
            ],
            device=accelerator.device,
            dtype=torch.float32,
        )
    else:
        local_memory = torch.zeros(2, device=accelerator.device, dtype=torch.float32)

    gathered = accelerator.gather(local_memory).detach().cpu().view(-1, 2)
    return {
        "cuda_max_memory_allocated_gib": gathered[:, 0].max().item(),
        "cuda_mean_memory_allocated_gib": gathered[:, 0].mean().item(),
        "cuda_max_memory_reserved_gib": gathered[:, 1].max().item(),
        "cuda_mean_memory_reserved_gib": gathered[:, 1].mean().item(),
    }


def unwrap_distributed_model(model: torch.nn.Module) -> GPT:
    """Unwrap DDP/compile wrappers without importing optional distributed plugins."""
    unwrapped = model
    while True:
        child = getattr(unwrapped, "module", None)
        if child is None:
            child = getattr(unwrapped, "_orig_mod", None)
        if not isinstance(child, torch.nn.Module):
            return cast(GPT, unwrapped)
        unwrapped = child


@torch.no_grad()
def run_validation(
    accelerator: Accelerator,
    model: GPT,
    dataloader: DataLoader,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    losses = []
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        with accelerator.autocast():
            outputs = model(**batch)
        if outputs.loss is None:
            raise RuntimeError("Validation loss was not computed.")
        batch_size = batch["input_ids"].shape[0]
        losses.append(accelerator.gather_for_metrics(outputs.loss.detach().repeat(batch_size)))

    if not losses:
        return {"val_loss": float("nan"), "val_perplexity": float("nan")}

    gathered_losses = torch.cat(losses)
    val_loss = gathered_losses.mean().item()
    try:
        val_perplexity = math.exp(val_loss)
    except OverflowError:
        val_perplexity = float("inf")
    return {"val_loss": val_loss, "val_perplexity": val_perplexity}


def save_checkpoint(
    accelerator: Accelerator,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    tokenizer,
    exp_config: dict,
    step: int,
) -> None:
    output_dir = Path(exp_config["output_dir"])
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    unwrapped = unwrap_distributed_model(model)
    checkpoint = {
        "step": step,
        "model_config": unwrapped.config.to_dict(),
        "model_state_dict": unwrapped.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "experiment_config": exp_config,
    }
    accelerator.save(checkpoint, checkpoint_dir / "model.pt")
    if accelerator.is_main_process:
        tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
        save_json(unwrapped.config.to_dict(), checkpoint_dir / "model_config.json")
    accelerator.wait_for_everyone()


def write_sample_text(
    accelerator: Accelerator,
    model: GPT,
    tokenizer,
    exp_config: dict,
    step: int,
) -> None:
    if not accelerator.is_main_process:
        return
    unwrapped = unwrap_distributed_model(model)
    was_training = unwrapped.training
    prompt = exp_config.get("sample_prompt", "Once upon a time")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(accelerator.device)
    samples_dir = ensure_dir(Path(exp_config["output_dir"]) / "samples")
    num_samples = exp_config.get("sample_num_return_sequences", 1)
    for sample_idx in range(num_samples):
        output_ids = generate_tokens(
            model=unwrapped,
            input_ids=input_ids,
            max_new_tokens=exp_config.get("sample_max_new_tokens", 120),
            temperature=exp_config.get("sample_temperature", 0.9),
            top_k=exp_config.get("sample_top_k", 50),
            top_p=exp_config.get("sample_top_p", 0.95),
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        sample_path = samples_dir / f"step_{step}_sample_{sample_idx + 1}.txt"
        sample_path.write_text(text, encoding="utf-8")
        if accelerator.trackers:
            tracker = accelerator.get_tracker("tensorboard", unwrap=True)
            tag = f"sample_text/{sample_idx + 1}"
            if hasattr(tracker, "add_text"):
                tracker.add_text(tag, text, step)
            elif hasattr(tracker, "writer"):
                tracker.writer.add_text(tag, text, step)
    if was_training:
        unwrapped.train()


def main() -> None:
    configure_logging()
    args = parse_args()
    exp_config = load_yaml(args.config)
    model_config_dict = load_json(exp_config["model_config"])
    exp_config["block_size"] = model_config_dict["block_size"]

    ensure_dir(exp_config["output_dir"])
    set_seed(exp_config["seed"])

    accelerator = create_accelerator(exp_config)
    tokenizer = AutoTokenizer.from_pretrained(
        exp_config["tokenizer_name_or_path"],
        local_files_only=exp_config.get("tokenizer_local_files_only", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config_dict["vocab_size"] = len(tokenizer)
    model_config = GPTConfig(**model_config_dict)
    model = GPT(model_config)
    model_parameter_count = count_trainable_parameters(model)

    train_loader, validation_loader = build_dataloaders(exp_config, tokenizer)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=exp_config["learning_rate"],
        betas=tuple(exp_config.get("adam_betas", [0.9, 0.95])),
        weight_decay=exp_config["weight_decay"],
    )
    warmup_steps = int(exp_config["warmup_ratio"] * exp_config["max_train_steps"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=exp_config["max_train_steps"],
    )

    model, optimizer, train_loader, validation_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        validation_loader,
        scheduler,
    )
    reset_cuda_peak_memory_stats()

    accelerator.init_trackers(
        project_name=exp_config.get("tracker_project_name", "project2-pretrain-gpt"),
        config={**exp_config, "model_parameters": model_parameter_count},
    )
    if accelerator.is_main_process:
        LOGGER.info("Model parameters: %s", model_parameter_count)

    progress = tqdm(
        range(exp_config["max_train_steps"]),
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    completed_steps = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start_time = time.perf_counter()

    while completed_steps < exp_config["max_train_steps"]:
        for batch in train_loader:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(**batch)
                if outputs.loss is None:
                    raise RuntimeError("Training loss was not computed.")
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), exp_config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue

            completed_steps += 1
            progress.update(1)

            if completed_steps % exp_config["logging_every_steps"] == 0:
                accelerator.log(
                    {
                        "train_loss": accelerator.reduce(loss.detach(), reduction="mean").item(),
                        "learning_rate": scheduler.get_last_lr()[0],
                    },
                    step=completed_steps,
                )

            if completed_steps % exp_config["eval_every_steps"] == 0:
                metrics = run_validation(
                    accelerator,
                    model,
                    validation_loader,
                    max_batches=exp_config.get("max_eval_batches"),
                )
                accelerator.log(metrics, step=completed_steps)
                if accelerator.is_main_process:
                    LOGGER.info("[step %s] %s", completed_steps, format_metrics(metrics))
                write_sample_text(accelerator, model, tokenizer, exp_config, completed_steps)
                model.train()

            if completed_steps % exp_config["save_every_steps"] == 0:
                save_checkpoint(
                    accelerator,
                    model,
                    optimizer,
                    scheduler,
                    tokenizer,
                    exp_config,
                    completed_steps,
                )

            if completed_steps >= exp_config["max_train_steps"]:
                break

    runtime_seconds = time.perf_counter() - start_time
    training_tokens = (
        completed_steps
        * exp_config["block_size"]
        * exp_config["per_device_train_batch_size"]
        * exp_config["gradient_accumulation_steps"]
        * accelerator.num_processes
    )
    benchmark_metrics = collect_cuda_memory_metrics(accelerator)
    benchmark_metrics.update(
        {
            "train_runtime_seconds": runtime_seconds,
            "train_tokens": float(training_tokens),
            "train_tokens_per_second": training_tokens / runtime_seconds
            if runtime_seconds > 0
            else float("nan"),
        }
    )
    accelerator.log(benchmark_metrics, step=completed_steps)

    final_metrics = run_validation(
        accelerator,
        model,
        validation_loader,
        max_batches=exp_config.get("max_eval_batches"),
    )
    accelerator.log(final_metrics, step=completed_steps)
    write_sample_text(accelerator, model, tokenizer, exp_config, completed_steps)
    if completed_steps % exp_config["save_every_steps"] != 0:
        save_checkpoint(
            accelerator,
            model,
            optimizer,
            scheduler,
            tokenizer,
            exp_config,
            completed_steps,
        )

    if accelerator.is_main_process:
        save_json(
            {**benchmark_metrics, **final_metrics},
            Path(exp_config["output_dir"]) / "final_metrics.json",
        )
        LOGGER.info("[benchmark] %s", format_metrics(benchmark_metrics))
        LOGGER.info("[final] %s", format_metrics(final_metrics))

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
