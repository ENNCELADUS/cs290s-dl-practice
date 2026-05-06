import torch

import generate as generate_module
from generate import generate_tokens, generate_tokens_cached, top_k_top_p_filtering
from gpt import GPT, GPTConfig


def test_top_k_filter_keeps_only_largest_logits() -> None:
    logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])

    filtered = top_k_top_p_filtering(logits, top_k=2, top_p=1.0)

    assert torch.isneginf(filtered[0, 0])
    assert torch.isneginf(filtered[0, 2])
    assert filtered[0, 1].item() == 4.0
    assert filtered[0, 3].item() == 3.0


def test_top_p_filter_keeps_at_least_one_token() -> None:
    logits = torch.tensor([[10.0, 1.0, 0.5]])

    filtered = top_k_top_p_filtering(logits, top_k=0, top_p=0.1)

    assert not torch.isneginf(filtered[0, 0])
    assert torch.isneginf(filtered[0, 1])
    assert torch.isneginf(filtered[0, 2])


def test_cached_generation_matches_uncached_generation_with_same_seed() -> None:
    torch.manual_seed(0)
    config = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    model = GPT(config)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3]])

    torch.manual_seed(123)
    uncached = generate_tokens(
        model=model,
        input_ids=input_ids,
        max_new_tokens=3,
        temperature=1.0,
        top_k=5,
        top_p=0.9,
    )
    torch.manual_seed(123)
    cached = generate_tokens_cached(
        model=model,
        input_ids=input_ids,
        max_new_tokens=3,
        temperature=1.0,
        top_k=5,
        top_p=0.9,
    )

    assert cached.shape == (1, 6)
    assert torch.equal(cached, uncached)


def test_cached_generation_stops_on_eos(monkeypatch) -> None:
    torch.manual_seed(0)
    config = GPTConfig(
        vocab_size=8,
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
    )
    model = GPT(config)
    model.eval()
    input_ids = torch.tensor([[1, 2]])

    def sample_eos(logits, temperature, top_k, top_p):
        return torch.zeros((logits.size(0), 1), dtype=torch.long, device=logits.device)

    monkeypatch.setattr(generate_module, "sample_next_token", sample_eos)

    output = generate_tokens_cached(
        model=model,
        input_ids=input_ids,
        max_new_tokens=5,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        eos_token_id=0,
    )

    assert output.shape == (1, 3)
    assert output[0, -1].item() == 0
