import pytest
import torch

from gpt import GPT, GPTConfig


def test_gpt_forward_returns_logits_and_loss() -> None:
    config = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    model = GPT(config)
    input_ids = torch.randint(0, config.vocab_size, (3, config.block_size))
    labels = torch.randint(0, config.vocab_size, (3, config.block_size))

    output = model(input_ids=input_ids, labels=labels)

    assert output.logits.shape == (3, config.block_size, config.vocab_size)
    assert output.loss is not None
    assert output.loss.ndim == 0


def test_attention_is_causal() -> None:
    config = GPTConfig(
        vocab_size=16,
        block_size=4,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
    )
    model = GPT(config)
    model.eval()

    prefix = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 9, 9]])

    with torch.no_grad():
        prefix_logits = model(prefix).logits
        changed_logits = model(changed_future).logits

    assert torch.allclose(prefix_logits[:, :2], changed_logits[:, :2], atol=1e-6)


def test_forward_with_cache_returns_one_entry_per_layer() -> None:
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
    input_ids = torch.randint(0, config.vocab_size, (3, 5))

    with torch.no_grad():
        output = model(input_ids=input_ids, use_cache=True)

    assert output.loss is None
    assert output.past_key_values is not None
    assert len(output.past_key_values) == config.n_layer
    key, value = output.past_key_values[0]
    assert key.shape == (3, config.n_head, 5, config.n_embd // config.n_head)
    assert value.shape == key.shape


def test_cached_decode_logits_match_full_forward() -> None:
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
    input_ids = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        full_logits = model(input_ids=input_ids).logits
        cached_prefix = model(input_ids=input_ids[:, :3], use_cache=True)
        cached_logits = model(
            input_ids=input_ids[:, 3:4],
            past_key_values=cached_prefix.past_key_values,
            use_cache=True,
            position_offset=3,
        ).logits

    assert torch.allclose(cached_logits[:, -1, :], full_logits[:, 3, :], atol=1e-6)


def test_cached_decode_extends_cache_by_one_token() -> None:
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

    with torch.no_grad():
        prefix_output = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True)
        decode_output = model(
            input_ids=torch.tensor([[4]]),
            past_key_values=prefix_output.past_key_values,
            use_cache=True,
            position_offset=3,
        )

    assert decode_output.past_key_values is not None
    assert decode_output.past_key_values[0][0].size(-2) == 4


def test_cache_rejects_training_labels() -> None:
    config = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    model = GPT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 4))

    with pytest.raises(ValueError, match="KV cache is only supported for inference"):
        model(input_ids=input_ids, labels=input_ids, use_cache=True)
