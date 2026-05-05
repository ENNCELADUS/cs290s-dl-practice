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
