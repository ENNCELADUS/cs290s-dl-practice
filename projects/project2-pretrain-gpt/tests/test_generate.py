import torch

from generate import top_k_top_p_filtering


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
