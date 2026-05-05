import torch

from dataset import TokenBlockDataset


def test_token_block_dataset_returns_shifted_blocks() -> None:
    dataset = TokenBlockDataset(token_ids=list(range(11)), block_size=5)

    assert len(dataset) == 2

    first = dataset[0]
    assert torch.equal(first["input_ids"], torch.tensor([0, 1, 2, 3, 4]))
    assert torch.equal(first["labels"], torch.tensor([1, 2, 3, 4, 5]))


def test_token_block_dataset_rejects_too_few_tokens() -> None:
    dataset = TokenBlockDataset(token_ids=[1, 2, 3], block_size=8)

    assert len(dataset) == 0
