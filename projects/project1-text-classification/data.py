import torch
from torch.utils.data import Dataset
    

class YourDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        """
        Args:
            - data (pd.DataFrame): the dataset
            - tokenizer (Tokenizer): the tokenizer
            - max_len (int): the maximum length of the sequence
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO implement the __getitem__ function

        raise NotImplementedError("__getitem__ method not implemented")
