from typing import Tuple, Any
import math
import torch
from torch.utils.data import Dataset

from source.constants import IMAGENET_MEAN, IMAGENET_STD


class GroupEncodingDataset(Dataset):
    """
    A dataset wrapper that encodes the protected attribute in the features

    Attributes:
        dataset (Dataset): The original dataset to be wrapped.
    """

    def __init__(self, dataset: Dataset, patch_size: int = 10) -> None:
        super().__init__()

        self.dataset = dataset
        self.patch_size = patch_size
        self.yield_protected_attribute = True

        # test if protected attribute is given
        assert len(self.dataset[0]) == 3, "Dataset does not provide protected attributes."

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        x, y, a = self.dataset[index]

        if len(x.shape) == 1:
            x = torch.append(x, a.float())
        elif len(x.shape) == 3:
            patch = ((a * 248) - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
            patch = patch.reshape(-1, 1, 1).expand(-1, self.patch_size, self.patch_size)
            x[:, :self.patch_size, :self.patch_size] = patch
            x[:, -self.patch_size:, :self.patch_size] = patch
            x[:, -self.patch_size:, -self.patch_size:] = patch
            x[:, :self.patch_size, -self.patch_size:] = patch

        if self.yield_protected_attribute:
            return x, y, a
        return x, y


class FairnessDataset(Dataset):
    """
    A dataset wrapper that extends the functionality of an existing dataset to provide
    access to a protected attribute for fairness analysis.

    Attributes:
        dataset (Dataset): The original dataset to be wrapped.
        protected_attribute_idx (int): The column index of the protected attribute in the dataset.

    Example:
        Assuming dataset.protected_attributes is a 2D array where each row represents 
        a data point and each column represents a protected attribute:
        
        fairness_data = FairnessDataset(original_dataset, protected_attribute_idx=1)
        x, y, protected_attr = fairness_data[5]
    """

    def __init__(self, dataset: Dataset, protected_attribute_idx: int = 0) -> None:
        super().__init__()

        self.dataset = dataset
        self.protected_attribute_idx = protected_attribute_idx

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        x, y = self.dataset[index]

        return x, y, self.dataset.protected_attributes[index, self.protected_attribute_idx]
