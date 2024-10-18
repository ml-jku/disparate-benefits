import os
import pandas as pd
from PIL import Image
from typing import Any, Callable, Final, Tuple, List

import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms

from ..constants import CHEXPERT_PATH, DATASETS_PATH, IMAGENET_MEAN, IMAGENET_STD

# preprocessing according to https://github.com/MLforHealth/CXR_Fairness/blob/master/cxr_fairness/data/data.py
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, 
                             std=IMAGENET_STD)
    ])
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale = (0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transform
])


class FairCheXpertDataset(Dataset):

    def __init__(self,
                 image_paths: List[str],
                 targets: torch.Tensor,
                 protected_attributes: torch.Tensor,
                 protected_attribute_index: int = 0,
                 transform: Callable[..., Any] = None,
                 load_to_ram: bool = True):
        super().__init__()

        self.images = []
        if load_to_ram:
            print("loading images to RAM")
            for path in image_paths:
                img = Image.open(path).convert('RGB')
                self.images.append(img)
            print("images loaded to RAM")
        else:
            self.images = image_paths
        self.targets = targets
        self.protected_attributes = protected_attributes
        self.protected_attribute_index = protected_attribute_index
        self.transform = transform

        self.yield_protected_attribute = True

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        
        img = self.images[index]

        if isinstance(img, str):
            img = Image.open(img).convert('RGB')

        img = self.transform(img)

        if self.yield_protected_attribute:
            return img, self.targets[index], self.protected_attributes[self.protected_attribute_index, index]
        else:
            return img, self.targets[index]


def get_chexpert(protected_attribute = 0, load_to_ram=True):

    demo_df = pd.read_excel(os.path.join(CHEXPERT_PATH, 'CHEXPERT_DEMO.xlsx'))
    demo_df = demo_df.rename(columns={'PATIENT': 'patient', 'GENDER': 'gender', 'AGE_AT_CXR': 'age', 'PRIMARY_RACE': 'race'})
    # drop other columns that the renamed ones
    demo_df = demo_df[['patient', 'gender', 'age', 'race']]
    # clean race
    demo_df['race'] = demo_df['race'].str.split(',').str[0]
    demo_df['race'] = demo_df['race'].str.split(' - ').str[0]
    demo_df['race'] = demo_df['race'].str.split(' or ').str[0]
    # drop without race
    print("# patients general", len(demo_df))
    demo_df.loc[demo_df['race'].str.contains('Unknown', na=False), 'race'] = pd.NA
    demo_df.dropna(subset=['race'], inplace=True)
    print("# patients with race", len(demo_df))

    # race to binary
    demo_df["white"]= demo_df['race'].apply(lambda x: 1 if x == "White" else 0)
    # age to binary
    demo_df['old'] = demo_df['age'].apply(lambda x: 1 if x > 40 else 0)
    # gender to binary
    demo_df['woman'] = demo_df['gender'].apply(lambda x: 1 if x == "Female" else 0)

    # drop irrelevant columns
    demo_df = demo_df[['patient', 'woman', 'old', 'white']]

    train_df = pd.read_csv(os.path.join(CHEXPERT_PATH, 'train.csv'))
    val_df = pd.read_csv(os.path.join(CHEXPERT_PATH, 'valid.csv'))
    test_df = pd.read_csv(os.path.join(CHEXPERT_PATH, 'test.csv'))

    dfs = [train_df, val_df, test_df]
    for i in range(len(dfs)):
        df = dfs[i]
        df = df[['Path', 'No Finding']]
        df = df.rename(columns={'Path': 'path', 'No Finding': 'label'})
        df['label'] = df['label'].apply(lambda x: 1 if x == 1 else 0)
        df['patient'] = df['path'].str.split('/').str[2]
        df = pd.merge(df, demo_df, on='patient', how='inner')
        dfs[i] = df
    train_df, val_df, test_df = dfs

    train_ds = FairCheXpertDataset(
        image_paths=[os.path.join(DATASETS_PATH, path) for path in train_df['path'].values],
        targets=torch.tensor(train_df['label'].values),
        protected_attributes=torch.stack([torch.tensor(train_df['old'].values),
                                          torch.tensor(train_df['woman'].values),
                                          torch.tensor(train_df['white'].values)]),
        protected_attribute_index=protected_attribute,
        transform=train_transform,
        load_to_ram=load_to_ram)
    val_ds = FairCheXpertDataset(
        image_paths=[os.path.join(DATASETS_PATH, path) for path in val_df['path'].values],
        targets=torch.tensor(val_df['label'].values),
        protected_attributes=torch.stack([torch.tensor(val_df['old'].values),
                                          torch.tensor(val_df['woman'].values),
                                          torch.tensor(val_df['white'].values)]),
        protected_attribute_index=protected_attribute,
        transform=transform,
        load_to_ram=load_to_ram)
    test_ds = FairCheXpertDataset(
        image_paths=[os.path.join(DATASETS_PATH, path) for path in test_df['path'].values],
        targets=torch.tensor(test_df['label'].values),
        protected_attributes=torch.stack([torch.tensor(test_df['old'].values),
                                          torch.tensor(test_df['woman'].values),
                                          torch.tensor(test_df['white'].values)]),
        protected_attribute_index=protected_attribute,
        transform=transform,
        load_to_ram=load_to_ram)

    return train_ds, val_ds, test_ds


class TransformWrapper(Dataset):

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        original_transform, yield_protected_attribute = _set_transform(self.dataset, transform)

        if yield_protected_attribute:
            img, target, pa = self.dataset[index]
        else:
            img, target = self.dataset[index]

        _set_transform(self.dataset, original_transform)

        if yield_protected_attribute:
            return img, target, pa
        return img, target


def _set_transform(dataset: Dataset, _transform: Callable[..., Any]) -> Tuple[Callable[..., Any], bool]:

    yield_protected_attribute = True

    # only goes to first transform found
    try:
        original_transform = dataset.transform
        dataset.transform = _transform
    except AttributeError:
        try:
            dataset.dataset
        except AttributeError:
            raise Exception("No transform found in any of the nested datasets")
        original_transform, yield_protected_attribute = _set_transform(dataset.dataset, _transform)

    try:
        ypa = dataset.yield_protected_attribute
        yield_protected_attribute = ypa and yield_protected_attribute
    except AttributeError:
        pass

    return original_transform, yield_protected_attribute
