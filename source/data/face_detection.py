import os
import gdown
import shutil
import pandas as pd
from typing import Any, Callable, Final, Tuple, List

import torch
from torchvision import datasets
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms

from ..constants import FAIRFACE_PATH, UTK_PATH, IMAGENET_MEAN, IMAGENET_STD

# Original source of UTK: https://susanqq.github.io/UTKFace/ not available anymore
# Took reupload from https://www.kaggle.com/datasets/abhikjha/utk-face-cropped

# Official transform from https://github.com/dchen236/FairFace/blob/master/predict.py
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, 
                             std=IMAGENET_STD)
    ])
# Adding horizontal flip augmentation (mild augmentation) for training on FairFace
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transform
])
# Adding resize to 224x224 for evaluation on UTK
utk_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transform
])

# Target codings FairFace
gender_dict = {'Female': 0, 'Male': 1}
age_dict = {'0-2': 0, '3-9': 1, '10-19': 2, '20-29': 3, '30-39': 4, '40-49': 5, '50-59': 6, '60-69': 7, 'more than 70': 8}
age_binarize = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1, 1]) # [[0, 1, 2, 3], [4, 5, 6, 7, 8]]
race_dict_ff = {'White': 0, 'Black': 1, 'Middle Eastern': 2, 'East Asian': 3, 'Southeast Asian': 4, 'Indian': 5, 'Latino_Hispanic': 6}
race_binarize_ff = torch.LongTensor([1, 0, 0, 0, 0, 0, 0]) # White / Non-White
race_binarize_ff_old = torch.LongTensor([1, 0, 1, 0, 0, 0, 1]) # [[0, 2, 6], [1, 3, 4, 5]]
# Target codings UTK
age_cutoffs = [2, 9, 19, 29, 39, 49, 59, 69] # corresponds to 'age_dict'
race_binarize_utk = torch.LongTensor([1, 0, 0, 0, 0]) # White / Non-White
race_binarize_utk_old = torch.LongTensor([1, 0, 0, 0, 1])

#########################
# FAIRFACE RELATED CODE #
#########################

class FairVisionDataset(datasets.ImageFolder):
    def __init__(self, 
                 root: str, 
                 transform: Callable[..., Any] | None, 
                 targets: torch.Tensor,
                 target_index = 0,
                 protected_attribute_index = 1,
                 target_transform: Callable[..., Any] | None = None, 
                 loader: Callable[[str], Any] = default_loader, 
                 is_valid_file: Callable[[str], bool] | None = None):
        super().__init__(root, transform, target_transform, loader, is_valid_file)

        self.targets = targets
        self.target_index = target_index
        self.protected_attribute_index = protected_attribute_index

        self.yield_protected_attribute = True

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # targets from ImageFolder only dummy, this is the real target and protected attribute
        img, _ = super().__getitem__(index)

        if self.yield_protected_attribute:
            return img, self.targets[self.target_index, index], self.targets[self.protected_attribute_index, index]
        else:
            return img, self.targets[self.target_index, index]


def get_fair_face(target: int = 0, protected_attribute = 1, binarize=True, augment: bool = True, download: bool = True):
    """
    Possible targets / protected attributes are ["age", "gender", "race (old)", "race"]
    """

    if download:
        _download_fair_face(FAIRFACE_PATH)

    targets = _get_labels_protected_attributes("train", binarize)
    full_train = FairVisionDataset(root=os.path.join(FAIRFACE_PATH, "train"), 
                                 transform=train_transform if augment else transform, 
                                 targets=targets,
                                 target_index=target,
                                 protected_attribute_index = protected_attribute)

    # use val set as testset, split val from full train set in main experiment code if necessary
    targets = _get_labels_protected_attributes("val", binarize)
    test = FairVisionDataset(root=os.path.join(FAIRFACE_PATH, "val"), 
                           transform=transform, 
                           targets=targets,
                           target_index=target,
                           protected_attribute_index = protected_attribute)

    return full_train, test
    
def _get_labels_protected_attributes(split:str, binarize:bool):
    if split == "train":
        df = pd.read_csv(os.path.join(FAIRFACE_PATH, "fairface_label_train.csv"), delimiter=',', header=0)
    elif split == "val":
        df = pd.read_csv(os.path.join(FAIRFACE_PATH, "fairface_label_val.csv"), delimiter=',', header=0)
    else:
        raise ValueError(f"'split' must be 'train' or 'val', was {split}")
    
    # sort by images, to have consistent matching with images
    df = df.sort_values(by='file')

    df.head(n=30)
    
    # convert strings to integers
    df['gender'].replace(gender_dict, inplace=True)
    df['age'].replace(age_dict, inplace=True)
    df['race'].replace(race_dict_ff, inplace=True)
    df['race (old)'] = df['race']

    targets = torch.stack([torch.LongTensor(df[t]) for t in ["age", "gender", "race (old)", "race"]], dim=0)

    if binarize:
        targets[0] = age_binarize[targets[0]]
        targets[2] = race_binarize_ff_old[targets[2]]
        targets[3] = race_binarize_ff[targets[3]]
    
    return targets

def move_contents_to_subfolder(original_folder, inner_directory_name):
    # Construct the path for the outer directory
    parent_path = os.path.dirname(original_folder)
    outer_directory_path = os.path.join(parent_path, "temp")
    new_directory_path = os.path.join(original_folder, inner_directory_name)
    temp_directory_path = os.path.join(original_folder, os.path.basename(original_folder))
    
    # Check if the outer directory already exists
    if os.path.exists(outer_directory_path):
        raise ValueError(f"'temp' already exists in '{parent_path}'. " +
                         "Choose a different name or location.")

    # Create the outer directory
    os.makedirs(outer_directory_path)

    # Move the original folder into the outer directory
    try:
        shutil.move(original_folder, outer_directory_path)
    except Exception as e:
        print(f"Error moving {original_folder}. Reason: {e}")

    try:
        os.rename(outer_directory_path, original_folder)
        os.rename(temp_directory_path, new_directory_path)
    except Exception as e:
        print(f"Error renaming paths. Reason: {e}")
    

def _download_fair_face(path:str):

    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(os.path.join(path, "fairface_label_train.csv")):
        print("Download train labels")
        gdown.download("https://drive.google.com/file/d/1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH/view", 
                       os.path.join(path, "fairface_label_train.csv"), fuzzy=True)
    if not os.path.exists(os.path.join(path, "fairface_label_val.csv")):
        print("Download val labels")
        gdown.download("https://drive.google.com/file/d/1wOdja-ezstMEp81tX1a-EYkFebev4h7D/view", 
                       os.path.join(path, "fairface_label_val.csv"), fuzzy=True)
    if not os.path.exists(os.path.join(path, "fairface-img-margin025-trainval.zip")):
        print("Download images")
        gdown.download("https://drive.google.com/file/d/1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86/view", 
                       os.path.join(path, "fairface-img-margin025-trainval.zip"), fuzzy=True)
    if not os.path.exists(os.path.join(path, "train") or os.path.join(path, "val")):
        print("unpacking...")
        shutil.unpack_archive(os.path.join(path, "fairface-img-margin025-trainval.zip"), os.path.join(path))
        print("moving to subfolder")
        # necessary for ImageFolder dataset
        move_contents_to_subfolder(os.path.join(path, "train"), "class_0")
        move_contents_to_subfolder(os.path.join(path, "val"), "class_0")
        print("unpacked")

####################
# UTK RELATED CODE #
####################
        
def get_utk(target: int = 0, protected_attribute = 1, binarize=True):
    """
    Possible targets / protected attributes are ["age", "gender", "race (old)", "race"]
    """

    os.makedirs(UTK_PATH, exist_ok=True)

    if not os.path.exists(os.path.join(UTK_PATH, "archive.zip")):
        raise Exception("It is assumed that the archive containing the dataset was downloaded from " \
                        + "'https://www.kaggle.com/datasets/abhikjha/utk-face-cropped' " \
                        + f"and put to {UTK_PATH}")

    if not os.path.exists(os.path.join(UTK_PATH, "utkcropped")):
        print("unpacking...")
        shutil.unpack_archive(os.path.join(UTK_PATH, "archive.zip"), os.path.join(UTK_PATH))
        # make sure the duplicate internal folder is removed
        if os.path.exists(os.path.join(UTK_PATH, "utkcropped", "utkcropped")):
            print("remove duplicate subfolder")
            shutil.rmtree(os.path.join(UTK_PATH, "utkcropped", "utkcropped"))
        print("moving to subfolder")
        move_contents_to_subfolder(os.path.join(UTK_PATH, "utkcropped"), "class_0")
        print("unpacked")

    # remove images without race annotation
    files = ["39_1_20170116174525125.jpg.chip.jpg",
             "61_1_20170109142408075.jpg.chip.jpg",
             "61_1_20170109150557335.jpg.chip.jpg",
             "61_3_20170109150557335.jpg.chip.jpg"]
    for file in files:
        if os.path.exists(os.path.join(UTK_PATH, "utkcropped", "class_0", file)):
            os.remove(os.path.join(UTK_PATH, "utkcropped", "class_0", file))
            print(f"Broken Image {file} removed successfully")
    

    targets = _get_utk_labels_protected_attributes(binarize)
    
    dataset = FairVisionDataset(root=os.path.join(UTK_PATH, "utkcropped"), 
                              transform=utk_transform, 
                              targets=targets,
                              target_index=target,
                              protected_attribute_index = protected_attribute)

    return dataset

def _get_utk_labels_protected_attributes(binarize:bool):
    ages, genders, races = list(), list(), list()

    # Loop through the files and directories in the specified folder
    entries = list()
    for _, _, fnames in sorted(os.walk(os.path.join(UTK_PATH, "utkcropped", "class_0"), followlinks=True)):
        for fname in sorted(fnames):
            entries.append(fname)
    for entry in entries:
        attributes = entry.split("_")

        assert len(attributes) == 4, f"Naming of Image is wrong, should be age_gender_race_... but was {entry}"
        
        age = int(attributes[0])
        for i in range(0, len(age_cutoffs)):
            if (i == 0 and age < age_cutoffs[i]) or (age > age_cutoffs[i - 1] and age < age_cutoffs[i]):
                ages.append(i)
                break
            elif i == (len(age_cutoffs) - 1):
                ages.append(len(age_cutoffs))
        
        # utk has inverted gender w.r.t. FairFace
        genders.append(abs(int(attributes[1]) - 1))
        races.append(int(attributes[2]))

    targets = torch.stack([torch.LongTensor(t) for t in [ages, genders, races, races]], dim=0)

    if binarize:
        targets[0] = age_binarize[targets[0]]
        targets[2] = race_binarize_utk_old[targets[2]]
        targets[3] = race_binarize_utk[targets[3]]
    
    return targets