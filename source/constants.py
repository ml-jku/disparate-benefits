import os
from typing import Final

# general paths
DATASETS_PATH: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "datasets"))
RESULTS_PATH: Final[str] =  os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "results"))
PLOTS_PATH: Final[str] =  os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "plots"))

# vision datasets
FAIRFACE_PATH: Final[str] = os.path.abspath(os.path.join(DATASETS_PATH, "FairFace"))
UTK_PATH: Final[str] = os.path.abspath(os.path.join(DATASETS_PATH, "UTK"))
CHEXPERT_PATH: Final[str] = os.path.abspath(os.path.join(DATASETS_PATH, "CheXpert-v1.0"))

# ImageNet statistics
IMAGENET_MEAN: Final[list] = [0.485, 0.456, 0.406]
IMAGENET_STD: Final[list] = [0.229, 0.224, 0.225]