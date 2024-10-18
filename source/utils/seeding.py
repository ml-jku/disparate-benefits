import random
import numpy as np
import torch


def fix_seeds(seed: int = 42):
    # standard
    random.seed(seed)
    np.random.seed(seed)

    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    """
    if not torch.are_deterministic_algorithms_enabled():
        torch.use_deterministic_algorithms(True)
    """
