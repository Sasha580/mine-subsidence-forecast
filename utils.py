import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)                     # Python RNG
    np.random.seed(seed)                  # NumPy RNG
    torch.manual_seed(seed)               # CPU RNG
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU

    # Make cuDNN deterministic (slower but repeatable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
