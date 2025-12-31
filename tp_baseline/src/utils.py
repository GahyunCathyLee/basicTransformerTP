# tp_baseline/src/utils.py
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_stats_npz(path: str):
    arr = np.load(path)
    stats = {
        "ego_mean": torch.from_numpy(arr["ego_mean"]).float(),
        "ego_std":  torch.from_numpy(arr["ego_std"]).float(),
        "nb_mean":  torch.from_numpy(arr["nb_mean"]).float(),
        "nb_std":   torch.from_numpy(arr["nb_std"]).float(),
    }
    return stats
