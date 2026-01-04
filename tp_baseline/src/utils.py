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

def build_model(cfg: dict):
    """
    Build model from cfg["model"].
    cfg["model"]["type"] supports:
      - "baseline"      -> TransformerBaseline
      - "style"         -> TransformerStyleBaseline
      - "wayformer"     -> WayformerBaseline (src/wayformer)
    """
    mcfg = cfg.get("model", {})
    mtype = mcfg.get("type", "style")

    # remove "type" from kwargs
    kwargs = {k: v for k, v in mcfg.items() if k != "type"}

    if mtype == "baseline":
        from src.models.transformer_baseline import TransformerBaseline
        return TransformerBaseline(**kwargs)

    if mtype == "style":
        from src.models.transformer_style import TransformerStyleBaseline
        return TransformerStyleBaseline(**kwargs)

    if mtype == "wayformer":
        from src.wayformer.transformer_wayformer import WayformerBaseline
        return WayformerBaseline(**kwargs)
    
    if mtype == "wayformer_style":
        from src.wayformer.transformer_wayformer import WayformerStyle
        return WayformerStyle(**kwargs)

    raise ValueError(f"Unknown model.type: {mtype}")