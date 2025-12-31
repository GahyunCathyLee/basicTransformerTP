from typing import Any, Dict, List
import torch

def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    x_ego = torch.stack([b["x_ego"] for b in batch], dim=0)       # (B,T,ego_dim)
    x_nb  = torch.stack([b["x_nb"] for b in batch], dim=0)        # (B,T,K,nb_dim)
    nb_mask = torch.stack([b["nb_mask"] for b in batch], dim=0)   # (B,T,K)
    y = torch.stack([b["y"] for b in batch], dim=0)               # (B,Tf,2)

    x_last_abs = torch.stack([b["x_last_abs"] for b in batch], dim=0)  # (B,2)

    out = {
        "x_ego": x_ego,
        "x_nb": x_nb,
        "nb_mask": nb_mask,
        "y": y,
        "x_last_abs": x_last_abs,
    }

    # ---- driving style (optional) ----
    if "style_prob" in batch[0]:
        out["style_prob"] = torch.stack([b["style_prob"] for b in batch], dim=0)   # (B,Ks)
        out["style_valid"] = torch.stack([b["style_valid"] for b in batch], dim=0) # (B,)
        if "style_label" in batch[0]:
            out["style_label"] = torch.stack([b["style_label"] for b in batch], dim=0)  # (B,)

    if "meta" in batch[0]:
        out["meta"] = [b.get("meta", {}) for b in batch]
    return out
