# src/datasets/highd_pt_dataset.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset

@dataclass
class HighDPtPaths:
    data_dir: Path
    split_file: Path

def read_split_list(split_file: Path) -> List[str]:
    items = [ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()]
    return items

class HighDPtDataset(Dataset):
    """
    Loads multiple recordings (.pt) and provides sample-level access.

    Returns per sample:
      x_ego: (T, 16)
      x_nb : (T, K, 9)
      nb_mask: (T, K) bool
      y: (Tf, 2)
      meta (optional): dict
    """
    def __init__(
        self,
        data_dir: str,
        split_file: str,
        stats: Optional[Dict[str, torch.Tensor]] = None,
        return_meta: bool = False,
        cache_in_memory: bool = True,

        # ---- ablation flags ----
        use_neighbors: bool = True,
        use_context: bool = True,
        use_safety: bool = True,
        use_preceding: bool = True,
        use_lane: bool = True,
        use_static: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split_file = Path(split_file)
        self.return_meta = return_meta
        self.stats = stats  # dict with means/stds or None
        self.cache_in_memory = cache_in_memory

        self.pt_names = read_split_list(self.split_file)
        self.pt_paths = [self.data_dir / n for n in self.pt_names]

        # Load all recordings (simple baseline: keep in memory)
        self.recs: List[Dict[str, Any]] = []
        self.rec_sizes: List[int] = []
        for p in self.pt_paths:
            d = torch.load(p, map_location="cpu", weights_only=False)
            self.recs.append(d)
            self.rec_sizes.append(int(d["x_hist"].shape[0]))

        # prefix sums for global index -> (rec_idx, local_idx)
        self.prefix = [0]
        s = 0
        for n in self.rec_sizes:
            s += n
            self.prefix.append(s)

        # infer T, Tf, K
        d0 = self.recs[0]
        self.T = int(d0["x_hist"].shape[1])
        self.Tf = int(d0["y_fut"].shape[1])
        self.K = int(d0["nb_hist"].shape[2])

        self.use_neighbors = use_neighbors
        self.use_context = use_context
        self.use_safety = use_safety
        self.use_preceding = use_preceding
        self.use_lane = use_lane
        self.use_static = use_static

        # full feature index map (for slicing stats)
        self.EGO_IDX = {
            "hist": list(range(0, 6)),
            "context": list(range(6, 8)),
            "safety": list(range(8, 11)),
            "preceding": [11],
            "lane": [12],
            "static": list(range(13, 16)),
        }
        self.NB_IDX = {
            "hist": list(range(0, 6)),
            "static": list(range(6, 9)),
        }

        # determine selected dims
        self.ego_feat_idx = self._make_ego_feat_idx()
        self.nb_feat_idx = self._make_nb_feat_idx()
        self.ego_dim = len(self.ego_feat_idx)
        self.nb_dim = len(self.nb_feat_idx)        

    def _make_ego_feat_idx(self):
        idx = []
        idx += self.EGO_IDX["hist"]
        if self.use_context: idx += self.EGO_IDX["context"]
        if self.use_safety: idx += self.EGO_IDX["safety"]
        if self.use_preceding: idx += self.EGO_IDX["preceding"]
        if self.use_lane: idx += self.EGO_IDX["lane"]
        if self.use_static: idx += self.EGO_IDX["static"]
        return idx

    def _make_nb_feat_idx(self):
        idx = []
        idx += self.NB_IDX["hist"]
        if self.use_static:  # neighbor static on/off is tied to use_static
            idx += self.NB_IDX["static"]
        return idx

    def __len__(self) -> int:
        return self.prefix[-1]

    def _locate(self, idx: int) -> Tuple[int, int]:
        # binary search over prefix sums
        lo, hi = 0, len(self.rec_sizes) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.prefix[mid] <= idx < self.prefix[mid + 1]:
                return mid, idx - self.prefix[mid]
            if idx < self.prefix[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        raise IndexError(idx)

    def _normalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        # std clamp
        std = torch.clamp(std, min=1e-6)
        return (x - mean) / std



    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_i, local_i = self._locate(idx)
        d = self.recs[rec_i]

        def _as_TD(x: torch.Tensor, T: int) -> torch.Tensor:
            x = x.to(torch.float32)
            # (T,) -> (T,1)
            if x.dim() == 1:
                return x.unsqueeze(-1)
            # (T, something...) -> (T, -1)
            return x.reshape(T, -1)

        # --- load raw ---
        x_hist = d["x_hist"][local_i]          # expected (T,6)
        y_fut  = d["y_fut"][local_i]           # (Tf,2)
        tv_static = d["tv_static"][local_i]    # (3,)
        tv_lane = d["tv_lane"][local_i]        # (T,) or (T,1)
        tv_context = d["tv_context"][local_i]  # maybe (T,2) or (T,1,2)
        tv_safety  = d["tv_safety"][local_i]   # maybe (T,3) or (T,1,3)
        tv_preceding = d["tv_preceding"][local_i]  # (T,) confirmed

        nb_hist  = d["nb_hist"][local_i]
        nb_mask  = d["nb_mask"][local_i].bool()
        nb_static = d["nb_static"][local_i]

        # --- force shapes to (T, D) ---
        x_hist      = _as_TD(x_hist, self.T)          # (T,6)
        # Stage1: last observed absolute position (meters)
        x_last_abs = x_hist[-1, 0:2].clone()  # (2,)
        tv_context  = _as_TD(tv_context, self.T)      # (T,2) even if stored weird
        tv_safety   = _as_TD(tv_safety, self.T)       # (T,3)
        tv_preceding= _as_TD(tv_preceding, self.T)    # (T,1)
        tv_lane_f   = _as_TD(tv_lane, self.T)         # (T,1)

        tv_static = tv_static.to(torch.float32).reshape(1, -1)   # (1,3)
        tv_static_rep = tv_static.expand(self.T, -1)             # (T,3)

        # --- ego concat: all are (T, *) now ---
        x_ego = torch.cat([
            x_hist,          # (T,6)
            tv_context,      # (T,2)
            tv_safety,       # (T,3)
            tv_preceding,    # (T,1)
            tv_lane_f,       # (T,1)
            tv_static_rep,   # (T,3)
        ], dim=-1)           # (T,16)

        # --- neighbors: enforce (T,K,*) ---
        nb_hist = nb_hist.to(torch.float32)
        if nb_hist.dim() == 4:
            # (T,K,6) ok
            pass
        elif nb_hist.dim() == 5:
            # (T,K,1,6) 같은 경우 대비
            nb_hist = nb_hist.reshape(self.T, self.K, -1)

        nb_static = nb_static.to(torch.float32).reshape(self.K, -1)  # (K,3)
        nb_static_rep = nb_static.unsqueeze(0).expand(self.T, -1, -1)  # (T,K,3)

        x_nb = torch.cat([nb_hist, nb_static_rep], dim=-1)  # (T,K,9)

        # -----------------------------
        # (A) feature selection (ablation)
        # -----------------------------
        x_ego_full = x_ego
        x_nb_full  = x_nb

        # select dims by indices computed in __init__
        x_ego = x_ego_full[:, self.ego_feat_idx]           # (T, ego_dim)
        x_nb  = x_nb_full[:, :, self.nb_feat_idx]          # (T, K, nb_dim)

        # -----------------------------
        # (B) normalization (slice stats to match selected dims)
        # -----------------------------
        if self.stats is not None:
            ego_mean = self.stats["ego_mean"][self.ego_feat_idx]
            ego_std  = self.stats["ego_std"][self.ego_feat_idx]
            x_ego = self._normalize(x_ego, ego_mean, ego_std)

            nb_mean = self.stats["nb_mean"][self.nb_feat_idx]
            nb_std  = self.stats["nb_std"][self.nb_feat_idx]
            x_nb = self._normalize(x_nb, nb_mean, nb_std)

        # -----------------------------
        # (C) neighbors ablation: keep shape but zero-out
        # -----------------------------
        if not self.use_neighbors:
            nb_mask = torch.zeros_like(nb_mask, dtype=torch.bool)  # (T,K)
            x_nb = torch.zeros_like(x_nb)                          # (T,K,nb_dim)
        
        out = {
            "x_ego": x_ego,        # (T, ego_dim)
            "x_nb": x_nb,          # (T, K, nb_dim)
            "nb_mask": nb_mask,    # (T, K) bool
            "y": y_fut.to(torch.float32),  # (Tf,2)
            "x_last_abs": x_last_abs,  
        }

        # ---- driving style (optional; backward compatible) ----
        # expected keys in .pt: style_prob (Ks,), style_valid (), optional style_label ()
        if "style_prob" in d and "style_valid" in d:
            out["style_prob"] = d["style_prob"][local_i].to(torch.float32)  # (Ks,)
            out["style_valid"] = d["style_valid"][local_i].bool()          # ()
            if "style_label" in d:
                out["style_label"] = d["style_label"][local_i].to(torch.int64)  # ()

        if self.return_meta:
            meta = {}
            for k in ["recordingId", "trackId", "t0_frame", "drivingDirection"]:
                if k in d:
                    meta[k] = d[k][local_i]
            out["meta"] = meta
        return out
