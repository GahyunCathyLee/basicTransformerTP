# scripts/compute_stats.py
import argparse
from pathlib import Path
import numpy as np
import torch

from src.datasets.highd_pt_dataset import HighDPtDataset

def save_npz_atomic(out_path: Path, **arrays):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_base = out_path.with_suffix("")          # ".../stats"
    tmp_path = Path(str(tmp_base) + ".tmp.npz")  # ".../stats.tmp.npz"

    np.savez_compressed(str(tmp_path), **arrays)
    tmp_path.replace(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--split", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = use all")
    args = ap.parse_args()

    ds = HighDPtDataset(args.data_dir, args.split, stats=None, return_meta=False)

    total = len(ds) if args.max_samples <= 0 else min(len(ds), args.max_samples)

    # we will accumulate sums in float64 for numeric stability
    ego_sum = torch.zeros(16, dtype=torch.float64)
    ego_sumsq = torch.zeros(16, dtype=torch.float64)

    nb_sum = torch.zeros(9, dtype=torch.float64)
    nb_sumsq = torch.zeros(9, dtype=torch.float64)

    ego_count = 0
    nb_count = 0

    for i in range(total):
        item = ds[i]
        x_ego = item["x_ego"]  # (T,16)
        x_nb  = item["x_nb"]   # (T,K,9)

        # ---- ego stats (always valid over T) ----
        ego_flat = x_ego.reshape(-1, 16).to(torch.float64)  # (T,16) -> (T,16)
        ego_sum += ego_flat.sum(dim=0)
        ego_sumsq += (ego_flat ** 2).sum(dim=0)
        ego_count += ego_flat.shape[0]

        # ---- neighbor stats (mask-aware) ----
        # if dataset provides nb_mask, use it to exclude padded neighbors
        nb_mask = item.get("nb_mask", None)  # (T,K) bool, True=valid expected
        nb_flat = x_nb.reshape(-1, 9).to(torch.float64)      # (T*K,9)

        if nb_mask is not None:
            mask_flat = nb_mask.reshape(-1)  # (T*K,)
            # keep only valid neighbors
            if mask_flat.dtype != torch.bool:
                mask_flat = mask_flat.bool()
            valid_nb = nb_flat[mask_flat]
            if valid_nb.numel() > 0:
                nb_sum += valid_nb.sum(dim=0)
                nb_sumsq += (valid_nb ** 2).sum(dim=0)
                nb_count += valid_nb.shape[0]
        else:
            # fallback: include all neighbors (may include padding)
            nb_sum += nb_flat.sum(dim=0)
            nb_sumsq += (nb_flat ** 2).sum(dim=0)
            nb_count += nb_flat.shape[0]

        if (i + 1) % 2000 == 0:
            print(f"[{i+1}/{total}] ego_count={ego_count}, nb_count={nb_count}")

    if ego_count == 0:
        raise RuntimeError("ego_count is 0. Dataset may be empty or corrupted.")
    if nb_count == 0:
        print("WARNING: nb_count is 0 (no valid neighbors found). nb_mean/std will be zeros/ones fallback.")

    ego_mean = ego_sum / ego_count
    ego_var = ego_sumsq / ego_count - ego_mean**2
    ego_std = torch.sqrt(torch.clamp(ego_var, min=1e-6))

    if nb_count > 0:
        nb_mean = nb_sum / nb_count
        nb_var = nb_sumsq / nb_count - nb_mean**2
        nb_std = torch.sqrt(torch.clamp(nb_var, min=1e-6))
    else:
        nb_mean = torch.zeros(9, dtype=torch.float64)
        nb_std = torch.ones(9, dtype=torch.float64)

    out = Path(args.out)
    save_npz_atomic(
        out,
        ego_mean=ego_mean.cpu().numpy().astype(np.float32),
        ego_std=ego_std.cpu().numpy().astype(np.float32),
        nb_mean=nb_mean.cpu().numpy().astype(np.float32),
        nb_std=nb_std.cpu().numpy().astype(np.float32),
    )
    print("Saved stats to:", out)

if __name__ == "__main__":
    main()