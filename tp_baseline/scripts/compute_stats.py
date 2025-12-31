# scripts/compute_stats.py
import argparse
from pathlib import Path
import numpy as np
import torch

from src.datasets.highd_pt_dataset import HighDPtDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--split", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = use all")
    args = ap.parse_args()

    ds = HighDPtDataset(args.data_dir, args.split, stats=None, return_meta=False)

    # streaming mean/std
    n = 0
    ego_sum = torch.zeros(16)
    ego_sumsq = torch.zeros(16)
    nb_sum = torch.zeros(9)
    nb_sumsq = torch.zeros(9)

    total = len(ds) if args.max_samples <= 0 else min(len(ds), args.max_samples)

    for i in range(total):
        item = ds[i]
        x_ego = item["x_ego"]  # (T,16)
        x_nb  = item["x_nb"]   # (T,K,9)
        # flatten over time (+ neighbor slots)
        ego_flat = x_ego.reshape(-1, 16)
        nb_flat  = x_nb.reshape(-1, 9)

        ego_sum += ego_flat.sum(dim=0)
        ego_sumsq += (ego_flat ** 2).sum(dim=0)
        nb_sum += nb_flat.sum(dim=0)
        nb_sumsq += (nb_flat ** 2).sum(dim=0)

        n += ego_flat.shape[0]
        # nb has different count:
        # we'll compute separate counts
    # counts
    ego_count = total * ds.T
    nb_count  = total * ds.T * ds.K

    ego_mean = (ego_sum / ego_count)
    ego_var  = (ego_sumsq / ego_count) - ego_mean**2
    ego_std  = torch.sqrt(torch.clamp(ego_var, min=1e-6))

    nb_mean = (nb_sum / nb_count)
    nb_var  = (nb_sumsq / nb_count) - nb_mean**2
    nb_std  = torch.sqrt(torch.clamp(nb_var, min=1e-6))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        ego_mean=ego_mean.numpy(),
        ego_std=ego_std.numpy(),
        nb_mean=nb_mean.numpy(),
        nb_std=nb_std.numpy(),
    )
    print("Saved stats to:", out)

if __name__ == "__main__":
    main()
