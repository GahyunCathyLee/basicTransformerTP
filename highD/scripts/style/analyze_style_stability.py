#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def entropy(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def tv_distance(p: np.ndarray) -> np.ndarray:
    # TV between consecutive distributions
    return 0.5 * np.sum(np.abs(p[1:] - p[:-1]), axis=1)


def js_divergence(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    q = p.copy()
    # consecutive q = p_{t-1}
    q[1:] = p[:-1]
    q[0] = p[0]
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=1)
    kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=1)
    js = 0.5 * (kl_pm + kl_qm)
    return js[1:]  # meaningful for transitions only


def switch_count(labels: np.ndarray) -> int:
    if labels.size <= 1:
        return 0
    return int(np.sum(labels[1:] != labels[:-1]))


def mean_run_length(labels: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    runs = []
    cur = labels[0]
    length = 1
    for x in labels[1:]:
        if x == cur:
            length += 1
        else:
            runs.append(length)
            cur = x
            length = 1
    runs.append(length)
    return float(np.mean(runs))


def dominance_ratio(labels: np.ndarray, K: int) -> float:
    if labels.size == 0:
        return float("nan")
    counts = np.bincount(labels, minlength=K)
    return float(counts.max() / counts.sum())


def label_entropy(labels: np.ndarray, K: int) -> float:
    if labels.size == 0:
        return float("nan")
    counts = np.bincount(labels, minlength=K).astype(np.float32)
    q = counts / counts.sum()
    return float(entropy(q[None, :])[0])


@dataclass
class AnalyzeCfg:
    style_npz: Path
    out_dir: Path
    min_windows_per_vehicle: int


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style_npz", required=True, type=str, help="e.g., out_style/style_windows_L3.0s.npz")
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--min_windows_per_vehicle", type=int, default=50)
    args = ap.parse_args()

    cfg = AnalyzeCfg(style_npz=Path(args.style_npz), out_dir=Path(args.out_dir), min_windows_per_vehicle=args.min_windows_per_vehicle)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(cfg.style_npz)
    L = float(d["history_sec"][0])
    rec = d["rec_id"].astype(np.int32)
    veh = d["veh_id"].astype(np.int32)
    t0 = d["t0_frame"].astype(np.int32)
    label = d["label"].astype(np.int32)
    prob = d["prob"].astype(np.float32)
    K = prob.shape[1]

    df = pd.DataFrame({"rec_id": rec, "veh_id": veh, "t0_frame": t0, "label": label})
    df["post_entropy"] = entropy(prob)
    # sort for time order within vehicle
    df.sort_values(["rec_id", "veh_id", "t0_frame"], inplace=True)

    # vehicle-level aggregation
    rows = []
    for (rid, vid), g in df.groupby(["rec_id", "veh_id"], sort=False):
        if len(g) < cfg.min_windows_per_vehicle:
            continue
        y = g["label"].to_numpy(dtype=np.int32)
        pe = g["post_entropy"].to_numpy(dtype=np.float32)
        # align probs for this group (need indices)
        idx = g.index.to_numpy()
        p = prob[idx]

        switches = switch_count(y)
        runlen = mean_run_length(y)
        dom = dominance_ratio(y, K)
        lent = label_entropy(y, K)
        pe_mean = float(np.mean(pe))

        # transition metrics
        tv = tv_distance(p)
        js = js_divergence(p)
        tv_mean = float(np.mean(tv)) if tv.size else float("nan")
        js_mean = float(np.mean(js)) if js.size else float("nan")

        rows.append({
            "rec_id": int(rid),
            "veh_id": int(vid),
            "n_windows": int(len(g)),
            "switches": switches,
            "switch_rate": float(switches / max(1, len(g) - 1)),
            "mean_run_length": runlen,
            "dominance_ratio": dom,
            "label_entropy": lent,
            "post_entropy_mean": pe_mean,
            "tv_mean": tv_mean,
            "js_mean": js_mean,
        })

    out_df = pd.DataFrame(rows).sort_values(["rec_id", "veh_id"])
    out_csv = cfg.out_dir / f"vehicle_style_stability_L{L:.1f}s.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[DONE] saved {out_csv}")

    # Plots
    def _hist(col: str):
        plt.figure()
        out_df[col].hist(bins=40)
        plt.title(f"{col} (L={L:.1f}s)")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(cfg.out_dir / f"hist_{col}_L{L:.1f}s.png", dpi=200)
        plt.close()

    for col in ["switch_rate", "mean_run_length", "dominance_ratio", "label_entropy", "post_entropy_mean", "js_mean"]:
        if col in out_df.columns and len(out_df) > 0:
            _hist(col)

    # Simple scatter: dominance vs switch_rate
    if len(out_df) > 0:
        plt.figure()
        plt.scatter(out_df["switch_rate"], out_df["dominance_ratio"], s=8)
        plt.xlabel("switch_rate")
        plt.ylabel("dominance_ratio")
        plt.title(f"dominance vs switch_rate (L={L:.1f}s)")
        plt.tight_layout()
        plt.savefig(cfg.out_dir / f"scatter_dom_vs_switch_L{L:.1f}s.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
