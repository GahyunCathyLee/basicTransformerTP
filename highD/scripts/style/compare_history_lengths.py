#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_dirs", required=True, nargs="+",
                    help="e.g., out_style/analysis_L2 out_style/analysis_L3 out_style/analysis_L5")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for d in args.analysis_dirs:
        dpath = Path(d)
        csvs = list(dpath.glob("vehicle_style_stability_L*.csv"))
        if not csvs:
            continue
        csv = csvs[0]
        df = pd.read_csv(csv)
        # parse L from filename
        name = csv.stem
        L = float(name.split("L")[-1].replace("s", "").replace("_", "").replace("vehicle_style_stability_", ""))
        df["history_sec"] = L
        all_rows.append(df)

    if not all_rows:
        raise SystemExit("No stability CSVs found.")

    big = pd.concat(all_rows, ignore_index=True)
    big.to_csv(out_dir / "stability_all_lengths.csv", index=False)
    print(f"[DONE] saved {out_dir/'stability_all_lengths.csv'}")

    # Aggregate summary per L
    agg = big.groupby("history_sec").agg({
        "switch_rate": "mean",
        "mean_run_length": "mean",
        "dominance_ratio": "mean",
        "label_entropy": "mean",
        "post_entropy_mean": "mean",
        "js_mean": "mean",
        "n_windows": "mean",
    }).reset_index()
    agg.to_csv(out_dir / "stability_summary_by_L.csv", index=False)

    # Plot key metrics vs L
    def plot_metric(col: str):
        plt.figure()
        plt.plot(agg["history_sec"], agg[col], marker="o")
        plt.xlabel("history_sec")
        plt.ylabel(f"mean {col}")
        plt.title(f"{col} vs history length")
        plt.tight_layout()
        plt.savefig(out_dir / f"{col}_vs_L.png", dpi=200)
        plt.close()

    for col in ["switch_rate", "dominance_ratio", "post_entropy_mean", "js_mean", "mean_run_length"]:
        if col in agg.columns:
            plot_metric(col)


if __name__ == "__main__":
    main()
