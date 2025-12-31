# scripts/summarize_results.py
"""
Summarize results/results.csv.

Run:
  python3 scripts/summarize_results.py --csv results/results.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="results/results.csv")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    p = Path(args.csv)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    rows = []
    with p.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if args.split and row.get("split") != args.split:
                continue
            # parse floats safely
            try:
                row["ade"] = float(row.get("ade", "nan"))
                row["fde"] = float(row.get("fde", "nan"))
                row["loss"] = float(row.get("loss", "nan"))
                row["n_samples"] = int(float(row.get("n_samples", "0")))
            except Exception:
                continue
            rows.append(row)

    if not rows:
        print("No rows found for split =", args.split)
        return

    rows.sort(key=lambda x: x["ade"])

    # print table
    print(f"Results: {p} (split={args.split})")
    print("-" * 110)
    print(f"{'rank':>4}  {'exp_name':<28}  {'ADE':>8}  {'FDE':>8}  {'loss':>8}  {'samples':>8}  {'ckpt':<30}")
    print("-" * 110)

    for i, row in enumerate(rows[: args.topk], 1):
        ckpt = row.get("ckpt_path", "")
        if len(ckpt) > 30:
            ckpt = "..." + ckpt[-27:]
        print(
            f"{i:>4}  {row.get('exp_name',''):<28}  "
            f"{row['ade']:>8.4f}  {row['fde']:>8.4f}  {row['loss']:>8.4f}  "
            f"{row['n_samples']:>8d}  {ckpt:<30}"
        )

    print("-" * 110)
    best = rows[0]
    print(f"Best by ADE: {best.get('exp_name')}  (ADE={best['ade']:.4f}, FDE={best['fde']:.4f})")


if __name__ == "__main__":
    main()
