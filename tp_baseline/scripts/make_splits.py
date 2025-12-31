# scripts/make_splits.py
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--train_end", type=int, default=45)  # 01-45 train
    ap.add_argument("--val_end", type=int, default=52)    # 46-52 val
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [f"highd_{i:02d}.pt" for i in range(1, 61)]
    for fn in files:
        if not (data_dir / fn).exists():
            raise FileNotFoundError(f"Missing: {data_dir / fn}")

    train = files[:args.train_end]
    val = files[args.train_end:args.val_end]
    test = files[args.val_end:]

    (out_dir / "train.txt").write_text("\n".join(train) + "\n")
    (out_dir / "val.txt").write_text("\n".join(val) + "\n")
    (out_dir / "test.txt").write_text("\n".join(test) + "\n")
    print("Wrote splits to:", out_dir)

if __name__ == "__main__":
    main()
