import argparse
from pathlib import Path
import numpy as np
import torch


def convert_npz_to_pt(npz_path: Path, pt_path: Path):
    data = np.load(npz_path, allow_pickle=True)

    out = {}
    for k in data.files:
        v = data[k]

        if isinstance(v, np.ndarray):
            # ✅ 문자열 / object array는 Tensor로 바꾸지 않음
            if v.dtype.kind in ("U", "S", "O"):
                out[k] = v
            else:
                out[k] = torch.from_numpy(v)
        else:
            out[k] = v

    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, pt_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", type=str, required=True)
    ap.add_argument("--pt_dir", type=str, required=True)
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    pt_dir = Path(args.pt_dir)

    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"No .npz files found in {npz_dir}")

    for npz_path in npz_files:
        pt_path = pt_dir / (npz_path.stem + ".pt")
        convert_npz_to_pt(npz_path, pt_path)
        print(f"Converted: {npz_path.name} → {pt_path.name}")


if __name__ == "__main__":
    main()
