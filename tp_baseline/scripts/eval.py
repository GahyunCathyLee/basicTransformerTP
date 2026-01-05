# tp_baseline/scripts/eval.py
"""
Evaluate a trained model checkpoint on a split (val/test).

Run (from tp_baseline/):
  python3 -m scripts.eval --config configs/wayformer.yaml --ckpt ckpts/wayformer/T9_Tf15_all/best.pt --split test
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import yaml
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

import numpy as np
import matplotlib.pyplot as plt

from src.datasets.highd_pt_dataset import HighDPtDataset
from src.datasets.collate import collate_batch
from src.losses import trajectory_loss, delta_to_abs, multimodal_loss
from src.metrics import ade, fde
from src.utils import set_seed, load_stats_npz, build_model


def _resolve_path(base: Path, p: str) -> Path:
    """Resolve relative paths against config file directory."""
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def _resolve_split_path(splits_dir: Path, split_arg: str) -> str:
    """
    split_arg can be:
      - "test" / "val" / "train"  -> use splits_dir/{split}.txt
      - a path to a txt file      -> use as-is
    """
    p = Path(split_arg)
    if p.exists() and p.is_file():
        return str(p.resolve())

    # treat as split name
    cand = (splits_dir / f"{split_arg}.txt").resolve()
    if not cand.exists():
        raise FileNotFoundError(
            f"Split file not found for split='{split_arg}'. Tried: {cand}\n"
            f"Either pass --split test/val/train, or pass a full path to a split txt."
        )
    return str(cand)


def _load_ckpt_state_dict(ckpt_path: Path) -> Dict[str, Any]:
    """
    Supports:
      - raw state_dict
      - dict checkpoint with 'model' key
    """
    obj = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(obj, dict) and ("model" in obj):
        return obj["model"]
    if isinstance(obj, dict) and any(k.endswith(".weight") or k.endswith(".bias") for k in obj.keys()):
        # looks like a state_dict already
        return obj
    # fallback: try common keys
    for k in ["state_dict", "net", "network"]:
        if isinstance(obj, dict) and k in obj:
            return obj[k]
    raise ValueError(
        f"Unrecognized checkpoint format at: {ckpt_path}\n"
        f"Top-level keys: {list(obj.keys()) if isinstance(obj, dict) else type(obj)}"
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    predict_delta: bool,
    w_traj: float,
    w_fde: float,
    w_cls: float,
    multimodal_k: Optional[int] = None,
    vis_one: bool = False,
    vis_out: Optional[str] = None,
) -> Dict[str, float]:
    """
    Matches train.py evaluation behavior:
      - out can be:
          pred (B,Tf,2)                         [baseline/style]
          pred (B,M,Tf,2)                       [wayformer without scores]
          (pred, scores_logits)                 [wayformer with scores]
      - If multimodal: best-of-M regression (minADE) + optional cls loss (w_cls)
      - Metrics are computed in ABS space.
    """
    model.eval()

    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    n_batches = 0
    n_samples = 0

    did_vis = False

    for batch in loader:
        x_ego = batch["x_ego"].to(device, non_blocking=True)
        x_nb = batch["x_nb"].to(device, non_blocking=True)
        nb_mask = batch["nb_mask"].to(device, non_blocking=True)

        style_prob = batch.get("style_prob", None)
        style_valid = batch.get("style_valid", None)
        if style_prob is not None:
            style_prob = style_prob.to(device, non_blocking=True)
        if style_valid is not None:
            style_valid = style_valid.to(device, non_blocking=True)

        y_abs = batch["y"].to(device, non_blocking=True)                 # (B,Tf,2)
        x_last_abs = batch["x_last_abs"].to(device, non_blocking=True)   # (B,2)

        with autocast(device_type="cuda", enabled=use_amp):
            out = model(x_ego, x_nb, nb_mask, style_prob=style_prob, style_valid=style_valid)

            if isinstance(out, (tuple, list)):
                pred, scores = out
            else:
                pred, scores = out, None

            # optionally cap number of modes for evaluation
            if pred.dim() == 4 and (multimodal_k is not None) and (multimodal_k > 0) and (multimodal_k < pred.shape[1]):
                pred = pred[:, :multimodal_k]
                if scores is not None:
                    scores = scores[:, :multimodal_k]

            if pred.dim() == 4:
                # multimodal
                loss, best_idx = multimodal_loss(
                    pred=pred,
                    y_abs=y_abs,
                    x_last_abs=x_last_abs,
                    predict_delta=predict_delta,
                    score_logits=scores,
                    w_traj=w_traj,
                    w_fde=w_fde,
                    w_cls=w_cls,
                )

                # metrics: evaluate best mode in ABS space
                if predict_delta:
                    pred_abs_all = torch.cumsum(pred, dim=2) + x_last_abs[:, None, None, :]
                else:
                    pred_abs_all = pred
                pred_abs = pred_abs_all[torch.arange(pred.shape[0], device=pred.device), best_idx]

            else:
                # single-mode
                loss = trajectory_loss(
                    pred=pred,
                    y_abs=y_abs,
                    x_last_abs=x_last_abs,
                    predict_delta=predict_delta,
                    w_traj=w_traj,
                    w_fde=w_fde,
                )
                pred_abs = delta_to_abs(pred, x_last_abs) if predict_delta else pred

        a = ade(pred_abs, y_abs)
        f = fde(pred_abs, y_abs)

        total_loss += float(loss.item())
        total_ade += float(a.item())
        total_fde += float(f.item())
        n_batches += 1
        n_samples += int(y_abs.shape[0])

        # optional visualization: first batch, first sample
        if vis_one and (not did_vis):
            history = x_ego[0, :, 0:2].detach().cpu().numpy()  # (T,2) (assumes first two ego feats are xy)
            future_gt = y_abs[0].detach().cpu().numpy()        # (Tf,2)
            future_pr = pred_abs[0].detach().cpu().numpy()     # (Tf,2)

            _plot_single_sample(history, future_gt, future_pr)

            if vis_out is not None:
                outp = Path(vis_out)
                outp.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(str(outp), dpi=150)
            plt.close()
            did_vis = True

    if n_batches == 0:
        return {"loss": float("nan"), "ade": float("nan"), "fde": float("nan"), "n_samples": 0}

    return {
        "loss": total_loss / n_batches,
        "ade": total_ade / n_batches,
        "fde": total_fde / n_batches,
        "n_samples": n_samples,
    }


def _plot_single_sample(history_xy: np.ndarray, gt_future_xy: np.ndarray, pred_future_xy: np.ndarray) -> None:
    plt.figure(figsize=(10, 4))
    # history + gt
    gt_full = np.concatenate([history_xy, gt_future_xy], axis=0)
    plt.plot(gt_full[:, 0], gt_full[:, 1], marker="o", markersize=2, linewidth=1.5, alpha=0.8, label="GT (hist+future)")
    # pred future
    pred_full = np.concatenate([history_xy[-1:, :], pred_future_xy], axis=0)
    plt.plot(pred_full[:, 0], pred_full[:, 1], marker="o", markersize=2, linewidth=1.5, alpha=0.8, label="Pred future")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectory (one sample)")
    plt.axis("equal")


def _append_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    # stable key order for readability
    base_cols = [
        "timestamp", "exp_name", "split", "ckpt_path",
        "n_samples", "loss", "ade", "fde",
        "data_dir", "stats_path",
        "model_type", "predict_delta",
        "use_neighbors",
        "w_traj", "w_fde", "w_cls",
        "multimodal_k",
    ]
    extra_cols = [k for k in row.keys() if k not in base_cols]
    fieldnames = base_cols + sorted(extra_cols)

    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)

def resolve_stats_path(stats_path: str, data_dir: str, base_dir: Path) -> Path:
    """
    stats_path:
      - "auto"면 stats/{leaf}/stats.npz 규칙으로 자동 탐색
      - 아니면 stats_path를 그대로 Path로 해석
    data_dir:
      - cfg.data.data_dir (예: data/highD_pt/T9_Tf15_all 같은 디렉토리)
    base_dir:
      - eval.py 기준 프로젝트 루트 또는 tp_baseline 루트
    """
    s = str(stats_path).strip()   
    if s != "auto":
        return Path(s)

    d = Path(data_dir)
    # data_dir이 파일을 가리키는 경우도 대비
    leaf = d.name if d.suffix == "" else d.parent.name

    # 규칙: stats/{leaf}/stats.npz
    candidates = [
        base_dir / "stats" / leaf / "stats.npz",
        base_dir / "tp_baseline" / "stats" / leaf / "stats.npz",  # 혹시 base_dir가 repo root일 때
        Path("stats") / leaf / "stats.npz",                       # 현재 cwd 기준
    ]

    for p in candidates:
        if p.exists():
            return p

    # 마지막 fallback: stats 폴더 전체에서 leaf 포함한 경로 탐색(느리지만 안전)
    stats_root = base_dir / "stats"
    if stats_root.exists():
        for p in stats_root.rglob("stats.npz"):
            if p.parent.name == leaf:
                return p

    # 못 찾으면 후보 경로를 에러 메시지에 보여주기
    cand_str = "\n".join(str(x) for x in candidates)
    raise FileNotFoundError(
        f"Stats file not found (auto).\n"
        f"data_dir leaf='{leaf}'\n"
        f"Tried:\n{cand_str}"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, required=True, help="test/val/train OR a path to split txt")
    ap.add_argument("--multimodal_k", type=int, default=0, help="0 = use all modes")
    ap.add_argument("--vis_one", action="store_true")
    ap.add_argument("--vis_out", type=str, default=None)
    ap.add_argument(
        "--csv_out",
        type=str,
        default=str(Path("results") / "results.csv"),
        help="CSV output path (default: results/results.csv)"
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg_dir = cfg_path.parent
    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text())

    # seed
    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)

    # device
    want_cuda = (cfg.get("train", {}).get("device", "cuda") == "cuda")
    device = torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")

    print("==== Environment ====")
    print("torch:", torch.__version__)
    print("device:", device)
    if device.type == "cuda":
        print("cuda:", torch.version.cuda)
        print("gpu:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    # paths
    data_dir = str(_resolve_path(cfg_dir, cfg["data"]["data_dir"]))
    splits_dir = _resolve_path(cfg_dir, cfg["data"]["splits_dir"])
    stats_path = cfg["data"]["stats_path"]
    split_path = _resolve_split_path(splits_dir, args.split)

    BASE_DIR = Path(__file__).resolve().parents[1]  # tp_baseline/

    stats_path = resolve_stats_path(
        stats_path=stats_path,
        data_dir=data_dir,
        base_dir=BASE_DIR,
    )

    # stats
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Stats file not found: {stats_path}\n"
            f"Compute it first (or ensure train auto-generated it)."
        )
    stats = load_stats_npz(str(stats_path))

    # dataset flags (match train)
    feat = cfg.get("features", {})
    use_neighbors = bool(feat.get("use_neighbors", True))

    ds = HighDPtDataset(
        data_dir, split_path, stats=stats, return_meta=False,
        use_neighbors=use_neighbors,
        use_context=feat.get("use_context", True),
        use_safety=feat.get("use_safety", True),
        use_preceding=feat.get("use_preceding", True),
        use_lane=feat.get("use_lane", True),
        use_static=feat.get("use_static", True),
    )

    batch_size = int(cfg["data"].get("batch_size", 128))
    num_workers = int(cfg["data"].get("num_workers", 2))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_batch,
        persistent_workers=(num_workers > 0),
    )

    # model (same as train)
    model = build_model(cfg).to(device)

    # load ckpt
    ckpt_path = Path(args.ckpt).resolve()
    state = _load_ckpt_state_dict(ckpt_path)
    model.load_state_dict(state, strict=True)

    # eval weights (match train defaults)
    use_amp = bool(cfg["train"].get("use_amp", True)) and (device.type == "cuda")
    predict_delta = bool(cfg.get("model", {}).get("predict_delta", False))
    w_traj = float(cfg.get("train", {}).get("w_traj", 1.0))
    w_fde = float(cfg.get("train", {}).get("w_fde", 0.0))
    w_cls = float(cfg.get("train", {}).get("w_cls", 0.0))

    multimodal_k = args.multimodal_k if args.multimodal_k > 0 else None

    metrics = evaluate(
        model=model,
        loader=loader,
        device=device,
        use_amp=use_amp,
        predict_delta=predict_delta,
        w_traj=w_traj,
        w_fde=w_fde,
        w_cls=w_cls,
        multimodal_k=multimodal_k,
        vis_one=args.vis_one,
        vis_out=args.vis_out,
    )

    model_type = str(cfg.get("model", {}).get("type", "unknown"))
    exp_name = str(cfg.get("exp_name", cfg_path.stem))

    print("\n==== Result ====")
    print("exp_name:", exp_name)
    print("split:", args.split)
    print("ckpt:", str(ckpt_path))
    print("n_samples:", metrics["n_samples"])
    print("loss:", metrics["loss"])
    print("ade :", metrics["ade"])
    print("fde :", metrics["fde"])

    # optional CSV logging
    if args.csv_out is not None:
        row: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "exp_name": exp_name,
            "split": args.split,
            "ckpt_path": str(ckpt_path),
            "n_samples": metrics["n_samples"],
            "loss": metrics["loss"],
            "ade": metrics["ade"],
            "fde": metrics["fde"],
            "data_dir": data_dir,
            "stats_path": str(stats_path),
            "model_type": model_type,
            "predict_delta": predict_delta,
            "use_neighbors": use_neighbors,
            "w_traj": w_traj,
            "w_fde": w_fde,
            "w_cls": w_cls,
            "multimodal_k": (multimodal_k if multimodal_k is not None else 0),
        }
        _append_csv(Path(args.csv_out), row)
        print("Appended to CSV:", args.csv_out)


if __name__ == "__main__":
    main()
