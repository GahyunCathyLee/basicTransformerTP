# tp_baseline/src/eval.py
"""
Evaluate a trained model checkpoint on a split (test/val).

Run (from tp_baseline/):
  python3 -m src.eval --config configs/ablation_no_safety.yaml --ckpt ckpts/ablation_no_safety/best.pt --split test
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from src.datasets.highd_pt_dataset import HighDPtDataset
from src.datasets.collate import collate_batch
from src.models.transformer_baseline import TransformerBaseline
from src.models.transformer_style import TransformerStyleBaseline
from src.losses import delta_to_abs, trajectory_loss
from src.metrics import ade, fde
from src.utils import load_stats_npz, set_seed

import matplotlib.pyplot as plt
import numpy as np

def _resolve_path(base: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


@dataclass
class EvalResult:
    exp_name: str
    split: str
    ckpt_path: str
    n_samples: int
    loss: float
    ade: float
    fde: float
    timestamp: str
    extra: Dict[str, Any]


@torch.no_grad()
def evaluate_single(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    predict_delta: bool,
    w_traj: float,
    w_fde: float,
    multimodal_k: Optional[int] = None,
    vis_one: bool = False,
    vis_out: Optional[str] = None,
) -> Dict[str, float]:
    """
    Supports:
      - single-modal: pred (B,Tf,2)
      - multi-modal: pred (B,M,Tf,2)
    Metrics are always computed in absolute space.
    Loss uses trajectory_loss (supports delta mode + FDE weight).
    """
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    n_batches = 0
    n_samples = 0

    for batch in loader:
        x_ego = batch["x_ego"].to(device, non_blocking=True)
        x_nb = batch["x_nb"].to(device, non_blocking=True)
        nb_mask = batch["nb_mask"].to(device, non_blocking=True)
        style_prob = batch.get("style_prob", None)
        style_valid = batch.get("style_valid", None)
        if style_prob is not None: style_prob = style_prob.to(device, non_blocking=True)
        if style_valid is not None: style_valid = style_valid.to(device, non_blocking=True)
        y_abs = batch["y"].to(device, non_blocking=True)
        x_last_abs = batch["x_last_abs"].to(device, non_blocking=True)  # (B,2)

        if n_batches == 0:
            print(list(batch.keys()))

        with autocast(device_type="cuda", enabled=use_amp):
            pred = model(x_ego, x_nb, nb_mask, style_prob=style_prob, style_valid=style_valid)

        # ---- multimodal: choose best mode per sample in ABS space (minADE) ----
        # pred shapes:
        #   single: (B,Tf,2)
        #   multi : (B,M,Tf,2)
        if pred.dim() == 4:
            B, M, Tf, D = pred.shape

            # optionally restrict to top-k modes (e.g., if model outputs many)
            if multimodal_k is not None and multimodal_k < M:
                pred = pred[:, :multimodal_k]
                M = multimodal_k

            # convert each mode to absolute space if needed
            if predict_delta:
                # pred_delta: (B,M,Tf,2) -> abs per mode
                # delta_to_abs expects (B,Tf,2), so vectorize with cumsum
                pred_abs_all = torch.cumsum(pred, dim=2) + x_last_abs[:, None, None, :]
            else:
                pred_abs_all = pred  # already abs

            # compute ADE per mode, per sample: (B,M)
            # ade() likely returns scalar over batch; do manual:
            # L2 per step -> mean over Tf
            err = torch.norm(pred_abs_all - y_abs[:, None, :, :], dim=-1)  # (B,M,Tf)
            ade_bm = err.mean(dim=-1)  # (B,M)
            best_idx = ade_bm.argmin(dim=1)  # (B,)

            best_pred = pred[torch.arange(B, device=pred.device), best_idx]  # (B,Tf,2) (delta or abs)
            best_pred_abs = pred_abs_all[torch.arange(B, device=pred.device), best_idx]  # (B,Tf,2) abs

            # loss: compute using trajectory_loss on the chosen best mode
            with autocast(device_type="cuda", enabled=use_amp):
                loss = trajectory_loss(
                    pred=best_pred,
                    y_abs=y_abs,
                    x_last_abs=x_last_abs,
                    predict_delta=predict_delta,
                    w_traj=w_traj,
                    w_fde=w_fde,
                )

            a = ade(best_pred_abs, y_abs)
            f = fde(best_pred_abs, y_abs)

            if vis_one and n_batches == 0:
                history = x_ego[0, :, 0:2].detach().cpu().numpy()      # (T,2)  ← x_ego 구조가 (B,T,ego_dim)일 때
                future_gt = y_abs[0].detach().cpu().numpy()           # (Tf,2)
                future_pred = best_pred_abs[0].detach().cpu().numpy() # (Tf,2)

                plot_single_sample(history, future_gt, future_pred)

                if vis_out is not None:
                    Path(vis_out).parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(vis_out, dpi=150)
                    plt.close()
                return {"loss": 0.0, "ade": 0.0, "fde": 0.0, "n_samples": 0}

        else:
            # single-modal
            with autocast(device_type="cuda", enabled=use_amp):
                loss = trajectory_loss(
                    pred=pred,
                    y_abs=y_abs,
                    x_last_abs=x_last_abs,
                    predict_delta=predict_delta,
                    w_traj=w_traj,
                    w_fde=w_fde,
                )

            pred_abs = delta_to_abs(pred, x_last_abs) if predict_delta else pred

            if vis_one and n_batches == 0:
                history = x_ego[0, :, 0:2].detach().cpu().numpy()
                future_gt = y_abs[0].detach().cpu().numpy()
                future_pred = pred_abs[0].detach().cpu().numpy()

                plot_single_sample(history, future_gt, future_pred)

                if vis_out is not None:
                    Path(vis_out).parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(vis_out, dpi=150)
                    plt.close()
                return {"loss": 0.0, "ade": 0.0, "fde": 0.0, "n_samples": 0}

            a = ade(pred_abs, y_abs)
            f = fde(pred_abs, y_abs)

        total_loss += float(loss.item())
        total_ade += float(a.item())
        total_fde += float(f.item())
        n_batches += 1
        n_samples += int(y_abs.shape[0])

    if n_batches == 0:
        return {"loss": float("nan"), "ade": float("nan"), "fde": float("nan"), "n_samples": 0}

    return {
        "loss": total_loss / n_batches,
        "ade": total_ade / n_batches,
        "fde": total_fde / n_batches,
        "n_samples": n_samples,
    }


def _append_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    base_cols = [
        "timestamp", "exp_name", "split", "ckpt_path",
        "n_samples", "loss", "ade", "fde",
        "data_dir", "stats_path",
        "use_neighbors", "use_slot_emb",
        "use_context", "use_safety", "use_preceding", "use_lane", "use_static",
        "predict_delta", "w_traj", "w_fde",
        "T", "Tf", "K", "ego_dim", "nb_dim", "d_model", "nhead", "num_layers", "dropout",
    ]
    extra_cols = [k for k in row.keys() if k not in base_cols]
    fieldnames = base_cols + sorted(extra_cols)

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def plot_single_sample(history, future_gt, future_pred, save_path=None):
    plt.xlim(280, 420)
    plt.ylim(20, 60)
    
    plt.figure(figsize=(10, 4))

    # History
    #plt.plot(history[:, 0], history[:, 1],
    #         marker='o', linewidth=2, markersize=4,
    #         alpha=0.9, label='History', zorder=3)

    # GT (thicker solid)
    plt.plot(future_gt[:, 0], future_gt[:, 1],
             marker='o', linewidth=3, markersize=4,
             alpha=0.9, label='GT', zorder=2)

    # Pred (dashed + slightly transparent)
    plt.plot(future_pred[:, 0], future_pred[:, 1],
             marker='x', linestyle='--', linewidth=2, markersize=5,
             alpha=0.8, label='Pred', zorder=4)

    # Emphasize endpoints
    #plt.scatter([history[-1,0]], [history[-1,1]], s=60, marker='*', label='Hist end', zorder=5)
    plt.scatter([future_gt[-1,0]], [future_gt[-1,1]], s=60, marker='s', label='GT end', zorder=5)
    plt.scatter([future_pred[-1,0]], [future_pred[-1,1]], s=60, marker='D', label='Pred end', zorder=5)
    


    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160)
        plt.close()
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    ap.add_argument("--out_csv", type=str, default="results/results.csv")
    ap.add_argument("--out_json_dir", type=str, default="results/json")
    ap.add_argument("--batch_size", type=int, default=256, help="override eval batch size if you want")
    ap.add_argument("--multimodal_k", type=int, default=None, help="if model outputs (B,M,Tf,2), keep first k modes")
    ap.add_argument("--vis_one", action="store_true", help="visualize one sample (first batch, first item)")
    ap.add_argument("--vis_out", type=str, default=None, help="path to save the visualization png")

    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg_dir = cfg_path.parent
    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text())

    set_seed(int(cfg.get("train", {}).get("seed", 42)))

    want_cuda = (cfg.get("train", {}).get("device", "cuda") == "cuda")
    device = torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")
    use_amp = bool(cfg.get("train", {}).get("use_amp", True)) and (device.type == "cuda")

    # Stage1/2 flags
    predict_delta = bool(cfg.get("model", {}).get("predict_delta", False))
    w_traj = float(cfg.get("train", {}).get("w_traj", 1.0))
    w_fde = float(cfg.get("train", {}).get("w_fde", 0.0))

    data_dir = str(_resolve_path(cfg_dir, cfg["data"]["data_dir"]))
    splits_dir = _resolve_path(cfg_dir, cfg["data"]["splits_dir"])
    stats_path = _resolve_path(cfg_dir, cfg["data"]["stats_path"])

    split_file = splits_dir / ("test.txt" if args.split == "test" else "val.txt")
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    stats = load_stats_npz(str(stats_path))

    feat = cfg.get("features", {})
    train_flags = dict(
        use_neighbors=feat.get("use_neighbors", True),
        use_context=feat.get("use_context", True),
        use_safety=feat.get("use_safety", True),
        use_preceding=feat.get("use_preceding", True),
        use_lane=feat.get("use_lane", True),
        use_static=feat.get("use_static", True),
    )

    ds = HighDPtDataset(
        data_dir, str(split_file), stats=stats, return_meta=False,
        **train_flags,
    )

    num_workers = int(cfg["data"].get("num_workers", 2))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_batch,
        persistent_workers=(num_workers > 0),
    )

    model = TransformerStyleBaseline(**cfg["model"]).to(device)
    
    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    metrics = evaluate_single(
        model=model,
        loader=loader,
        device=device,
        use_amp=use_amp,
        predict_delta=predict_delta,
        w_traj=w_traj,
        w_fde=w_fde,
        multimodal_k=args.multimodal_k,
        vis_one=args.vis_one,
        vis_out=args.vis_out,
    )

    exp_name = cfg_path.stem
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result = EvalResult(
        exp_name=exp_name,
        split=args.split,
        ckpt_path=str(ckpt_path),
        n_samples=int(metrics["n_samples"]),
        loss=float(metrics["loss"]),
        ade=float(metrics["ade"]),
        fde=float(metrics["fde"]),
        timestamp=ts,
        extra={
            "data_dir": data_dir,
            "stats_path": str(stats_path),
            **train_flags,
            "predict_delta": predict_delta,
            "w_traj": w_traj,
            "w_fde": w_fde,
            **{k: cfg["model"].get(k) for k in ["T","Tf","K","ego_dim","nb_dim","d_model","nhead","num_layers","dropout","use_neighbors","use_slot_emb"]},
        },
    )

    # Write JSON
    out_json_dir = Path(args.out_json_dir)
    out_json_dir.mkdir(parents=True, exist_ok=True)
    out_json_path = out_json_dir / f"{exp_name}_{args.split}.json"
    out_json_path.write_text(json.dumps({
        "timestamp": result.timestamp,
        "exp_name": result.exp_name,
        "split": result.split,
        "ckpt_path": result.ckpt_path,
        "n_samples": result.n_samples,
        "loss": result.loss,
        "ade": result.ade,
        "fde": result.fde,
        **result.extra,
    }, indent=2))

    # Append CSV
    row = {
        "timestamp": result.timestamp,
        "exp_name": result.exp_name,
        "split": result.split,
        "ckpt_path": result.ckpt_path,
        "n_samples": result.n_samples,
        "loss": result.loss,
        "ade": result.ade,
        "fde": result.fde,
        **result.extra,
    }
    _append_csv(Path(args.out_csv), row)

    print("==== Eval Result ====")
    print(f"exp_name: {result.exp_name}")
    print(f"split:    {result.split}")
    print(f"ckpt:     {result.ckpt_path}")
    print(f"samples:  {result.n_samples}")
    print(f"loss:     {result.loss:.4f}")
    print(f"ADE:      {result.ade:.4f}")
    print(f"FDE:      {result.fde:.4f}")
    print(f"saved:    {out_json_path}")
    print(f"append:   {args.out_csv}")


if __name__ == "__main__":
    main()
