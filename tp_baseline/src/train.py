# tp_baseline/src/train.py
"""
Train script for highD Transformer baseline.

Recommended run (from tp_baseline/):
  python3 -m src.train --config configs/baseline.yaml

If you run as a file:
  PYTHONPATH=. python3 src/train.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from src.datasets.highd_pt_dataset import HighDPtDataset
from src.datasets.collate import collate_batch
from src.losses import trajectory_loss, delta_to_abs, multimodal_loss
from src.utils import set_seed, load_stats_npz, build_model
from src.metrics import ade, fde


def _resolve_path(base: Path, p: str) -> Path:
    """Resolve relative paths against config file directory."""
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    sched_type: str,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Warmup + (optional) cosine decay.
    Returns LambdaLR that should be stepped EVERY optimizer step.
    """
    sched_type = (sched_type or "none").lower()

    def lr_lambda(step: int) -> float:
        # step starts at 0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if sched_type == "cosine":
            # cosine from 1.0 -> 0.0 after warmup
            denom = max(1, total_steps - warmup_steps)
            progress = (step - warmup_steps) / float(denom)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

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
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="val", leave=False)
    for batch in pbar:
        x_ego = batch["x_ego"].to(device, non_blocking=True)
        x_nb = batch["x_nb"].to(device, non_blocking=True)
        nb_mask = batch["nb_mask"].to(device, non_blocking=True)
        style_prob = batch.get("style_prob", None)
        style_valid = batch.get("style_valid", None)
        if style_prob is not None: style_prob = style_prob.to(device, non_blocking=True)
        if style_valid is not None: style_valid = style_valid.to(device, non_blocking=True)
        y_abs = batch["y"].to(device, non_blocking=True)
        x_last_abs = batch["x_last_abs"].to(device, non_blocking=True)  # (B,2)

        with autocast(device_type="cuda", enabled=use_amp):
            out = model(x_ego, x_nb, nb_mask, style_prob=style_prob, style_valid=style_valid)

        # out can be:
        #  - pred (B,Tf,2)  [baseline/style]
        #  - pred (B,M,Tf,2)  [wayformer without scores]
        #  - (pred, scores)   [wayformer with scores]
        if isinstance(out, (tuple, list)):
            pred, scores = out
        else:
            pred, scores = out, None

        with autocast(device_type="cuda", enabled=use_amp):
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

        # metrics
        a = ade(pred_abs, y_abs)
        f = fde(pred_abs, y_abs)

        total_loss += float(loss.item())
        total_ade += float(a.item())
        total_fde += float(f.item())
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "ADE": f"{a.item():.3f}",
            "FDE": f"{f.item():.3f}",
        })

    if n_batches == 0:
        return {"loss": float("nan"), "ade": float("nan"), "fde": float("nan")}

    return {
        "loss": total_loss / n_batches,
        "ade": total_ade / n_batches,
        "fde": total_fde / n_batches,
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    device: torch.device,
    scaler: Optional[GradScaler],
    use_amp: bool,
    grad_clip_norm: float,
    global_step_start: int,
    predict_delta: bool,
    w_traj: float,
    w_fde: float,
    w_cls: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    n_batches = 0

    global_step = global_step_start

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        x_ego = batch["x_ego"].to(device, non_blocking=True)
        x_nb = batch["x_nb"].to(device, non_blocking=True)
        nb_mask = batch["nb_mask"].to(device, non_blocking=True)

        style_prob = batch.get("style_prob", None)
        style_valid = batch.get("style_valid", None)
        if style_prob is not None:
            style_prob = style_prob.to(device, non_blocking=True)
        if style_valid is not None:
            style_valid = style_valid.to(device, non_blocking=True)

        y_abs = batch["y"].to(device, non_blocking=True)
        x_last_abs = batch["x_last_abs"].to(device, non_blocking=True)  # (B,2)

        optimizer.zero_grad(set_to_none=True)

        # ---- forward + loss ----
        with autocast(device_type="cuda", enabled=use_amp):
            out = model(x_ego, x_nb, nb_mask, style_prob=style_prob, style_valid=style_valid)

            # out can be:
            #  - pred (B,Tf,2)  [baseline/style]
            #  - pred (B,M,Tf,2)  [wayformer without scores]
            #  - (pred, scores)   [wayformer with scores]
            if isinstance(out, (tuple, list)):
                pred, scores = out
            else:
                pred, scores = out, None

            if pred.dim() == 4:
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
            else:
                loss = trajectory_loss(
                    pred=pred,
                    y_abs=y_abs,
                    x_last_abs=x_last_abs,
                    predict_delta=predict_delta,
                    w_traj=w_traj,
                    w_fde=w_fde,
                )
                best_idx = None  # for clarity

        # ---- backward/step (unchanged pattern) ----
        if use_amp:
            assert scaler is not None
            scaler.scale(loss).backward()

            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        # Scheduler step PER ITERATION
        if scheduler is not None:
            scheduler.step()

        # ---- metrics (absolute space) ----
        with torch.no_grad():
            if pred.dim() == 4:
                # metrics: best mode in ABS space
                if predict_delta:
                    pred_abs_all = torch.cumsum(pred, dim=2) + x_last_abs[:, None, None, :]
                else:
                    pred_abs_all = pred
                pred_abs = pred_abs_all[torch.arange(pred.shape[0], device=pred.device), best_idx]
            else:
                pred_abs = delta_to_abs(pred, x_last_abs) if predict_delta else pred

            a = ade(pred_abs, y_abs)
            f = fde(pred_abs, y_abs)

            total_loss += float(loss.item())
            total_ade += float(a.item())
            total_fde += float(f.item())
            n_batches += 1

        lr_now = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "ADE": f"{a.item():.3f}",
            "FDE": f"{f.item():.3f}",
            "lr": f"{lr_now:.2e}",
        })

        global_step += 1

    return {
        "loss": total_loss / max(1, n_batches),
        "ade": total_ade / max(1, n_batches),
        "fde": total_fde / max(1, n_batches),
        "global_step_end": global_step,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg_dir = cfg_path.parent
    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text())

    # ---- seed ----
    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)

    # ---- device ----
    want_cuda = (cfg.get("train", {}).get("device", "cuda") == "cuda")
    device = torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")

    print("==== Environment ====")
    print("torch:", torch.__version__)
    print("device:", device)
    if device.type == "cuda":
        print("cuda:", torch.version.cuda)
        print("gpu:", torch.cuda.get_device_name(0))
        print("cudnn:", torch.backends.cudnn.version())
        torch.backends.cudnn.benchmark = True

    # ---- data paths ----
    data_dir = str(_resolve_path(cfg_dir, cfg["data"]["data_dir"]))
    splits_dir = _resolve_path(cfg_dir, cfg["data"]["splits_dir"])
    stats_path = _resolve_path(cfg_dir, cfg["data"]["stats_path"])

    train_split = str(splits_dir / "train.txt")
    val_split = str(splits_dir / "val.txt")

    # ---- load stats ----
    if stats_path.exists():
        stats = load_stats_npz(str(stats_path))
    else:
        raise FileNotFoundError(
            f"Stats file not found: {stats_path}\n"
            f"Run compute_stats first to create it."
        )

    # ---- datasets / loaders ----
    batch_size = int(cfg["data"].get("batch_size", 128))
    num_workers = int(cfg["data"].get("num_workers", 2))

    feat = cfg.get("features", {})

    train_ds = HighDPtDataset(
        data_dir, train_split, stats=stats, return_meta=False,
        use_neighbors=feat.get("use_neighbors", True),
        use_context=feat.get("use_context", True),
        use_safety=feat.get("use_safety", True),
        use_preceding=feat.get("use_preceding", True),
        use_lane=feat.get("use_lane", True),
        use_static=feat.get("use_static", True),
    )
    val_ds = HighDPtDataset(
        data_dir, val_split, stats=stats, return_meta=False,
        use_neighbors=feat.get("use_neighbors", True),
        use_context=feat.get("use_context", True),
        use_safety=feat.get("use_safety", True),
        use_preceding=feat.get("use_preceding", True),
        use_lane=feat.get("use_lane", True),
        use_static=feat.get("use_static", True),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate_batch,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_batch,
        persistent_workers=(num_workers > 0),
    )

    # ---- model ----
    model = build_model(cfg).to(device)

    # ---- optimizer ----
    lr = float(cfg["train"].get("lr", 3e-4))
    weight_decay = float(cfg["train"].get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- AMP ----
    use_amp = bool(cfg["train"].get("use_amp", True)) and (device.type == "cuda")
    scaler = GradScaler('cuda', enabled=use_amp)

    # ---- Stage1/2 flags ----
    predict_delta = bool(cfg.get("model", {}).get("predict_delta", False))
    w_traj = float(cfg.get("train", {}).get("w_traj", 1.0))
    w_fde = float(cfg.get("train", {}).get("w_fde", 0.0))

    # ---- training settings ----
    epochs = int(cfg["train"].get("epochs", 50))
    grad_clip_norm = float(cfg["train"].get("grad_clip_norm", 1.0))

    # scheduler settings (Stage2)
    warmup_steps = int(cfg["train"].get("warmup_steps", 0))
    lr_schedule = str(cfg["train"].get("lr_schedule", "none"))
    total_steps = epochs * len(train_loader)
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        sched_type=lr_schedule,
    )

    ckpt_dir = _resolve_path(cfg_dir, cfg["train"].get("ckpt_dir", "ckpts"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_ade = float("inf")
    global_step = 0

    print("==== Train Config ====")
    print(f"data_dir: {data_dir}")
    print(f"train samples: {len(train_ds)} | val samples: {len(val_ds)}")
    print(f"batch_size: {batch_size} | num_workers: {num_workers}")
    print(f"use_amp: {use_amp} | grad_clip_norm: {grad_clip_norm}")
    print(f"predict_delta: {predict_delta} | w_traj: {w_traj} | w_fde: {w_fde}")
    print(f"warmup_steps: {warmup_steps} | lr_schedule: {lr_schedule} | total_steps: {total_steps}")
    print(f"ckpt_dir: {ckpt_dir}")
    print("======================")

    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch:03d}/{epochs}]")

        w_cls = float(cfg["train"].get("w_cls", 0.0))
        
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip_norm=grad_clip_norm,
            global_step_start=global_step,
            predict_delta=predict_delta,
            w_traj=w_traj,
            w_fde=w_fde,
            w_cls=w_cls,
        )
        global_step = int(train_metrics["global_step_end"])

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            use_amp=use_amp,
            predict_delta=predict_delta,
            w_traj=w_traj,
            w_fde=w_fde,
            w_cls=w_cls,
        )

        print(
            f"  train | loss {train_metrics['loss']:.4f} | ADE {train_metrics['ade']:.4f} | FDE {train_metrics['fde']:.4f}\n"
            f"  val   | loss {val_metrics['loss']:.4f} | ADE {val_metrics['ade']:.4f} | FDE {val_metrics['fde']:.4f}"
        )

        # ---- checkpoint (best by val ADE) ----
        is_best = val_metrics["ade"] < best_val_ade
        if is_best:
            best_val_ade = val_metrics["ade"]
            ckpt_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if use_amp else None,
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "cfg": cfg,
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_ade": best_val_ade,
                },
                ckpt_path,
            )
            print(f"  ✅ saved best checkpoint: {ckpt_path} (best_val_ade={best_val_ade:.4f})")

        # ---- save last ----
        if bool(cfg["train"].get("save_last", True)):
            last_path = ckpt_dir / "last.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if use_amp else None,
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "cfg": cfg,
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_ade": best_val_ade,
                },
                last_path,
            )

    print("\nDone.")
    print("Best val ADE:", best_val_ade)


if __name__ == "__main__":
    main()
