# tp_baseline/src/losses.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_loss(name: str = "smoothl1"):
    if name.lower() == "mse":
        return nn.MSELoss()
    return nn.SmoothL1Loss()

def build_gt_delta(y_abs: torch.Tensor, x_last_abs: torch.Tensor) -> torch.Tensor:
    """
    y_abs: (B,Tf,2)
    x_last_abs: (B,2)
    returns y_delta: (B,Tf,2)
    delta[0] = y[0] - x_last
    delta[t] = y[t] - y[t-1]
    """
    B, Tf, _ = y_abs.shape
    prev = torch.cat([x_last_abs.unsqueeze(1), y_abs[:, :-1, :]], dim=1)  # (B,Tf,2)
    return y_abs - prev

def delta_to_abs(pred_delta: torch.Tensor, x_last_abs: torch.Tensor) -> torch.Tensor:
    """
    pred_delta: (B,Tf,2)
    x_last_abs: (B,2)
    """
    return torch.cumsum(pred_delta, dim=1) + x_last_abs.unsqueeze(1)

def trajectory_loss(
    pred: torch.Tensor,
    y_abs: torch.Tensor,
    x_last_abs: torch.Tensor,
    predict_delta: bool,
    w_traj: float = 1.0,
    w_fde: float = 0.0,
) -> torch.Tensor:
    """
    If predict_delta:
      - loss on delta space (SmoothL1 over all steps)
      - optional last-step loss on absolute (FDE-weighted)
    Else:
      - loss on absolute space
      - optional last-step loss on absolute
    """
    if predict_delta:
        y_delta = build_gt_delta(y_abs, x_last_abs)
        loss_traj = F.smooth_l1_loss(pred, y_delta)
        if w_fde > 0:
            pred_abs = delta_to_abs(pred, x_last_abs)
            loss_last = F.smooth_l1_loss(pred_abs[:, -1, :], y_abs[:, -1, :])
            return w_traj * loss_traj + w_fde * loss_last
        return w_traj * loss_traj

    # absolute prediction
    loss_traj = F.smooth_l1_loss(pred, y_abs)
    if w_fde > 0:
        loss_last = F.smooth_l1_loss(pred[:, -1, :], y_abs[:, -1, :])
        return w_traj * loss_traj + w_fde * loss_last
    return w_traj * loss_traj

def _best_mode_by_minade(pred_abs_all: torch.Tensor, y_abs: torch.Tensor) -> torch.Tensor:
    """
    pred_abs_all: (B,M,Tf,2) absolute
    y_abs:        (B,Tf,2)
    return: best_idx (B,)
    """
    err = torch.norm(pred_abs_all - y_abs[:, None, :, :], dim=-1)  # (B,M,Tf)
    ade_bm = err.mean(dim=-1)                                      # (B,M)
    return ade_bm.argmin(dim=1)


def multimodal_loss(
    pred: torch.Tensor,              # (B,M,Tf,2) delta or abs
    y_abs: torch.Tensor,             # (B,Tf,2)
    x_last_abs: torch.Tensor,        # (B,2)
    predict_delta: bool,
    score_logits: torch.Tensor | None = None,   # (B,M)
    w_traj: float = 1.0,
    w_fde: float = 0.0,
    w_cls: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (loss, best_idx).
    - best-of-M regression using minADE in absolute space
    - optional classification loss if score_logits provided and w_cls>0
    """
    assert pred.dim() == 4, f"Expected pred (B,M,Tf,2), got {pred.shape}"
    B, M, Tf, _ = pred.shape

    # Convert to absolute for selecting best mode
    if predict_delta:
        pred_abs_all = torch.cumsum(pred, dim=2) + x_last_abs[:, None, None, :]
    else:
        pred_abs_all = pred

    best_idx = _best_mode_by_minade(pred_abs_all, y_abs)  # (B,)

    # pick best mode trajectory (still in same space as pred)
    best_pred = pred[torch.arange(B, device=pred.device), best_idx]  # (B,Tf,2)

    # reuse existing single-mode trajectory_loss
    loss_reg = trajectory_loss(
        pred=best_pred,
        y_abs=y_abs,
        x_last_abs=x_last_abs,
        predict_delta=predict_delta,
        w_traj=w_traj,
        w_fde=w_fde,
    )

    if (score_logits is None) or (w_cls <= 0.0):
        return loss_reg, best_idx

    loss_cls = F.cross_entropy(score_logits, best_idx)
    return loss_reg + w_cls * loss_cls, best_idx
