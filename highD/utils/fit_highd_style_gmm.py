#!/usr/bin/env python3
"""
Fit a global driving-style GMM from preprocessed highD window NPZs.

Input:
  - A directory containing NPZ files produced by raw_to_npz.py (e.g., highd_01.npz ...)

Output:
  - style_model.joblib  (scaler + gmm + metadata)
  - style_labels.npz    (per-window labels keyed by recordingId/trackId/t0_frame)

Notes:
  - Features follow the paper's spirit (Table II): velocity std, accel stats, jerk stats, dhw/thw stats.
  - We approximate acceleration/jerk using longitudinal xAcceleration / diff(xAcceleration)/dt.
  - Optional CF filtering is provided, but defaults to "use all windows that have safety data".
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Install with: pip install scikit-learn joblib"
    ) from e

try:
    import joblib
except ImportError as e:
    raise SystemExit("joblib is required. Install with: pip install joblib") from e


# --------------------------
# Feature extraction helpers
# --------------------------

def _nan_safe_stats(x: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Return (max, min, std, mean) with NaN-safe behavior.
    If all NaN, returns (nan, nan, nan, nan).
    """
    if np.all(np.isnan(x)):
        return (np.nan, np.nan, np.nan, np.nan)
    return (
        float(np.nanmax(x)),
        float(np.nanmin(x)),
        float(np.nanstd(x)),
        float(np.nanmean(x)),
    )


def window_features(
    x_hist: np.ndarray,         # (T, 6)
    tv_safety: np.ndarray | None, # (T, 3) or None
    dt: float,
) -> np.ndarray:
    """
    Build feature vector for one window, similar to Table II.

    velocity: std(xVelocity)
    acceleration: stats(xAcceleration)
    jerk: stats(diff(xAcceleration)/dt)
    DHW: stats(dhw)
    THW: stats(thw)

    Returns: (D,) float32
    """
    # columns in x_hist: [x, y, xV, yV, xAcc, yAcc]
    xV = x_hist[:, 2].astype(np.float32)
    xAcc = x_hist[:, 4].astype(np.float32)

    vel_std = float(np.nanstd(xV))

    acc_max, acc_min, acc_std, acc_mean = _nan_safe_stats(xAcc)

    jerk = np.diff(xAcc) / dt if xAcc.shape[0] >= 2 else np.array([np.nan], dtype=np.float32)
    jerk_max, jerk_min, jerk_std, jerk_mean = _nan_safe_stats(jerk)

    if tv_safety is not None:
        dhw = tv_safety[:, 0].astype(np.float32)
        thw = tv_safety[:, 1].astype(np.float32)
        dhw_max, dhw_min, dhw_std, dhw_mean = _nan_safe_stats(dhw)
        thw_max, thw_min, thw_std, thw_mean = _nan_safe_stats(thw)
    else:
        dhw_max = dhw_min = dhw_std = dhw_mean = np.nan
        thw_max = thw_min = thw_std = thw_mean = np.nan

    feats = np.array(
        [
            vel_std,
            acc_max, acc_min, acc_std, acc_mean,
            jerk_max, jerk_min, jerk_std, jerk_mean,
            dhw_max, dhw_min, dhw_std, dhw_mean,
            thw_max, thw_min, thw_std, thw_mean,
        ],
        dtype=np.float32,
    )
    return feats


def build_feature_matrix_from_npz(
    npz_path: Path,
    dt: float,
    cf_only: bool,
    thc: float,
    tcc: float,
    require_preceding: bool,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load a single NPZ and compute per-window features.

    Returns:
      X: (M, D) float32 features for selected windows (M may be <= N)
      meta: dict arrays aligned with X rows:
         recordingId, trackId, t0_frame, sel_idx (indices into original N)
    """
    data = np.load(npz_path, allow_pickle=False)

    x_hist = data["x_hist"]            # (N, T, 6)
    recordingId = data["recordingId"]  # (N,)
    trackId = data["trackId"]          # (N,)
    t0_frame = data["t0_frame"]        # (N,)

    tv_safety = data["tv_safety"] if "tv_safety" in data.files else None
    tv_preceding = data["tv_preceding"] if "tv_preceding" in data.files else None
    # In raw_to_npz.py, tv_preceding may contain precedingXVelocity. We can optionally require it not NaN.

    N = x_hist.shape[0]

    # Select windows
    sel = np.ones((N,), dtype=bool)

    # If safety is missing, we cannot do CF-only filtering based on THW/TTC
    if cf_only:
        if tv_safety is None:
            sel[:] = False
        else:
            thw = tv_safety[:, :, 1]  # (N, T)
            ttc = tv_safety[:, :, 2]  # (N, T)

            # Use NaN-safe "any time step satisfies" heuristic
            # (This is a practical approximation; you can refine later.)
            thw_ok = np.nanmin(thw, axis=1) < thc
            ttc_ok = np.nanmin(ttc, axis=1) < tcc
            sel &= thw_ok & ttc_ok

    if require_preceding:
        if tv_preceding is None:
            sel[:] = False
        else:
            # Require at least one finite precedingXVelocity value in the history
            sel &= np.isfinite(tv_preceding).any(axis=1)

    sel_idx = np.where(sel)[0]
    if sel_idx.size == 0:
        return np.zeros((0, 17), dtype=np.float32), {
            "recordingId": np.zeros((0,), dtype=np.int32),
            "trackId": np.zeros((0,), dtype=np.int32),
            "t0_frame": np.zeros((0,), dtype=np.int32),
            "sel_idx": np.zeros((0,), dtype=np.int32),
        }

    # Compute features
    feats = np.empty((sel_idx.size, 17), dtype=np.float32)
    for i, k in enumerate(sel_idx):
        feats[i] = window_features(
            x_hist[k],
            tv_safety[k] if tv_safety is not None else None,
            dt=dt,
        )

    meta = {
        "recordingId": recordingId[sel_idx].astype(np.int32, copy=False),
        "trackId": trackId[sel_idx].astype(np.int32, copy=False),
        "t0_frame": t0_frame[sel_idx].astype(np.int32, copy=False),
        "sel_idx": sel_idx.astype(np.int32, copy=False),
    }
    return feats, meta


def concat_meta(metas: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = metas[0].keys()
    return {k: np.concatenate([m[k] for m in metas], axis=0) for k in keys}


def impute_nan_with_col_median(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace NaNs with column medians. Returns (X_filled, col_median).
    """
    X2 = X.copy()
    med = np.nanmedian(X2, axis=0)
    # If a column is all-NaN, nanmedian returns NaN; replace those with 0.0
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    inds = np.where(~np.isfinite(X2))
    X2[inds] = med[inds[1]]
    return X2.astype(np.float32), med


# --------------------------
# Main fit/predict pipeline
# --------------------------

@dataclass
class FitConfig:
    npz_dir: Path
    glob: str
    out_dir: Path
    max_k: int
    covariance_type: str
    n_init: int
    random_state: int
    use_bic: bool

    cf_only: bool
    thc: float
    tcc: float
    require_preceding: bool

    # derive dt from npz or assume
    assume_hz: float


def load_dt_from_any_npz(npz_path: Path, assume_hz: float) -> float:
    """
    Try to use target_hz stored in npz; otherwise fallback to assume_hz.
    """
    data = np.load(npz_path, allow_pickle=False)
    if "target_hz" in data.files:
        hz = float(np.array(data["target_hz"]).reshape(-1)[0])
        if hz > 0:
            return 1.0 / hz
    return 1.0 / assume_hz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", type=str, required=True, help="Directory containing highd_*.npz windows")
    ap.add_argument("--glob", type=str, default="*.npz", help="Glob pattern (default: *.npz)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for model + labels")
    ap.add_argument("--max_k", type=int, default=6, help="Max number of GMM components to try (default: 6)")
    ap.add_argument("--covariance_type", type=str, default="full", choices=["full", "diag", "tied", "spherical"])
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--random_state", type=int, default=0)
    ap.add_argument("--criterion", type=str, default="bic", choices=["bic", "aic"], help="Model selection criterion")

    # Filtering knobs
    ap.add_argument("--cf_only", action="store_true", help="Filter to (approx) car-following windows only")
    ap.add_argument("--thc", type=float, default=5.0, help="THW threshold for CF filter (default: 5s)")
    ap.add_argument("--tcc", type=float, default=1.0, help="TTC threshold for CF filter (default: 1s)")
    ap.add_argument("--require_preceding", action="store_true",
                    help="Require preceding info to exist (uses tv_preceding finite check)")

    # If target_hz not present
    ap.add_argument("--assume_hz", type=float, default=5.0, help="Fallback target_hz if missing in npz (default: 5Hz)")

    args = ap.parse_args()

    cfg = FitConfig(
        npz_dir=Path(args.npz_dir),
        glob=args.glob,
        out_dir=Path(args.out_dir),
        max_k=args.max_k,
        covariance_type=args.covariance_type,
        n_init=args.n_init,
        random_state=args.random_state,
        use_bic=(args.criterion == "bic"),
        cf_only=args.cf_only,
        thc=args.thc,
        tcc=args.tcc,
        require_preceding=args.require_preceding,
        assume_hz=args.assume_hz,
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(cfg.npz_dir.glob(cfg.glob))
    if not npz_paths:
        raise SystemExit(f"No NPZ files found in {cfg.npz_dir} with glob '{cfg.glob}'")

    dt = load_dt_from_any_npz(npz_paths[0], assume_hz=cfg.assume_hz)
    print(f"[INFO] Using dt={dt:.6f}s")

    # 1) Build global feature matrix
    all_X: List[np.ndarray] = []
    all_meta: List[Dict[str, np.ndarray]] = []

    total_windows = 0
    total_selected = 0

    for p in npz_paths:
        data = np.load(p, allow_pickle=False)
        n = int(data["x_hist"].shape[0])
        total_windows += n

        X, meta = build_feature_matrix_from_npz(
            p,
            dt=dt,
            cf_only=cfg.cf_only,
            thc=cfg.thc,
            tcc=cfg.tcc,
            require_preceding=cfg.require_preceding,
        )
        if X.shape[0] == 0:
            print(f"[WARN] {p.name}: selected=0 / total={n}")
            continue
        total_selected += X.shape[0]
        all_X.append(X)
        all_meta.append(meta)
        print(f"[INFO] {p.name}: selected={X.shape[0]} / total={n}")

    if not all_X:
        raise SystemExit("No windows selected. Try disabling --cf_only/--require_preceding or check tv_safety exists.")

    X = np.concatenate(all_X, axis=0)  # (M, D)
    meta = concat_meta(all_meta)

    print(f"[INFO] Total windows: {total_windows}, selected for fit: {total_selected}")
    print(f"[INFO] Feature matrix shape: {X.shape}")

    # 2) Impute NaNs (median)
    X_filled, col_median = impute_nan_with_col_median(X)

    # 3) Scale
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_filled)

    # 4) Fit GMMs and select by AIC/BIC
    scores: List[float] = []
    models: List[GaussianMixture] = []

    Ks = list(range(1, cfg.max_k + 1))
    for k in Ks:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=cfg.covariance_type,
            n_init=cfg.n_init,
            random_state=cfg.random_state,
        )
        gmm.fit(Xz)
        score = gmm.bic(Xz) if cfg.use_bic else gmm.aic(Xz)
        scores.append(float(score))
        models.append(gmm)
        print(f"[INFO] k={k}  {'BIC' if cfg.use_bic else 'AIC'}={score:.3f}")

    best_i = int(np.argmin(scores))
    best_k = Ks[best_i]
    best_gmm = models[best_i]
    print(f"[INFO] Selected k={best_k}")

    # 5) Predict labels + probabilities for all selected windows
    probs = best_gmm.predict_proba(Xz).astype(np.float32)  # (M, K)
    labels = probs.argmax(axis=1).astype(np.int32)         # (M,)

    # 6) Save outputs
    model_out = cfg.out_dir / "style_model.joblib"
    labels_out = cfg.out_dir / "style_labels.npz"

    joblib.dump(
        {
            "scaler": scaler,
            "gmm": best_gmm,
            "dt": dt,
            "feature_dim": int(X.shape[1]),
            "col_median": col_median,
            "criterion": "bic" if cfg.use_bic else "aic",
            "best_k": best_k,
            "scores": np.array(scores, dtype=np.float64),
            "Ks": np.array(Ks, dtype=np.int32),
            "cfg": {
                "cf_only": cfg.cf_only,
                "thc": cfg.thc,
                "tcc": cfg.tcc,
                "require_preceding": cfg.require_preceding,
                "covariance_type": cfg.covariance_type,
                "n_init": cfg.n_init,
                "random_state": cfg.random_state,
            },
        },
        model_out,
    )

    np.savez_compressed(
        labels_out,
        recordingId=meta["recordingId"],
        trackId=meta["trackId"],
        t0_frame=meta["t0_frame"],
        sel_idx=meta["sel_idx"],
        feat=X_filled.astype(np.float32),
        label=labels,
        prob=probs,
        best_k=np.array([best_k], dtype=np.int32),
    )

    print(f"[DONE] Saved model:  {model_out}")
    print(f"[DONE] Saved labels: {labels_out}")


if __name__ == "__main__":
    main()
