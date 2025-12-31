#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import joblib

import re

NEIGHBOR_COLS = [
    "precedingId",
    "followingId",
    "leftPrecedingId",
    "leftAlongsideId",
    "leftFollowingId",
    "rightPrecedingId",
    "rightAlongsideId",
    "rightFollowingId",
]

ROOT = Path(__file__).resolve().parents[1]   # highD/
DEFAULT_RAW = ROOT / "raw"
DEFAULT_OUT = ROOT / "out_style"


def _nan_safe_stats(x: np.ndarray) -> Tuple[float, float, float, float]:
    if x.size == 0 or np.all(~np.isfinite(x)):
        return (np.nan, np.nan, np.nan, np.nan)
    return (float(np.nanmax(x)), float(np.nanmin(x)), float(np.nanstd(x)), float(np.nanmean(x)))


def _impute_nan_col_median(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    X2 = X.copy()
    bad = ~np.isfinite(X2)
    X2[bad] = med[np.where(bad)[1]]
    return X2.astype(np.float32), med


def load_recording(raw_dir: Path, rec_id: str):
    rec_meta = pd.read_csv(raw_dir / f"{rec_id}_recordingMeta.csv")
    tracks_meta = pd.read_csv(raw_dir / f"{rec_id}_tracksMeta.csv")
    tracks = pd.read_csv(raw_dir / f"{rec_id}_tracks.csv")
    return rec_meta, tracks_meta, tracks


def normalize_upper_xy_like_raw_to_npz(
    tracks: pd.DataFrame,
    tracks_meta: pd.DataFrame,
    rec_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the same direction unification as your raw_to_npz.py:
      - upper lane vehicles (drivingDirection==1) are flipped:
        x' = Xmax - x, y' = C_y - y
        derivatives sign flipped
        precedingXVelocity sign flipped if exists
        laneId mirrored within upper vehicles
    """
    df = tracks.copy()

    # map vehicle id -> drivingDirection
    id_to_dd = dict(zip(tracks_meta["id"].astype(int), tracks_meta["drivingDirection"].astype(int)))

    # C_y = upperLaneMarkings[-1] (same idea as in highD; raw_to_npz.py uses last upper marking)
    # recordingMeta usually has upperLaneMarkings as string like "[...]" in highD.
    # We parse robustly:
    def _parse_list(s):
        """
        Robustly parse lane markings from recordingMeta.
        Works for formats like:
        "[0.0, 3.5, 7.0]"
        "[0.0; 3.5; 7.0]"
        "0.0 3.5 7.0"
        '"[0.0, 3.5, 7.0]"'
        """
        if s is None:
            return []
        if isinstance(s, (list, tuple, np.ndarray)):
            return [float(x) for x in s if np.isfinite(x)]
        if not isinstance(s, str):
            try:
                return [float(s)]
            except:
                return []

        # Extract all numbers (integers/floats) robustly
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
        out = []
        for n in nums:
            try:
                out.append(float(n))
            except:
                pass
        return out

    upper_marks = _parse_list(rec_meta.loc[0, "upperLaneMarkings"]) if "upperLaneMarkings" in rec_meta.columns else []
    if len(upper_marks) == 0:
        # 확실하지 않음: recordingMeta에 upperLaneMarkings가 없거나 파싱 실패한 경우
        # 이런 경우 flip을 하지 않고 진행할 수도 있지만, 여기서는 에러로 막는 게 안전.
        raise ValueError("Cannot parse upperLaneMarkings from recordingMeta. Please check recordingMeta format.")
    C_y = float(upper_marks[-1])

    # upper mask per row using id -> drivingDirection
    ids = df["id"].astype(int).to_numpy()
    dd = np.array([id_to_dd.get(int(i), 0) for i in ids], dtype=np.int32)
    upper_mask = (dd == 1)

    Xmax = float(df["x"].max())

    # flip x,y
    df.loc[upper_mask, "x"] = Xmax - df.loc[upper_mask, "x"].astype(np.float32)
    df.loc[upper_mask, "y"] = C_y - df.loc[upper_mask, "y"].astype(np.float32)

    # flip derivatives
    for col in ["xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]:
        if col in df.columns:
            df.loc[upper_mask, col] = -df.loc[upper_mask, col].astype(np.float32)

    # flip precedingXVelocity if exists
    if "precedingXVelocity" in df.columns:
        df.loc[upper_mask, "precedingXVelocity"] = -df.loc[upper_mask, "precedingXVelocity"].astype(np.float32)

    # laneId mirror (within upper only)
    if "laneId" in df.columns and upper_mask.any():
        lane_u = df.loc[upper_mask, "laneId"].astype(int)
        mn, mx = int(lane_u.min()), int(lane_u.max())
        df.loc[upper_mask, "laneId"] = (mn + mx) - lane_u

    return df


def compute_window_feature_from_rows(rows: pd.DataFrame, dt: float) -> np.ndarray:
    """
    rows: one vehicle's contiguous time slice (history window), sorted by frame
    Uses:
      velocity std: std(xVelocity)
      accel stats: max/min/std/mean(xAcceleration)
      jerk stats: stats(diff(xAcceleration)/dt)
      dhw/thw stats: stats(dhw), stats(thw) if present
    """
    xV = rows["xVelocity"].to_numpy(dtype=np.float32)
    xA = rows["xAcceleration"].to_numpy(dtype=np.float32)

    vel_std = float(np.nanstd(xV))
    acc_max, acc_min, acc_std, acc_mean = _nan_safe_stats(xA)

    if xA.size >= 2:
        jerk = np.diff(xA) / dt
    else:
        jerk = np.array([np.nan], dtype=np.float32)
    jerk_max, jerk_min, jerk_std, jerk_mean = _nan_safe_stats(jerk)

    if "dhw" in rows.columns and "thw" in rows.columns:
        dhw = rows["dhw"].to_numpy(dtype=np.float32)
        thw = rows["thw"].to_numpy(dtype=np.float32)
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


@dataclass
class StyleExtractConfig:
    raw_dir: Path
    rec_ids: List[str]
    out_dir: Path
    history_secs: List[float]
    fit_max_k: int
    criterion: str
    random_state: int
    n_init: int
    cov_type: str
    fps: float
    normalize_upper_xy: bool

    min_speed_mps: float
    require_track_len_secs: float


def extract_features_all(
    cfg: StyleExtractConfig,
) -> Dict[float, Dict[str, np.ndarray]]:
    """
    For each history length L, output dict with:
      feat: (M, D)
      rec_id: (M,)
      veh_id: (M,)
      t0_frame: (M,)
    """
    out: Dict[float, Dict[str, List[np.ndarray]]] = {L: {"feat": [], "rec": [], "veh": [], "t0": []} for L in cfg.history_secs}
    dt = 1.0 / cfg.fps

    for rec in cfg.rec_ids:
        rec_meta, tracks_meta, tracks = load_recording(cfg.raw_dir, rec)

        # filters at track-level
        tm = tracks_meta.copy()
        if "meanXVelocity" in tm.columns:
            tm = tm[tm["meanXVelocity"] >= cfg.min_speed_mps]
        if "finalFrame" in tm.columns and "initialFrame" in tm.columns:
            dur = (tm["finalFrame"] - tm["initialFrame"] + 1) / cfg.fps
            tm = tm[dur >= cfg.require_track_len_secs]
        valid_ids = set(tm["id"].astype(int).tolist())

        tr = tracks[tracks["id"].astype(int).isin(valid_ids)].copy()

        if cfg.normalize_upper_xy:
            tr = normalize_upper_xy_like_raw_to_npz(tr, tracks_meta, rec_meta)

        # ensure sorted
        tr.sort_values(["id", "frame"], inplace=True)

        # group by vehicle
        for vid, g in tr.groupby("id", sort=False):
            g = g.sort_values("frame")
            frames = g["frame"].to_numpy(dtype=np.int32)

            for L in cfg.history_secs:
                T = int(round(L * cfg.fps))
                if T < 2:
                    continue

                # sliding t0: we define t0 as the LAST frame index of history window
                # (i.e., window rows correspond to frames [t0-T+1 ... t0])
                if len(g) < T:
                    continue

                # to keep it comparable to your TP sampling, you can subsample t0 stride (e.g., every 1 frame or every N frames)
                # here: every 1 frame (max detail); you can set stride later if too big.
                for end in range(T - 1, len(g)):
                    t0_frame = int(frames[end])
                    hist_rows = g.iloc[end - T + 1 : end + 1]
                    feat = compute_window_feature_from_rows(hist_rows, dt=dt)

                    out[L]["feat"].append(feat[None, :])
                    out[L]["rec"].append(np.array([int(rec)], dtype=np.int32))
                    out[L]["veh"].append(np.array([int(vid)], dtype=np.int32))
                    out[L]["t0"].append(np.array([t0_frame], dtype=np.int32))

    # concat
    packed: Dict[float, Dict[str, np.ndarray]] = {}
    for L in cfg.history_secs:
        if len(out[L]["feat"]) == 0:
            packed[L] = {"feat": np.zeros((0, 17), np.float32),
                         "rec_id": np.zeros((0,), np.int32),
                         "veh_id": np.zeros((0,), np.int32),
                         "t0_frame": np.zeros((0,), np.int32)}
            continue
        packed[L] = {
            "feat": np.concatenate(out[L]["feat"], axis=0).astype(np.float32),
            "rec_id": np.concatenate(out[L]["rec"], axis=0).astype(np.int32),
            "veh_id": np.concatenate(out[L]["veh"], axis=0).astype(np.int32),
            "t0_frame": np.concatenate(out[L]["t0"], axis=0).astype(np.int32),
        }
    return packed


def fit_gmm_select_k(Xz: np.ndarray, max_k: int, criterion: str, cov_type: str, n_init: int, random_state: int):
    models = []
    scores = []
    Ks = list(range(1, max_k + 1))
    for k in Ks:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=cov_type,
            n_init=n_init,
            random_state=random_state,
            reg_covar=1e-4,           
            init_params="kmeans",       
            max_iter=300, 
        )
        gmm.fit(Xz)
        score = gmm.bic(Xz) if criterion == "bic" else gmm.aic(Xz)
        models.append(gmm)
        scores.append(float(score))
    best_i = int(np.argmin(scores))
    return Ks[best_i], models[best_i], np.array(Ks, dtype=np.int32), np.array(scores, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default=str(DEFAULT_RAW))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--rec_ids", required=True, type=str, nargs="+", help="e.g., 01 02 03")

    ap.add_argument("--history_secs", type=float, nargs="+", default=[2.0, 3.0, 5.0])
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--normalize_upper_xy", action="store_true")

    ap.add_argument("--min_speed_mps", type=float, default=15.0)
    ap.add_argument("--require_track_len_secs", type=float, default=8.0)

    ap.add_argument("--max_k", type=int, default=6)
    ap.add_argument("--criterion", type=str, choices=["bic", "aic"], default="bic")
    ap.add_argument("--cov_type", type=str, choices=["full", "diag", "tied", "spherical"], default="full")
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--random_state", type=int, default=0)

    args = ap.parse_args()

    cfg = StyleExtractConfig(
        raw_dir=Path(args.raw_dir),
        rec_ids=[str(r).zfill(2) for r in args.rec_ids],
        out_dir=Path(args.out_dir),
        history_secs=list(args.history_secs),
        fit_max_k=args.max_k,
        criterion=args.criterion,
        random_state=args.random_state,
        n_init=args.n_init,
        cov_type=args.cov_type,
        fps=args.fps,
        normalize_upper_xy=bool(args.normalize_upper_xy),
        min_speed_mps=args.min_speed_mps,
        require_track_len_secs=args.require_track_len_secs,
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    packed = extract_features_all(cfg)
    for L in cfg.history_secs:
        X = packed[L]["feat"]
        print(f"[INFO] L={L}s features: {X.shape}")
        if X.shape[0] == 0:
            continue

        X_filled, col_med = _impute_nan_col_median(X)
        X_filled64 = X_filled.astype(np.float64, copy=False)
        scaler = StandardScaler()
        Xz = scaler.fit_transform(X_filled64)

        best_k, gmm, Ks, scores = fit_gmm_select_k(
            Xz, max_k=cfg.fit_max_k, criterion=cfg.criterion, cov_type=cfg.cov_type,
            n_init=cfg.n_init, random_state=cfg.random_state
        )
        probs = gmm.predict_proba(Xz).astype(np.float32)
        labels = probs.argmax(axis=1).astype(np.int32)

        # save per-L artifacts
        out_npz = cfg.out_dir / f"style_windows_L{L:.1f}s.npz"
        np.savez_compressed(
            out_npz,
            history_sec=np.array([L], np.float32),
            fps=np.array([cfg.fps], np.float32),
            feat=X_filled.astype(np.float32),
            label=labels,
            prob=probs,
            rec_id=packed[L]["rec_id"],
            veh_id=packed[L]["veh_id"],
            t0_frame=packed[L]["t0_frame"],
            best_k=np.array([best_k], np.int32),
            Ks=Ks,
            scores=scores,
            col_median=col_med.astype(np.float32),
        )

        out_model = cfg.out_dir / f"style_model_L{L:.1f}s.joblib"
        joblib.dump(
            {"scaler": scaler, "gmm": gmm, "best_k": best_k, "col_median": col_med, "fps": cfg.fps, "history_sec": L},
            out_model,
        )

        print(f"[DONE] Saved: {out_npz.name}, {out_model.name} (k={best_k})")


if __name__ == "__main__":
    main()
