#!/usr/bin/env python3
import re
from pathlib import Path
import numpy as np
import pandas as pd
import multiprocessing as mp

"""
highD CSV -> NPZ preprocessing (FASTER CPU version)

What this version does (vs. the slow pandas-.loc version):
- Still reads CSV with pandas (fast C parser), but immediately converts to NumPy arrays.
- Avoids per-step pandas MultiIndex lookups.
- Uses per-vehicle frame->row-index maps for O(1) neighbor row access.
- Processes recordings in parallel (multiprocessing) if --num_workers > 1.
- Does NOT do chunked saving (as requested): each recording writes one .npz.

Coordinate normalization:
- Bounding box upper-left -> center using tracksMeta width/height.
- If --normalize_upper_xy:
  * For upper-lane vehicles (drivingDirection==1): x' = Xmax - x, y' = C_y - y
  * Flip x/y velocities and accelerations sign.
  * Flip precedingXVelocity sign (if present).
  * laneId mirrored for upper-lane rows only:
      new_laneId = (min_lane_upper + max_lane_upper) - old_laneId
- frontSightDistance/backSightDistance are kept AS-IS (no swapping).

Neighbor slots:
- Uses the 8 highD neighbor IDs by default.
- If --visible_only: keeps only [precedingId, leftPrecedingId, leftAlongsideId, rightPrecedingId, rightAlongsideId]
  (indices [0,2,3,5,6]) and masks others.

Output (per recording):
- x_hist: (N,T,6)  [x,y,xV,yV,xA,yA]
- y_fut : (N,Tf,2) [x,y]
- nb_hist: (N,T,8,6) relative to ego at each t (zeros where missing)
- nb_mask: (N,T,8) bool
- nb_id  : (N,T,8) int32 neighbor ids per slot per timestep (0 if none)
- tv_static: (N,3)  [width,height,classId] classId: Car=1, Truck=2, else 0
- tv_context: (N,T,2) [frontSightDistance, backSightDistance] (zeros if missing)
- tv_safety : (N,T,3) [dhw, thw, ttc] (zeros if missing)
- tv_preceding: (N,T,1) [precedingXVelocity] (zeros if missing)
- tv_lane  : (N,T) int16 laneId (zeros if missing)
- nb_static: (N,8,3) [width,height,classId] from t0 neighbor ids (zeros if missing)
- metadata arrays: recordingId, trackId, t0_frame, drivingDirection, etc.

Folder scheme:
  ./data/highd_T{T}_Tf{Tf}_hz{hz}_flipXY|noflip_vis|all/highd_{rec_id}.npz
"""

import argparse
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import multiprocessing as mp

NEIGHBOR_COLS_8 = [
    "precedingId",
    "followingId",
    "leftPrecedingId",
    "leftAlongsideId",
    "leftFollowingId",
    "rightPrecedingId",
    "rightAlongsideId",
    "rightFollowingId",
]

VISIBLE_KEEP_IDXS = [0, 2, 3, 5, 6]


def map_class_to_id(cls) -> int:
    if isinstance(cls, str):
        c = cls.strip().lower()
        if c == "car":
            return 1
        if c == "truck":
            return 2
    return 0


def parse_semicolon_floats(s: str) -> List[float]:
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s:
        return []
    out = []
    for p in s.split(";"):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except ValueError:
            pass
    return out


def find_recording_ids(raw_dir: Path) -> List[str]:
    ids = []
    for p in raw_dir.glob("*_tracks.csv"):
        m = re.match(r"(\d+)_tracks\.csv$", p.name)
        if m:
            ids.append(m.group(1))
    return sorted(set(ids))


@dataclass
class Config:
    raw_dir: Path
    out_dir: Path
    target_hz: float
    history_sec: float
    future_sec: float
    stride_sec: float
    min_speed_mps: float
    visible_only: bool
    normalize_upper_xy: bool


def load_recording(raw_dir: Path, rec_id: str):
    rec_meta_path = raw_dir / f"{rec_id}_recordingMeta.csv"
    tracks_meta_path = raw_dir / f"{rec_id}_tracksMeta.csv"
    tracks_path = raw_dir / f"{rec_id}_tracks.csv"
    recording_meta = pd.read_csv(rec_meta_path)
    tracks_meta = pd.read_csv(tracks_meta_path)
    tracks = pd.read_csv(tracks_path)
    return recording_meta, tracks_meta, tracks


def compute_downsample_step(frame_rate: float, target_hz: float) -> int:
    step = int(round(frame_rate / target_hz))
    if step <= 0:
        raise ValueError(f"Invalid downsample step: frame_rate={frame_rate}, target_hz={target_hz}")
    return step


def build_meta_arrays(tracks_meta: pd.DataFrame) -> Tuple[Dict[int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      id_to_idx: vehicleId -> meta row idx
      w, h: float32 arrays per meta idx
      cls_id: int16 array per meta idx (Car=1, Truck=2)
      dd: int8 array per meta idx (drivingDirection)
    """
    for c in ["id", "width", "height", "class", "drivingDirection"]:
        if c not in tracks_meta.columns:
            raise ValueError(f"tracksMeta missing column: {c}. columns={list(tracks_meta.columns)}")

    ids = tracks_meta["id"].astype(int).to_numpy()
    w = tracks_meta["width"].astype(np.float32).to_numpy()
    h = tracks_meta["height"].astype(np.float32).to_numpy()
    cls_id = tracks_meta["class"].map(map_class_to_id).astype(np.int16).to_numpy()
    dd = tracks_meta["drivingDirection"].astype(np.int8).to_numpy()

    id_to_idx = {int(vid): i for i, vid in enumerate(ids)}
    return id_to_idx, w, h, cls_id, dd


def flip_constants(recording_meta: pd.DataFrame) -> Tuple[float, float]:
    if "frameRate" not in recording_meta.columns:
        raise ValueError("recordingMeta missing frameRate")
    frame_rate = float(recording_meta.loc[0, "frameRate"])

    upper = parse_semicolon_floats(str(recording_meta.loc[0, "upperLaneMarkings"])) if "upperLaneMarkings" in recording_meta.columns else []
    lower = parse_semicolon_floats(str(recording_meta.loc[0, "lowerLaneMarkings"])) if "lowerLaneMarkings" in recording_meta.columns else []

    if len(upper) == 0 or len(lower) == 0:
        C_y = 0.0
    else:
        C_y = float(upper[-1] + lower[0])
    return C_y, frame_rate


def normalize_tracks_numpy(
    tracks: pd.DataFrame,
    id_to_meta: Dict[int, int],
    meta_w: np.ndarray,
    meta_h: np.ndarray,
    meta_dd: np.ndarray,
    C_y: float,
    normalize_upper_xy: bool,
) -> pd.DataFrame:
    """
    Returns a (still) DataFrame but with numeric columns normalized.
    We keep DataFrame briefly for easy downsample & column selection, then convert to NumPy.
    """
    need = ["frame", "id", "x", "y", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"] + NEIGHBOR_COLS_8
    missing = [c for c in need if c not in tracks.columns]
    if missing:
        raise ValueError(f"tracks missing columns: {missing}")

    df = tracks.copy()
    df["id"] = df["id"].astype(int)
    df["frame"] = df["frame"].astype(int)

    # width/height from meta (constant per vehicle)
    meta_idx = df["id"].map(id_to_meta).to_numpy()
    # meta_idx can contain NaN -> object; handle unknown ids
    # unknown rows: leave widths as df['width']/df['height'] if exist, else 0
    if "width" in df.columns:
        w_fallback = df["width"].astype(np.float32).to_numpy()
    else:
        w_fallback = np.zeros(len(df), np.float32)
    if "height" in df.columns:
        h_fallback = df["height"].astype(np.float32).to_numpy()
    else:
        h_fallback = np.zeros(len(df), np.float32)

    w = np.where(pd.isna(meta_idx), w_fallback, meta_w[meta_idx.astype(int)])
    h = np.where(pd.isna(meta_idx), h_fallback, meta_h[meta_idx.astype(int)])
    dd = np.where(pd.isna(meta_idx), 0, meta_dd[meta_idx.astype(int)]).astype(np.int8)

    # bbox upper-left -> center
    df["x"] = df["x"].astype(np.float32) + (w / 2.0)
    df["y"] = df["y"].astype(np.float32) + (h / 2.0)

    if not normalize_upper_xy:
        return df

    upper_mask = (dd == 1)
    if not np.any(upper_mask):
        return df

    Xmax = float(df["x"].max())
    # flip x,y for upper
    df.loc[upper_mask, "x"] = Xmax - df.loc[upper_mask, "x"].astype(np.float32)
    df.loc[upper_mask, "y"] = float(C_y) - df.loc[upper_mask, "y"].astype(np.float32)

    # flip derivatives
    for col in ["xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]:
        df.loc[upper_mask, col] = -df.loc[upper_mask, col].astype(np.float32)

    # flip precedingXVelocity if exists
    if "precedingXVelocity" in df.columns:
        df.loc[upper_mask, "precedingXVelocity"] = -df.loc[upper_mask, "precedingXVelocity"].astype(np.float32)

    # laneId manual mirror for upper only (min+max-old)
    if "laneId" in df.columns:
        lane_u = df.loc[upper_mask, "laneId"].astype(int)
        if len(lane_u) > 0:
            mn, mx = int(lane_u.min()), int(lane_u.max())
            df.loc[upper_mask, "laneId"] = (mn + mx) - lane_u

    return df


def build_vehicle_maps(df_ds: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, Dict[int, int]]]:
    """
    Convert downsampled df to numpy and build:
      - ids_unique: list of vehicle IDs present
      - veh_rows: dict vid -> row indices in global arrays (sorted by frame)
      - veh_frame_to_pos: dict vid -> {frame -> pos_in_veh_rows}
    """
    df_ds = df_ds.sort_values(["id", "frame"])
    ids = df_ds["id"].to_numpy(dtype=np.int32)
    frames = df_ds["frame"].to_numpy(dtype=np.int32)

    # group by id
    uniq_ids, start_idx, counts = np.unique(ids, return_index=True, return_counts=True)
    veh_rows: Dict[int, np.ndarray] = {}
    veh_frame_to_pos: Dict[int, Dict[int, int]] = {}

    for vid, st, ct in zip(uniq_ids.tolist(), start_idx.tolist(), counts.tolist()):
        idxs = np.arange(st, st + ct, dtype=np.int32)
        veh_rows[int(vid)] = idxs
        # frame -> position in idxs
        f = frames[idxs]
        # frames can repeat? typically no; guard by last write
        d = {int(fr): int(i) for i, fr in enumerate(f.tolist())}
        veh_frame_to_pos[int(vid)] = d

    return uniq_ids.astype(np.int32), veh_rows, veh_frame_to_pos


def make_windows_for_recording(rec_id: str, cfg: Config) -> Tuple[str, int, Optional[str]]:
    """
    Returns (rec_id, num_samples, error_message_if_any)
    """
    try:
        recording_meta, tracks_meta, tracks = load_recording(cfg.raw_dir, rec_id)
        id_to_meta, mw, mh, mcls, mdd = build_meta_arrays(tracks_meta)
        C_y, frame_rate = flip_constants(recording_meta)
        ds_step = compute_downsample_step(frame_rate, cfg.target_hz)

        T = int(round(cfg.history_sec * cfg.target_hz))
        Tf = int(round(cfg.future_sec * cfg.target_hz))
        stride = max(1, int(round(cfg.stride_sec * cfg.target_hz)))

        # normalize (still DataFrame)
        tracks_norm = normalize_tracks_numpy(
            tracks=tracks,
            id_to_meta=id_to_meta,
            meta_w=mw,
            meta_h=mh,
            meta_dd=mdd,
            C_y=C_y,
            normalize_upper_xy=cfg.normalize_upper_xy,
        )

        # downsample
        df_ds = tracks_norm[((tracks_norm["frame"] - 1) % ds_step) == 0].copy()
        if df_ds.empty:
            return rec_id, 0, "Downsample produced no rows."

        # keep only needed columns for speed
        base6 = ["x", "y", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]
        optional_cols = []
        for c in ["frontSightDistance", "backSightDistance", "dhw", "thw", "ttc", "precedingXVelocity", "laneId"]:
            if c in df_ds.columns:
                optional_cols.append(c)

        keep_cols = ["id", "frame"] + base6 + NEIGHBOR_COLS_8 + optional_cols
        df_ds = df_ds[keep_cols]

        # global arrays
        ids = df_ds["id"].to_numpy(np.int32)
        frames = df_ds["frame"].to_numpy(np.int32)
        feats6 = df_ds[base6].to_numpy(np.float32)

        # neighbors: (M,8) int32 (handle NaN)
        nb_ids = df_ds[NEIGHBOR_COLS_8].fillna(0).astype(np.int32).to_numpy()

        # optional arrays
        has_ctx = ("frontSightDistance" in df_ds.columns) and ("backSightDistance" in df_ds.columns)
        ctx = df_ds[["frontSightDistance", "backSightDistance"]].to_numpy(np.float32) if has_ctx else None

        has_safe = ("dhw" in df_ds.columns) and ("thw" in df_ds.columns) and ("ttc" in df_ds.columns)
        safety = df_ds[["dhw", "thw", "ttc"]].to_numpy(np.float32) if has_safe else None

        has_pre = ("precedingXVelocity" in df_ds.columns)
        prexv = df_ds[["precedingXVelocity"]].to_numpy(np.float32) if has_pre else None

        has_lane = ("laneId" in df_ds.columns)
        lane = df_ds["laneId"].fillna(0).astype(np.int16).to_numpy() if has_lane else None

        # vehicle maps
        uniq_ids, veh_rows, veh_frame_to_pos = build_vehicle_maps(df_ds)

        # visible slot mask
        keep_slot_mask = np.ones(8, dtype=bool)
        if cfg.visible_only:
            keep_slot_mask[:] = False
            keep_slot_mask[VISIBLE_KEEP_IDXS] = True

        # output lists
        X_hist_list, Y_fut_list = [], []
        NB_hist_list, NB_mask_list, NB_id_list = [], [], []
        TV_static_list, NB_static_list = [], []
        TV_context_list, TV_safety_list, TV_preceding_list, TV_lane_list = [], [], [], []
        trackId_list, t0_frame_list, drivingDirection_list = [], [], []

        # precompute meta for quick access
        # drivingDirection per vehicle (from tracksMeta)
        def get_meta(vid: int):
            mi = id_to_meta.get(vid, None)
            if mi is None:
                return 0.0, 0.0, 0, 0
            return float(mw[mi]), float(mh[mi]), int(mcls[mi]), int(mdd[mi])

        for vid in uniq_ids.tolist():
            v = int(vid)
            if v not in veh_rows:
                continue

            rows = veh_rows[v]  # global row indices for this vid, sorted by frame
            f = frames[rows]    # (Nv,)
            Nv = f.shape[0]
            if Nv < (T + Tf):
                continue

            # optional speed filter: mean abs xVelocity for this vehicle
            if cfg.min_speed_mps > 0.0:
                xv = feats6[rows, 2]  # xVelocity
                if float(np.mean(np.abs(xv))) < float(cfg.min_speed_mps):
                    continue

            # continuity check: frames should be equally spaced by ds_step
            # We'll allow windows only where the segment is perfect.
            # Precompute a boolean array good_start where the (T+Tf) segment is continuous.
            # We do it by checking fend - f0 == ds_step*((T+Tf)-1) (fast) and no gaps (stronger).
            # Stronger: check diff == ds_step in the segment. To keep it fast, do a quick gate then a local diff check.

            total_len = T + Tf
            for start in range(0, Nv - total_len + 1, stride):
                f0 = int(f[start])
                fend = int(f[start + total_len - 1])
                if (fend - f0) != ds_step * (total_len - 1):
                    continue
                # local diff check (cheap: only length total_len-1)
                if not np.all(np.diff(f[start:start + total_len]) == ds_step):
                    continue

                hist_rows = rows[start:start + T]
                fut_rows = rows[start + T:start + T + Tf]

                x_hist = feats6[hist_rows, :]   # (T,6)
                y_fut = feats6[fut_rows, 0:2]   # (Tf,2)

                # optional per-step
                tv_context = ctx[hist_rows, :] if has_ctx else np.zeros((T, 2), np.float32)
                tv_safety = safety[hist_rows, :] if has_safe else np.zeros((T, 3), np.float32)
                tv_pre = prexv[hist_rows, :] if has_pre else np.zeros((T, 1), np.float32)
                tv_lane = lane[hist_rows] if has_lane else np.zeros((T,), np.int16)

                tv_w, tv_h, tv_cls, tv_dd = get_meta(v)
                tv_static = np.array([tv_w, tv_h, tv_cls], dtype=np.float32)

                nb_hist = np.zeros((T, 8, 6), np.float32)
                nb_mask = np.zeros((T, 8), bool)
                nb_id_hist = np.zeros((T, 8), np.int32)

                # neighbor static from t0
                nb_static = np.zeros((8, 3), np.float32)
                nb0 = nb_ids[hist_rows[0], :]  # (8,)
                for s in range(8):
                    nbid0 = int(nb0[s])
                    if nbid0 != 0:
                        w0, h0, cls0, _dd0 = get_meta(nbid0)
                        if cls0 != 0 or (w0 != 0.0 or h0 != 0.0):
                            nb_static[s, :] = np.array([w0, h0, cls0], np.float32)

                # fill neighbor trajectories (relative)
                # Looping here is still O(T*slots), but avoids pandas overhead entirely.
                for t in range(T):
                    ego_row = hist_rows[t]
                    fr = int(frames[ego_row])
                    ego_feat = feats6[ego_row, :]  # (6,)
                    nbs = nb_ids[ego_row, :]       # (8,)

                    for s in range(8):
                        if not keep_slot_mask[s]:
                            continue
                        nbid = int(nbs[s])
                        nb_id_hist[t, s] = nbid
                        if nbid == 0:
                            continue
                        fm = veh_frame_to_pos.get(nbid, None)
                        if fm is None:
                            continue
                        pos = fm.get(fr, None)
                        if pos is None:
                            continue
                        nb_row = veh_rows[nbid][pos]
                        nb_feat = feats6[nb_row, :]
                        nb_hist[t, s, :] = (nb_feat - ego_feat)
                        nb_mask[t, s] = True

                X_hist_list.append(x_hist)
                Y_fut_list.append(y_fut)
                NB_hist_list.append(nb_hist)
                NB_mask_list.append(nb_mask)
                NB_id_list.append(nb_id_hist)
                TV_static_list.append(tv_static)
                TV_context_list.append(tv_context)
                TV_safety_list.append(tv_safety)
                TV_preceding_list.append(tv_pre)
                TV_lane_list.append(tv_lane)
                NB_static_list.append(nb_static)
                trackId_list.append(v)
                t0_frame_list.append(int(f0))
                drivingDirection_list.append(int(tv_dd))

        if len(X_hist_list) == 0:
            return rec_id, 0, None

        X_hist = np.stack(X_hist_list, axis=0)
        Y_fut = np.stack(Y_fut_list, axis=0)
        NB_hist = np.stack(NB_hist_list, axis=0)
        NB_mask = np.stack(NB_mask_list, axis=0)
        NB_id = np.stack(NB_id_list, axis=0)
        TV_static = np.stack(TV_static_list, axis=0)
        TV_context = np.stack(TV_context_list, axis=0)
        TV_safety = np.stack(TV_safety_list, axis=0)
        TV_preceding = np.stack(TV_preceding_list, axis=0)
        TV_lane = np.stack(TV_lane_list, axis=0)
        NB_static = np.stack(NB_static_list, axis=0)

        trackId = np.asarray(trackId_list, dtype=np.int32)
        t0_frame = np.asarray(t0_frame_list, dtype=np.int32)
        drivingDirection = np.asarray(drivingDirection_list, dtype=np.int8)
        recordingId_arr = np.full((X_hist.shape[0],), int(rec_id), dtype=np.int32)

        # output folder name from config
        Tn = int(round(cfg.history_sec * cfg.target_hz))
        Tfn = int(round(cfg.future_sec * cfg.target_hz))
        cfg_name = (
            f"highd_T{Tn}_Tf{Tfn}_hz{cfg.target_hz:g}_"
            f"{'flipXY' if cfg.normalize_upper_xy else 'noflip'}_"
            f"{'vis' if cfg.visible_only else 'all'}"
        )
        out_base = cfg.out_dir / cfg_name
        out_base.mkdir(parents=True, exist_ok=True)
        out_path = out_base / f"highd_{rec_id}.npz"

        np.savez_compressed(
            out_path,
            x_hist=X_hist,
            y_fut=Y_fut,
            nb_hist=NB_hist,
            nb_mask=NB_mask,
            nb_id=NB_id,
            tv_static=TV_static,
            tv_context=TV_context,
            tv_safety=TV_safety,
            tv_preceding=TV_preceding,
            tv_lane=TV_lane,
            nb_static=NB_static,
            recordingId=recordingId_arr,
            trackId=trackId,
            t0_frame=t0_frame,
            drivingDirection=drivingDirection,
            neighbor_slot_order=np.array(NEIGHBOR_COLS_8),
            target_hz=np.array([cfg.target_hz], dtype=np.float32),
            history_sec=np.array([cfg.history_sec], dtype=np.float32),
            future_sec=np.array([cfg.future_sec], dtype=np.float32),
            stride_sec=np.array([cfg.stride_sec], dtype=np.float32),
            visible_only=np.array([1 if cfg.visible_only else 0], dtype=np.int8),
            normalize_upper_xy=np.array([1 if cfg.normalize_upper_xy else 0], dtype=np.int8),
            y_flip_C=np.array([C_y], dtype=np.float32),
            downsample_step=np.array([ds_step], dtype=np.int32),
        )

        return rec_id, int(X_hist.shape[0]), None

    except Exception as e:
        return rec_id, 0, f"{type(e).__name__}: {e}"


def _worker(args):
    rec_id, cfg = args
    return make_windows_for_recording(rec_id, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="./raw")
    ap.add_argument("--out_dir", type=str, default="./data")
    ap.add_argument("--recording_ids", type=str, default="")
    ap.add_argument("--target_hz", type=float, default=5.0)
    ap.add_argument("--history_sec", type=float, default=3.0)
    ap.add_argument("--future_sec", type=float, default=5.0)
    ap.add_argument("--stride_sec", type=float, default=1.0)
    ap.add_argument("--min_speed_mps", type=float, default=0.0)
    ap.add_argument("--visible_only", action="store_true")
    ap.add_argument("--normalize_upper_xy", action="store_true")
    ap.add_argument("--num_workers", type=int, default=1, help=">1 to parallelize across recordings")
    args = ap.parse_args()

    cfg = Config(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        target_hz=float(args.target_hz),
        history_sec=float(args.history_sec),
        future_sec=float(args.future_sec),
        stride_sec=float(args.stride_sec),
        min_speed_mps=float(args.min_speed_mps),
        visible_only=bool(args.visible_only),
        normalize_upper_xy=bool(args.normalize_upper_xy),
    )

    if args.recording_ids.strip():
        rec_ids = [s.strip() for s in args.recording_ids.split(",") if s.strip()]
    else:
        rec_ids = find_recording_ids(cfg.raw_dir)
        if not rec_ids:
            raise RuntimeError(f"No '*_tracks.csv' found in {cfg.raw_dir}")

    work = [(rid, cfg) for rid in rec_ids]

    if int(args.num_workers) > 1:
        with mp.Pool(processes=int(args.num_workers)) as pool:
            results = pool.map(_worker, work)
    else:
        results = [_worker(w) for w in work]

    total = 0
    for rid, n, err in results:
        if err:
            print(f"[recording {rid}] FAIL: {err}")
            continue
        if n == 0:
            print(f"[recording {rid}] No samples produced. (Check filters/parameters)")
        else:
            print(f"[recording {rid}] Saved {n} samples.")
            total += n

    print(f"Done. Total samples saved: {total}")


if __name__ == "__main__":
    main()