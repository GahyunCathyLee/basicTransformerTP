import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("./raw")
REC_ID = "01"          # 디버깅할 recording
TARGET_HZ = 5.0
HISTORY_SEC = 3.0
FUTURE_SEC = 5.0
STRIDE_SEC = 1.0

def main():
    print(f"=== Debugging recording {REC_ID} ===")

    # ----------------------------
    # Load CSVs
    # ----------------------------
    tracks = pd.read_csv(RAW_DIR / f"{REC_ID}_tracks.csv")
    tracks_meta = pd.read_csv(RAW_DIR / f"{REC_ID}_tracksMeta.csv")
    rec_meta = pd.read_csv(RAW_DIR / f"{REC_ID}_recordingMeta.csv")

    print("[1] CSV loaded")
    print("tracks rows:", len(tracks))
    print("unique vehicles:", tracks["id"].nunique())

    # ----------------------------
    # Frame rate & downsample
    # ----------------------------
    frame_rate = float(rec_meta.loc[0, "frameRate"])
    ds_step = int(round(frame_rate / TARGET_HZ))

    T = int(round(HISTORY_SEC * TARGET_HZ))
    Tf = int(round(FUTURE_SEC * TARGET_HZ))
    stride = max(1, int(round(STRIDE_SEC * TARGET_HZ)))

    print("\n[2] Parameters")
    print("frame_rate:", frame_rate)
    print("target_hz:", TARGET_HZ)
    print("ds_step:", ds_step)
    print("T(history):", T)
    print("Tf(future):", Tf)
    print("stride:", stride)

    # ----------------------------
    # Downsample frames
    # ----------------------------
    tracks["frame"] = tracks["frame"].astype(int)
    tracks_ds = tracks[((tracks["frame"] - 1) % ds_step) == 0].copy()

    print("\n[3] Downsampling")
    print("rows after downsample:", len(tracks_ds))
    print("unique vehicles after downsample:", tracks_ds["id"].nunique())

    # ----------------------------
    # Per-vehicle frame continuity check
    # ----------------------------
    print("\n[4] Frame continuity check (per vehicle)")
    bad_continuity = 0
    good_continuity = 0

    for vid, df_v in tracks_ds.groupby("id"):
        frames = df_v["frame"].values
        if len(frames) < (T + Tf):
            continue

        diffs = np.diff(frames)
        if not np.all(diffs == ds_step):
            bad_continuity += 1
        else:
            good_continuity += 1

    print("vehicles with perfect continuity:", good_continuity)
    print("vehicles with broken continuity:", bad_continuity)

    # ----------------------------
    # Window-level check
    # ----------------------------
    print("\n[5] Window-level survival analysis")

    total_windows = 0
    ok_windows = 0
    fail_length = 0
    fail_continuity = 0

    for vid, df_v in tracks_ds.groupby("id"):
        frames = df_v["frame"].values
        if len(frames) < (T + Tf):
            continue

        for start in range(0, len(frames) - (T + Tf) + 1, stride):
            total_windows += 1
            f0 = frames[start]
            fend = frames[start + (T + Tf) - 1]

            if (fend - f0) != ds_step * ((T + Tf) - 1):
                fail_continuity += 1
                continue

            ok_windows += 1

    print("total candidate windows:", total_windows)
    print("windows passing continuity check:", ok_windows)
    print("windows failing continuity check:", fail_continuity)

    # ----------------------------
    # Neighbor existence sanity check
    # ----------------------------
    print("\n[6] Neighbor ID sanity check (t0 only)")
    neighbor_cols = [
        "precedingId", "followingId",
        "leftPrecedingId", "leftAlongsideId", "leftFollowingId",
        "rightPrecedingId", "rightAlongsideId", "rightFollowingId",
    ]

    sample_vid = tracks_ds["id"].iloc[0]
    sample_row = tracks_ds[tracks_ds["id"] == sample_vid].iloc[0]

    print(f"Sample vehicle id: {sample_vid}")
    for c in neighbor_cols:
        print(f"{c}: {sample_row[c]}")

    print("\n=== Debug finished ===")


if __name__ == "__main__":
    main()
