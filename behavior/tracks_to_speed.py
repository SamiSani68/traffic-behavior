from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

KMH = 3.6

# ───────────────────────────────────────────────────────────────────────────────
# helpers
# ───────────────────────────────────────────────────────────────────────────────

def derive_m_per_px(dy_px: Sequence[float], dy_m: Sequence[float]) -> float:
    """Return average metres‑per‑pixel from parallel pixel / metre lists."""
    ratios = [m / px for px, m in zip(dy_px, dy_m) if px > 0]
    if not ratios:
        raise ValueError("No valid dy_px / dy_m pairs provided")
    return float(np.mean(ratios))


def main(tracks_csv: Path, fps: float, m_per_px: float, out_csv: Path):
    with tracks_csv.open() as fh:
        first = fh.readline().lower()
    df = pd.read_csv(tracks_csv,
                     header=0 if first.startswith("frame") else None,
                     names=["frame", "id", "cls", "xc", "yc", "w", "h"])

    # bottom‑centre pixel Y
    df["y_px"] = df["yc"] + df["h"] / 2.0
    df.sort_values(["id", "frame"], inplace=True)

    # time delta (s) and pixel delta (abs)
    df["dt"] = df.groupby("id")["frame"].diff() / fps
    df["dy"] = df.groupby("id")["y_px"].diff().abs()

    # speed calculation
    df["speed_kmh"] = (df["dy"] * m_per_px / df["dt"]) * KMH
    df.loc[df["dt"].isna(), "speed_kmh"] = 0.0   # first row per track

    df_out = df[["frame", "id", "y_px", "speed_kmh"]]
    df_out.to_csv(out_csv, index=False)
    print(f"✅ wrote {out_csv}   rows={len(df_out)}   m_per_px={m_per_px:.4f}")


# ───────────────────────────────────────────────────────────────────────────────
# cli
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Vertical‑motion speed estimator from DeepSORT tracks")
    ap.add_argument("--tracks", required=True, type=Path, help="DeepSORT CSV file")
    ap.add_argument("--fps", default=30.0, type=float, help="Video frame‑rate [30]")
    ap.add_argument("--m_per_px", type=float, help="Metres per pixel in Y (if known)")
    ap.add_argument("--dy_px", nargs=2, type=float, metavar=("B2M_PX", "M2T_PX"),
                    help="Pixel gaps for bottom→middle and middle→top strips")
    ap.add_argument("--dy_m", nargs=2, type=float, metavar=("B2M_M", "M2T_M"),
                    help="Real‑world depths (m) for the same two gaps")
    ap.add_argument("--out_csv", type=Path, help="Output CSV path")
    args = ap.parse_args()

    if args.m_per_px is None:
        if args.dy_px and args.dy_m:
            args.m_per_px = derive_m_per_px(args.dy_px, args.dy_m)
        else:
            ap.error("Need either --m_per_px or both --dy_px and --dy_m")

    out_path = args.out_csv or args.tracks.with_name(args.tracks.stem + "_vyspeed.csv")
    main(args.tracks, args.fps, args.m_per_px, out_path)
