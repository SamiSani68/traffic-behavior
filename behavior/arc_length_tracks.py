#!/usr/bin/env python3
"""
arc_length_speed.py ─ compute speeds for A & B series from projected tracks
---------------------------------------------------------------------------
• projects smoothed detections onto the centre-line (arc_points.csv)
• converts frame-to-frame arc-length Δ to m/s → km/h
• filters parked vehicles (total displacement < 5 m)
• clamps physically impossible spikes (> 200 km/h)
• writes <stem>_speed.csv to behavior/speed_tracks   (same place as C-series)
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# ───────────────────────── helpers ─────────────────────────
def cumulative_lengths(poly: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))


def project(pt: np.ndarray, poly: np.ndarray, arc: np.ndarray) -> float:
    best_arc, best_d2 = 0.0, float("inf")
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        ab, ap = b - a, pt - a
        t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
        proj = a + t * ab
        d2 = np.square(pt - proj).sum()
        if d2 < best_d2:
            best_d2, best_arc = d2, arc[i] + np.linalg.norm(proj - a)
    return best_arc


def rolling(series: pd.Series, win: int) -> pd.Series:
    return series.rolling(win, center=True, min_periods=1).mean()


# ───────────────────── processing routine ───────────────────
def process_ab(
    track_csv: Path,
    arc_dir: Path,
    out_dir: Path,
    fps: float,
    win: int,
    move_thresh: float,
    spike_kmh: float,
):
    stem = track_csv.stem.replace("_projected", "")
    arc_csv = arc_dir / f"{stem}_arc_points.csv"  # already in metres
    if not arc_csv.exists():
        print(f"[skip] no arc points for {stem}")
        return

    df = pd.read_csv(track_csv)
    if not {"x_ground", "y_ground", "track_id", "frame"}.issubset(df.columns):
        print(f"[skip] missing columns in {track_csv.name}")
        return

    # centre-line
    poly = pd.read_csv(arc_csv)[["x", "y"]].to_numpy()
    arc = cumulative_lengths(poly)

    # smooth detections
    df.sort_values(["track_id", "frame"], inplace=True)
    df["x_s"] = df.groupby("track_id")["x_ground"].transform(lambda s: rolling(s, win))
    df["y_s"] = df.groupby("track_id")["y_ground"].transform(lambda s: rolling(s, win))

    # project each detection ⇒ arc-length
    df["arc_length"] = df.apply(
        lambda r: project(np.array([r.x_s, r.y_s]), poly, arc), axis=1
    )

    # parked-car filter (movement in metres along the centre-line)
    moving_ids = (
        df.groupby("track_id")["arc_length"]
        .agg(lambda s: s.max() - s.min())
        .pipe(lambda s: s[s >= move_thresh].index)
    )
    df = df[df["track_id"].isin(moving_ids)].copy()
    if df.empty:
        print(f"[skip] all stationary for {stem}")
        return

    # speed computation
    df["d_arc"] = df.groupby("track_id")["arc_length"].diff().abs()
    df["dt"] = df.groupby("track_id")["frame"].diff() / fps
    df["speed_mps"] = df["d_arc"].div(df["dt"].replace(0, np.nan))
    df["speed_mps"].fillna(0, inplace=True)

    # spike clamp (> spike_kmh)
    spike_mps = spike_kmh / 3.6
    df.loc[df["speed_mps"] > spike_mps, "speed_mps"] = np.nan
    df["speed_mps"].interpolate(limit_direction="both", inplace=True)

    # enforce 0 km/h on first row of every track
    first_rows = df.groupby("track_id").head(1).index
    df.loc[first_rows, "speed_mps"] = 0.0

    df["speed_kmh"] = df["speed_mps"] * 3.6

    # save
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_speed.csv"
    df[["frame", "track_id", "arc_length", "speed_kmh"]].to_csv(out_path, index=False)
    print(f"✅ {stem}: tracks={df['track_id'].nunique():>4}  rows={len(df):>7}  → {out_path.name}")


# ─────────────────────────── CLI ───────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Arc-length speed for A/B series")
    ap.add_argument("--tracks_dir", default="behavior/projected_tracks", type=Path)
    ap.add_argument("--arc_dir",    default="behavior/GCP_arc_points", type=Path)
    ap.add_argument("--out_dir",    default="behavior/speed_tracks",   type=Path)
    ap.add_argument("--fps",        default=30.0, type=float)
    ap.add_argument("--window",     default=5,    type=int,   help="moving-average window")
    ap.add_argument("--min_move",   default=5.0,  type=float, help="min. displacement (m)")
    ap.add_argument("--spike_kmh",  default=200.0,type=float, help="speed spike clamp")
    args = ap.parse_args()

    for csv in sorted(args.tracks_dir.glob("A*_projected.csv")):
        process_ab(csv, args.arc_dir, args.out_dir,
                   fps=args.fps, win=args.window,
                   move_thresh=args.min_move, spike_kmh=args.spike_kmh)

    for csv in sorted(args.tracks_dir.glob("B*_projected.csv")):
        process_ab(csv, args.arc_dir, args.out_dir,
                   fps=args.fps, win=args.window,
                   move_thresh=args.min_move, spike_kmh=args.spike_kmh)

    print("All A/B series finished.")