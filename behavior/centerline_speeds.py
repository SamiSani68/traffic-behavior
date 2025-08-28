#!/usr/bin/env python3
"""
Compute along-road coordinates and speeds from DeepSORT tracks using per-region homographies.

Inputs
------
- --tracks_dir: directory with per-scene DeepSORT CSVs (e.g., A_30m_deepsort_tracks.csv).
  The script is robust to common column names:
    * (xc, yc) or (center_x, center_y)          -> directly used
    * (x, y, w, h) or (left, top, width, height)-> center = (x + w/2, y + h/2)
    * (xmin, ymin, xmax, ymax)                   -> center = ((xmin+xmax)/2, (ymin+ymax)/2)
- --gcp_dir:     directory with region polygons:
    <scene>_<region>_points_pixel.csv   (4 rows with X*, Y* columns)
- --H_dir:       directory with homographies produced by fit_homography.py:
    H_<scene>_<region>.npy
    meta/<scene>_<region>.json  (optional; used to compute offset_m if present)
- --fps:         frames per second of the video

Outputs (per scene)
-------------------
- <scene>_speeds_per_frame.csv
    columns: frame, track_id, cx, cy, region, s_local_m, offset_m,
             speed_kmh_raw, speed_kmh_smooth
- <scene>_track_speeds_summary.csv
- console report

Notes
-----
- Region assignment uses point-in-polygon on the detection center with a margin (hysteresis) so we
  avoid flicker near seams. Additional temporal stabilization is applied per track.
- s_local_m is the along-road coordinate in the region’s local metric rectangle (X axis).
  Global continuity across regions is handled later by the calibration step.
"""

import argparse
import glob
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import json


REGIONS = ("bottom", "middle", "top")


# ----------------------------- geometry utils -----------------------------

def order_quad_by_angle(pts: np.ndarray) -> np.ndarray:
    """Return 4x2 polygon ordered counter-clockwise by angle around centroid."""
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    idx = np.argsort(ang)
    return pts[idx]


def point_in_convex_polygon_with_margin(pt: Tuple[float, float], poly: np.ndarray, margin: float) -> bool:
    """
    Check if a point is inside a convex polygon and at least `margin` pixels away from every edge.
    poly: Nx2 ordered CCW (we'll order it if needed).
    """
    P = order_quad_by_angle(poly.astype(float))
    x, y = float(pt[0]), float(pt[1])

    # inside test via cross products (CCW -> all >= 0)
    inside = True
    min_edge_dist = float("inf")
    for i in range(len(P)):
        x1, y1 = P[i]
        x2, y2 = P[(i + 1) % len(P)]
        # edge vector
        ex, ey = x2 - x1, y2 - y1
        # vector from edge start to point
        px, py = x - x1, y - y1
        # cross product (z-component)
        cross = ex * py - ey * px
        if cross < 0:  # outside for CCW
            inside = False
            break
        # point-to-segment distance
        seg_len2 = ex * ex + ey * ey
        if seg_len2 == 0:
            d = np.hypot(px, py)
        else:
            t = max(0.0, min(1.0, (px * ex + py * ey) / seg_len2))
            dx = x1 + t * ex - x
            dy = y1 + t * ey - y
            d = np.hypot(dx, dy)
        if d < min_edge_dist:
            min_edge_dist = d

    return inside and (min_edge_dist >= margin)


def apply_H_point(H: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    """
    Apply 3x3 homography to a pixel point -> metric rectangle coords (s, t).
    """
    v = H @ np.array([x, y, 1.0], dtype=float)
    if v[2] == 0:
        return (np.nan, np.nan)
    return (float(v[0] / v[2]), float(v[1] / v[2]))


# ----------------------------- IO helpers --------------------------------

def find_track_files(tracks_dir: Path) -> Dict[str, Path]:
    """
    Map scene -> CSV path by inspecting filenames in tracks_dir.
    Accepts patterns like:
      A_30m_deepsort_tracks.csv, A_30m_tracks.csv, A_30m.csv
    """
    files = []
    for p in tracks_dir.glob("*.csv"):
        files.append(p)
    # also search recursively if user keeps subfolders per scene
    if not files:
        files = [Path(p) for p in glob.glob(str(tracks_dir / "**/*.csv"), recursive=True)]
    mapping: Dict[str, Path] = {}
    for p in files:
        name = p.name
        # scene stem is everything before first suffix/underscore pattern ending with region
        # try to detect *_<region>_*
        try:
            base = name.rsplit(".", 1)[0]
            # normalize like A_30m or C_220m or B_70m
            parts = base.split("_")
            # find last part that looks like a region hint ('bottom', 'top' etc.) -> ignore
            # we want scene stem like A_30m, B_50m, C_220m
            # heuristic: scene is first two parts (e.g., A and 30m) or one part with 'm'
            if len(parts) >= 2 and parts[1].endswith("m"):
                scene = "_".join(parts[:2])
            else:
                # fallback: until part that ends with 'm'
                idx = 0
                for i, t in enumerate(parts):
                    if t.endswith("m"):
                        idx = i
                        break
                scene = "_".join(parts[:idx + 1]) if idx > 0 else base
            mapping[scene] = p
        except Exception:
            continue
    return mapping


def load_polygon(gcp_dir: Path, scene: str, region: str) -> Optional[np.ndarray]:
    csv_path = gcp_dir / f"{scene}_{region}_points_pixel.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    # find first X*/Y* columns
    xcols = [c for c in df.columns if c.lower().startswith("x")]
    ycols = [c for c in df.columns if c.lower().startswith("y")]
    if not xcols or not ycols:
        return None
    pts = df[[xcols[0], ycols[0]]].to_numpy(dtype=float)
    if pts.shape[0] < 3:
        return None
    return pts[:, :2]


def load_H(H_dir: Path, scene: str, region: str) -> Optional[np.ndarray]:
    path = H_dir / f"H_{scene}_{region}.npy"
    if not path.exists():
        return None
    return np.load(path)


def load_meta_W(H_dir: Path, scene: str, region: str) -> Optional[float]:
    meta = H_dir / "meta" / f"{scene}_{region}.json"
    if not meta.exists():
        return None
    try:
        d = json.loads(meta.read_text(encoding="utf-8"))
        return float(d.get("W_m", np.nan))
    except Exception:
        return None


# ----------------------------- track utils -------------------------------

def detect_center_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Return column names for detection center (cx, cy).
    Supports several common schemas.
    """
    candidates = [
        ("xc", "yc"), ("center_x", "center_y"), ("cx", "cy"),
        ("x_center", "y_center"), ("x_c", "y_c")
    ]
    for a, b in candidates:
        if a in df.columns and b in df.columns:
            return a, b

    # bbox forms
    if {"x", "y", "w", "h"}.issubset(df.columns):
        df["__cx"] = df["x"] + df["w"] / 2.0
        df["__cy"] = df["y"] + df["h"] / 2.0
        return "__cx", "__cy"
    if {"left", "top", "width", "height"}.issubset(df.columns):
        df["__cx"] = df["left"] + df["width"] / 2.0
        df["__cy"] = df["top"] + df["height"] / 2.0
        return "__cx", "__cy"
    if {"xmin", "ymin", "xmax", "ymax"}.issubset(df.columns):
        df["__cx"] = (df["xmin"] + df["xmax"]) / 2.0
        df["__cy"] = (df["ymin"] + df["ymax"]) / 2.0
        return "__cx", "__cy"

    # last resort: any x/y
    xs = [c for c in df.columns if c.lower() in ("x", "px", "u")]
    ys = [c for c in df.columns if c.lower() in ("y", "py", "v")]
    if xs and ys:
        return xs[0], ys[0]

    raise ValueError("Could not infer detection center columns.")


def assign_region(cx: float, cy: float,
                  polys: Dict[str, np.ndarray],
                  margin_px: float) -> Optional[str]:
    """
    Decide which region a point belongs to; returns one of 'bottom','middle','top' or None.
    If inside multiple, prefer the one with the greatest minimum edge distance (deeper inside).
    """
    best_region = None
    best_depth = -1.0
    for reg, poly in polys.items():
        if poly is None:
            continue
        # Depth-aware inside test
        P = order_quad_by_angle(poly.astype(float))
        inside = True
        min_edge_dist = float("inf")
        x, y = float(cx), float(cy)
        for i in range(len(P)):
            x1, y1 = P[i]
            x2, y2 = P[(i + 1) % len(P)]
            ex, ey = x2 - x1, y2 - y1
            px, py = x - x1, y - y1
            cross = ex * py - ey * px
            if cross < 0:
                inside = False
                break
            seg_len2 = ex * ex + ey * ey
            if seg_len2 == 0:
                d = np.hypot(px, py)
            else:
                t = max(0.0, min(1.0, (px * ex + py * ey) / seg_len2))
                dx = x1 + t * ex - x
                dy = y1 + t * ey - y
                d = np.hypot(dx, dy)
            if d < min_edge_dist:
                min_edge_dist = d
        if inside and min_edge_dist >= margin_px:
            # choose region where we're deepest inside
            if min_edge_dist > best_depth:
                best_depth = min_edge_dist
                best_region = reg
    return best_region


def stabilize_regions(frames: np.ndarray,
                      raw_regions: np.ndarray,
                      min_stable: int,
                      drop_at_switch: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Temporal stabilization of region labels per track.
    Returns (stable_regions, drop_mask) where drop_mask is True on frames to drop.
    Rule: a change only takes effect after 'min_stable' consecutive frames of the new region.
    We drop 'drop_at_switch' frames starting at the first frame of an accepted switch.
    """
    n = len(frames)
    stable = np.array(raw_regions, dtype=object).copy()
    drop = np.zeros(n, dtype=bool)

    last = raw_regions[0]
    cur = last
    streak = 1

    for i in range(1, n):
        r = raw_regions[i]
        if r == cur:
            streak += 1
            stable[i] = cur
        else:
            # New candidate region starts/continues
            if r == last:
                # bouncing back to previous reading; keep current
                stable[i] = cur
                streak = 1
            else:
                # count consecutive frames in the new region
                # look ahead window to verify stability
                ahead = 1
                while i + ahead < n and raw_regions[i + ahead] == r:
                    ahead += 1
                if ahead >= min_stable and r is not None:
                    # accept switch
                    cur = r
                    stable[i:i + ahead] = r
                    # drop a few frames at the start of the switch
                    drop[i:i + min(drop_at_switch, ahead)] = True
                    streak = ahead
                else:
                    # reject switch for now
                    stable[i] = cur
                    streak = 1
        last = r if r is not None else last

    return stable, drop


# ----------------------------- main processing -----------------------------

def process_scene(scene: str,
                  tracks_path: Path,
                  gcp_dir: Path,
                  H_dir: Path,
                  out_dir: Path,
                  fps: float,
                  roll_win: int,
                  vmax_kmh: float,
                  margin_px: float,
                  min_stable_frames: int,
                  drop_frames_at_switch: int) -> None:
    # polygons & H per region
    polys: Dict[str, Optional[np.ndarray]] = {r: load_polygon(gcp_dir, scene, r) for r in REGIONS}
    Hs: Dict[str, Optional[np.ndarray]] = {r: load_H(H_dir, scene, r) for r in REGIONS}
    Ws: Dict[str, Optional[float]] = {r: load_meta_W(H_dir, scene, r) for r in REGIONS}

    # read tracks
    df = pd.read_csv(tracks_path)
    # required basic columns
    # try common names for frame and track id
    frame_col = "frame" if "frame" in df.columns else ("frame_id" if "frame_id" in df.columns else None)
    track_col = "track_id" if "track_id" in df.columns else ("id" if "id" in df.columns else None)
    if frame_col is None or track_col is None:
        raise ValueError(f"{tracks_path.name}: missing 'frame'/'track_id' (or 'frame_id'/'id') columns")

    cx_col, cy_col = detect_center_columns(df)
    df["__cx"] = pd.to_numeric(df[cx_col], errors="coerce")
    df["__cy"] = pd.to_numeric(df[cy_col], errors="coerce")
    df["frame"] = pd.to_numeric(df[frame_col], errors="coerce")
    df["track_id"] = pd.to_numeric(df[track_col], errors="coerce")
    df = df.dropna(subset=["__cx", "__cy", "frame", "track_id"]).copy()

    out_rows = []

    for tid, g in df.groupby("track_id", sort=True):
        g = g.sort_values("frame").copy()
        cxs = g["__cx"].to_numpy(float)
        cys = g["__cy"].to_numpy(float)
        frames = g["frame"].to_numpy(int)

        # raw region assignment
        raw_regs = []
        for x, y in zip(cxs, cys):
            reg = assign_region(x, y, polys, margin_px=margin_px)
            raw_regs.append(reg)
        raw_regs = np.array(raw_regs, dtype=object)

        # stabilize region sequence
        stable_regs, drop_mask = stabilize_regions(frames, raw_regs,
                                                   min_stable=min_stable_frames,
                                                   drop_at_switch=drop_frames_at_switch)

        # project with region-specific H
        s_local = np.full(len(frames), np.nan, dtype=float)
        t_local = np.full(len(frames), np.nan, dtype=float)

        for i, (x, y, reg) in enumerate(zip(cxs, cys, stable_regs)):
            if reg is None:
                continue
            H = Hs.get(reg)
            if H is None:
                continue
            s, t = apply_H_point(H, x, y)
            s_local[i] = s
            t_local[i] = t

        # derive offset_m if W available (centered across width)
        offset = np.full(len(frames), np.nan, dtype=float)
        for i, reg in enumerate(stable_regs):
            if reg is None:
                continue
            W = Ws.get(reg)
            if W is not None and np.isfinite(W):
                offset[i] = t_local[i] - (W / 2.0)
            else:
                offset[i] = np.nan

        # compute speeds (km/h) from s_local
        speed_raw = np.full(len(frames), np.nan, dtype=float)
        # invalidate speeds at switches and drops
        same_reg = np.array([False] + [stable_regs[i] == stable_regs[i - 1] for i in range(1, len(stable_regs))], dtype=bool)
        valid = (~drop_mask) & same_reg & np.isfinite(s_local)
        # compute diffs only where same region and finite
        ds = np.diff(s_local)
        dt = np.diff(frames) / float(fps)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.abs(ds / dt) * 3.6
        # place back into array (shifted by 1)
        speed_raw[1:] = np.where((valid[1:]) & np.isfinite(v), v, np.nan)
        # smooth (median window)
        roll = int(max(1, roll_win))
        s_raw_series = pd.Series(speed_raw)
        speed_smooth = s_raw_series.rolling(roll, center=True, min_periods=1).median().to_numpy()

        # clamp implausible values
        for arr in (speed_raw, speed_smooth):
            bad = (~np.isfinite(arr)) | (arr < 0) | (arr > float(vmax_kmh))
            arr[bad] = np.nan

        out = pd.DataFrame({
            "frame": frames,
            "track_id": tid,
            "cx": cxs,
            "cy": cys,
            "region": stable_regs,
            "s_local_m": s_local,
            "offset_m": offset,
            "speed_kmh_raw": speed_raw,
            "speed_kmh_smooth": speed_smooth
        })
        out_rows.append(out)

    if not out_rows:
        print(f"[warn] {scene}: no valid tracks/frames after filtering")
        return

    pf = pd.concat(out_rows, ignore_index=True)
    pf = pf.sort_values(["track_id", "frame"])

    # write per-frame
    out_dir.mkdir(parents=True, exist_ok=True)
    pf.to_csv(out_dir / f"{scene}_speeds_per_frame.csv", index=False)

    # per-track summary
    sm = (pf.groupby("track_id")
            .agg(n_frames=("frame", "count"),
                 mean_kmh=("speed_kmh_smooth", "mean"),
                 median_kmh=("speed_kmh_smooth", "median"),
                 max_kmh=("speed_kmh_smooth", "max"))
            .reset_index())
    sm.to_csv(out_dir / f"{scene}_track_speeds_summary.csv", index=False)

    print(f"[ok] {scene}: frames={len(pf)} tracks={len(sm)} -> {out_dir}/{scene}_speeds_per_frame.csv")


# ----------------------------- CLI ---------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Compute along-road coordinates and speeds using region homographies")
    ap.add_argument("--tracks_dir", required=True, help="Dir with per-scene DeepSORT CSVs")
    ap.add_argument("--gcp_dir", required=True, help="Dir with <scene>_<region>_points_pixel.csv")
    ap.add_argument("--H_dir", required=True, help="Dir with H_<scene>_<region>.npy and meta/")
    ap.add_argument("--out_dir", required=True, help="Output dir for per-frame and summary CSVs")

    ap.add_argument("--fps", type=float, default=29.97)
    ap.add_argument("--roll_win", type=int, default=9, help="Median window (frames) for speed smoothing")
    ap.add_argument("--vmax_kmh", type=float, default=180.0, help="Clamp speeds above this to NaN")

    # Region assignment stabilization
    ap.add_argument("--hyst_px", type=float, default=16.0, help="Margin from region boundary (pixels) to accept membership")
    ap.add_argument("--keep_frames_after_switch", type=int, default=5,
                    help="Min consecutive frames required in new region to confirm switch")
    ap.add_argument("--drop_frames_at_switch", type=int, default=2,
                    help="Drop this many frames starting at a confirmed switch")

    args = ap.parse_args()

    tracks_dir = Path(args.tracks_dir)
    gcp_dir = Path(args.gcp_dir)
    H_dir = Path(args.H_dir)
    out_dir = Path(args.out_dir)

    mapping = find_track_files(tracks_dir)
    if not mapping:
        raise SystemExit(f"No CSV tracks found in {tracks_dir}")

    for scene, p in sorted(mapping.items()):
        process_scene(scene=scene,
                      tracks_path=p,
                      gcp_dir=gcp_dir,
                      H_dir=H_dir,
                      out_dir=out_dir,
                      fps=float(args.fps),
                      roll_win=int(args.roll_win),
                      vmax_kmh=float(args.vmax_kmh),
                      margin_px=float(args.hyst_px),
                      min_stable_frames=int(args.keep_frames_after_switch),
                      drop_frames_at_switch=int(args.drop_frames_at_switch))


if __name__ == "__main__":
    main()
