#!/usr/bin/env python3
"""
click_gcp.py (Single Region Version)

Click GCPs directly on real video frames (not screenshots) so pixel coords
match your tracker exactly.

For each video:
  - scrub to a frame (trackbar / keys), press ENTER to lock the frame
  - click 4 points in order 1→2→3→4 around the quad
  - then enter distances (meters) for edges:
      1–2 (along lane), 2–3 (across), 3–4 (along), 4–1 (across)
  - CSVs are saved:
      <code>_points_pixel.csv  (columns: x,y)
      <code>_distances.csv     (columns: from,to,dist_m)
  - a PNG snapshot + JSON metadata with the chosen frame is saved too.

Optional: overlay DeepSORT bottom points for the current frame to verify alignment.

Usage:
  python click_gcp_from_video_single.py \
    --videos_dir /path/to/your/videos \
    --out_dir    /path/to/your/output \
    --tracks_dir /path/to/deepsort_tracks (optional)

      python click_gcp.py \
    --videos_dir /path/to/your/videos \
    --out_dir    /path/to/your/output \
    --tracks_dir /path/to/deepsort_tracks (optional)

Controls:
  - Trackbar   : scrub frames
  - ← / →      : -/+ 1 frame
  - , / .      : -/+ 10 frames
  - [ / ]      : -/+ 100 frames
  - SPACE      : toggle play/pause
  - ENTER      : lock current frame for this video (start clicking)
  - c          : clear current clicks
  - r          : restart whole video (unlock frame)
  - q          : skip whole video / quit when asked
"""

import os
import re
import json
import time
import glob
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd


def infer_code(path):
    m = re.search(r"([A-Za-z]_\d+m)", os.path.basename(path))
    return m.group(1) if m else Path(path).stem


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_tracks_for_video(tracks_dir: Path, code: str):
    csv_path = None
    for f in Path(tracks_dir).glob(f"{code}_deepsort_tracks.csv"):
        csv_path = f
        break
    if not csv_path:
        cand = list(Path(tracks_dir).glob("*_deepsort_tracks.csv"))
        for f in cand:
            if infer_code(f) == code:
                csv_path = f
                break
    if not csv_path or not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"frame", "x_bottom", "y_bottom"}.issubset(df.columns):
        return None

    frames = defaultdict(list)
    for _, r in df.iterrows():
        fr = int(r["frame"])
        xb = float(r.get("x_bottom", np.nan))
        yb = float(r.get("y_bottom", np.nan))
        if np.isfinite(xb) and np.isfinite(yb):
            frames[fr].append((xb, yb))
    return frames


def draw_tracks_overlay(img, frames_map, frame_idx):
    pts = frames_map.get(frame_idx)
    if not pts:
        return
    for (x, y) in pts:
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 255), -1)


def save_points_csv(out_csv: Path, pts):
    df = pd.DataFrame(pts, columns=["x", "y"])
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)


def save_distances_csv(out_csv: Path, dists):
    # dists order: [d12, d23, d34, d41]
    df = pd.DataFrame({
        "from": [1, 2, 3, 4],
        "to": [2, 3, 4, 1],
        "dist_m": dists
    })
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)


def prompt_distances(code):
    print(f"\nEnter real-world distances (meters) for {code}")
    print("  Tip: 1–2 and 3–4 are the long edges (along the lane).")
    vals = []
    labels = ["1→2 (long)", "2→3 (width)", "3→4 (long)", "4→1 (width)"]
    for lab in labels:
        while True:
            s = input(f"  Distance {lab}: ").strip()
            if s.lower() in ("q", "quit"):
                return None
            try:
                v = float(s)
                if v <= 0:
                    print("    Must be > 0.")
                    continue
                vals.append(v)
                break
            except ValueError:
                print("    Please enter a number, or 'q' to cancel this video.")
    return vals


def draw_ui_text(img, lines, x=10, y=20, lh=20, color=(255, 255, 255)):
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (x, y + i * lh), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def click_loop(win, base_frame, code, locked_frame_idx):
    """Collect 4 points in order 1..4 on the locked frame; then prompt distances."""
    img = base_frame.copy()
    h, w = img.shape[:2]
    clicks = []

    def on_mouse(event, x, y, flags, param):
        nonlocal clicks, img
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicks) < 4:
                clicks.append((x, y))
                # draw marker + index
                cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
                cv2.putText(img, str(len(clicks)), (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                # draw edge if not first
                if len(clicks) > 1:
                    cv2.line(img, clicks[-2], clicks[-1], (0, 0, 255), 2)
                if len(clicks) == 4:
                    cv2.line(img, clicks[-1], clicks[0], (0, 0, 255), 2)

    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = img.copy()
        draw_ui_text(canvas, [
            f"{code}   frame: {locked_frame_idx}",
            "Click 4 points in order 1→2→3→4 around the quad.",
            "[c]=clear   [Enter]=finish points & enter distances   [q]=cancel video"
        ], x=10, y=20)

        cv2.imshow(win, canvas)
        k = cv2.waitKey(30) & 0xFF

        if k in (ord('q'), 27):  # q or ESC
            return None, None
        elif k == ord('c'):
            clicks = []
            img = base_frame.copy()
        elif k in (13, 10):  # Enter
            if len(clicks) != 4:
                print("[WARN] Need exactly 4 clicks (got {}).".format(len(clicks)))
                continue
            # prompt distances in console
            dists = prompt_distances(code)
            if dists is None:
                print("[WARN] Video cancelled during distance entry.")
                return None, None

            return clicks, dists


def process_video(video_path: Path, out_dir: Path, tracks_dir=None, overwrite=False):
    code = infer_code(video_path)

    # Define output paths without region
    pixel_csv = out_dir / f"{code}_points_pixel.csv"
    dist_csv = out_dir / f"{code}_distances.csv"

    if pixel_csv.exists() and dist_csv.exists() and not overwrite:
        print(f"[SKIP] {code} (already exists)")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[SKIP] Cannot open {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_map = load_tracks_for_video(tracks_dir, code) if tracks_dir else None

    win = f"GCP picker: {code}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w, 1280), min(h, 720))

    cur = total // 2
    playing = False

    def on_trackbar(pos):
        nonlocal cur
        cur = int(pos)

    cv2.createTrackbar("frame", win, max(cur, 0), max(total - 1, 1), on_trackbar)

    locked_frame_idx = None
    base_frame = None

    print(f"\n=== {code} ===")
    print("Scrub to a clean frame, then press ENTER to lock it.")

    while True:
        if playing:
            cur = min(cur + 1, total - 1)
            cv2.setTrackbarPos("frame", win, cur)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
        ok, frame = cap.read()
        if not ok:
            playing = False
            time.sleep(0.02)
            continue

        canvas = frame.copy()
        draw_ui_text(canvas, [
            f"{code}  ({w}x{h})  fps={fps:.2f}  frame={cur}/{max(total - 1, 0)}",
            "Controls: ←/→ ±1 | ,/. ±10 | [/ ] ±100 | SPACE play/pause | ENTER lock frame | q skip video"
        ], x=10, y=20)

        if frames_map is not None:
            draw_tracks_overlay(canvas, frames_map, cur)
            cv2.putText(canvas, "DeepSORT points overlay", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win, canvas)
        k = cv2.waitKey(20) & 0xFF

        if k in (ord('q'), 27):
            cap.release()
            cv2.destroyWindow(win)
            print(f"[SKIP] {code}")
            return

        # --- Key controls for scrubbing video ---
        if k == ord(' '):
            playing = not playing
        elif k in (81, ord('h'), ord('a')):
            cur = max(0, cur - 1);
            cv2.setTrackbarPos("frame", win, cur)
        elif k in (83, ord('l'), ord('d')):
            cur = min(total - 1, cur + 1);
            cv2.setTrackbarPos("frame", win, cur)
        elif k == ord(','):
            cur = max(0, cur - 10);
            cv2.setTrackbarPos("frame", win, cur)
        elif k == ord('.'):
            cur = min(total - 1, cur + 10);
            cv2.setTrackbarPos("frame", win, cur)
        elif k == ord('['):
            cur = max(0, cur - 100);
            cv2.setTrackbarPos("frame", win, cur)
        elif k == ord(']'):
            cur = min(total - 1, cur + 100);
            cv2.setTrackbarPos("frame", win, cur)
        elif k in (13, 10):  # ENTER -> lock frame
            locked_frame_idx = int(cur)
            base_frame = frame.copy()
            break

    # Once frame is locked, start the clicking process
    res_pts, res_dists = click_loop(win, base_frame, code, locked_frame_idx)

    cap.release()
    cv2.destroyWindow(win)

    if res_pts is None or res_dists is None:
        print(f"[CANCEL] Video {code} was cancelled. No data saved.")
        return

    # --- Save all data at the end ---
    print(f"\n[INFO] Saving data for {code}...")

    # Save snapshot + metadata
    snap_dir = ensure_dir(out_dir / "_snapshots")
    meta_dir = ensure_dir(out_dir / "_meta")

    snap_path = snap_dir / f"{code}_frame{locked_frame_idx}.png"
    cv2.imwrite(str(snap_path), base_frame)
    meta_path = meta_dir / f"{code}.json"
    with open(meta_path, "w") as f:
        json.dump({"code": code, "frame": locked_frame_idx, "fps": float(fps),
                   "video": str(video_path)}, f, indent=2)
    print(f"[SAVED] snapshot: {snap_path}")
    print(f"[SAVED] meta    : {meta_path}")

    # Save points and distances
    save_points_csv(pixel_csv, res_pts)
    save_distances_csv(dist_csv, res_dists)
    print(f"[SAVED] {pixel_csv}")
    print(f"[SAVED] {dist_csv}")
    print(f"[DONE] {code}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--tracks_dir", type=Path, default=None,
                    help="Folder with *_deepsort_tracks.csv to overlay bottom points (optional).")
    ap.add_argument("--include", default="", help="Regex to include certain videos (e.g., '^A_|^B_')")
    ap.add_argument("--exclude", default="", help="Regex to exclude videos")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing CSVs for a video")
    args = ap.parse_args()

    videos = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        videos += glob.glob(str(args.videos_dir / ext))
    videos = sorted(videos)

    if args.include:
        rx = re.compile(args.include)
        videos = [v for v in videos if rx.search(os.path.basename(v))]
    if args.exclude:
        rx = re.compile(args.exclude)
        videos = [v for v in videos if not rx.search(os.path.basename(v))]

    if not videos:
        print("No videos found.")
        return

    out_dir = ensure_dir(args.out_dir)

    for vp in videos:
        process_video(Path(vp), out_dir, tracks_dir=args.tracks_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()