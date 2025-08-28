from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import argparse

# ── default directories ────────────────────────────────────────────────────────
VIDEOS_DIR   = Path("video-analysis/videos")
TRACKS_DIR   = Path("video-analysis/tracked_videos/deepsort")
SPEEDS_DIR   = Path("behavior/speed_tracks")
OUT_DIR      = Path("behavior/annotated_videos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE   = 0.5
FONT_THICK   = 1
BOX_COLOR    = (  0,255,  0)    # green
TEXT_COLOR   = (255,255,255)    # white

# ── helpers ────────────────────────────────────────────────────────────────────
def load_merge(tracks_csv: Path, speed_csv: Path) -> dict[int, list[dict]]:
    """Merge tracks+speed into dict keyed by frame → list(row_dict)."""
    tcols = ["frame","track_id","cls","xc","yc","w","h"]
    tracks = pd.read_csv(tracks_csv, names=tcols) if not tracks_csv.read_text().startswith("frame") else pd.read_csv(tracks_csv)
    speed  = pd.read_csv(speed_csv)[["frame","track_id","speed_kmh"]]
    df = tracks.merge(speed, on=["frame","track_id"], how="left") \
               .sort_values(["frame","track_id"])
    return {f: grp.to_dict("records") for f,grp in df.groupby("frame")}

def annotate_video(video_path: Path, frame_rows: dict, out_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (W,H))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        for r in frame_rows.get(frame_idx, []):
            x1,y1,x2,y2 = map(int, [r["xc"]-r["w"]/2, r["yc"]-r["h"]/2,
                                    r["xc"]+r["w"]/2, r["yc"]+r["h"]/2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),BOX_COLOR,2)
            speed_txt = f"{r['speed_kmh']:.1f} km/h" if not np.isnan(r["speed_kmh"]) else "-"
            label = f"ID {int(r['track_id'])} | {speed_txt}"
            (tw,th),_ = cv2.getTextSize(label,FONT,FONT_SCALE,FONT_THICK)
            cv2.rectangle(frame,(x1,y1-th-4),(x1+tw+4,y1),BOX_COLOR,-1)
            cv2.putText(frame,label,(x1+2,y1-2),FONT,FONT_SCALE,TEXT_COLOR,FONT_THICK,cv2.LINE_AA)
        out.write(frame); frame_idx += 1
    cap.release(); out.release()

# ── CLI & batch driver ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch annotate every video with bounding-boxes + speed")
    ap.add_argument("--videos_dir",  default=VIDEOS_DIR,  type=Path)
    ap.add_argument("--tracks_dir",  default=TRACKS_DIR,  type=Path)
    ap.add_argument("--speeds_dir",  default=SPEEDS_DIR,  type=Path)
    ap.add_argument("--out_dir",     default=OUT_DIR,     type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for video in sorted(args.videos_dir.glob("*.mp4")):
        stem  = video.stem
        tracks_csv = args.tracks_dir / f"{stem}_deepsort_tracks.csv"
        speed_csv  = args.speeds_dir  / f"{stem}_speed.csv"
        if not (tracks_csv.exists() and speed_csv.exists()):
            print(f"[skip] {stem}: missing tracks or speed CSV"); continue
        rows_per_frame = load_merge(tracks_csv, speed_csv)
        out_path = args.out_dir / f"{stem}_annot.mp4"
        annotate_video(video, rows_per_frame, out_path)
        print(f"✅ annotated {stem} → {out_path}")

    print("All videos processed.")