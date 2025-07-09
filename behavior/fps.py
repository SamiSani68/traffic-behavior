import argparse
import csv
from pathlib import Path
from typing import Optional
import cv2

def get_fps(video_path: Path) -> Optional[float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else None

def main(video_dir: Path, output_csv: Path) -> None:
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    video_files = sorted(
        p for p in video_dir.rglob("*") if p.suffix.lower() in video_exts
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "fps"])
        for clip in video_files:
            fps_val = get_fps(clip)
            rel_name = clip.relative_to(video_dir)
            writer.writerow([
                rel_name.as_posix(), f"{fps_val:.3f}" if fps_val else "ERROR"
            ])

    print(f"Wrote FPS for {len(video_files)} video(s) → {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate FPS for every video in a directory (recursively)."
    )
    parser.add_argument(
        "--video_dir",
        default="video-analysis/videos",
        type=Path,
        help="Directory that contains the video clips [default: video-analysis/videos]",
    )
    parser.add_argument(
        "--output_csv",
        default="video-analysis/video_fps.csv",
        type=Path,
        help="Where to write the CSV summary [default: video-analysis/video_fps.csv]",
    )
    args = parser.parse_args()

    if not args.video_dir.exists():
        parser.error(f"Video directory not found: {args.video_dir}")

    main(args.video_dir, args.output_csv)
