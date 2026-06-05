from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2


DEFAULT_SECONDS = [15, 25, 35, 45, 55]


def parse_seconds(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    seconds: list[int] = []
    for part in parts:
        try:
            seconds.append(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid seconds value: {part}") from exc
    if not seconds:
        raise argparse.ArgumentTypeError("Seconds list cannot be empty.")
    return seconds


def safe_stem(path: Path, index: int) -> str:
    stem = path.stem
    ascii_only = "".join(
        c if (c.isascii() and (c.isalnum() or c in "-_")) else "_" for c in stem
    )
    ascii_only = "_".join(part for part in ascii_only.split("_") if part)
    if not ascii_only:
        ascii_only = f"video_{index:02d}"
    return ascii_only


def categorize_output_dir(base_dir: Path, video_path: Path) -> Path:
    stem = video_path.stem
    if stem.startswith("窗戶"):
        return base_dir / "window"
    if stem.startswith("箱子"):
        return base_dir / "box"
    if stem.startswith("電腦"):
        return base_dir / "computer"
    return base_dir / "other"


def save_png(out_path: Path, frame) -> bool:
    ok, buffer = cv2.imencode(".png", frame)
    if not ok:
        return False
    try:
        out_path.write_bytes(buffer.tobytes())
        return True
    except OSError:
        return False


def extract_frames(
    video_path: Path,
    out_dir: Path,
    seconds: list[int],
    frame_interval: int,
    video_index: int,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_dir.mkdir(parents=True, exist_ok=True)
    base = safe_stem(video_path, video_index)

    targets = sorted(set(int(s) for s in seconds))
    if not targets:
        cap.release()
        return

    next_idx = 0
    frame_idx = 0
    last_sample = None  # (timestamp, frame, frame_idx)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps

            while next_idx < len(targets) and timestamp >= targets[next_idx]:
                target_sec = targets[next_idx]
                chosen_frame = frame
                chosen_ts = timestamp
                chosen_idx = frame_idx
                if last_sample is not None:
                    last_ts, last_frame, last_idx = last_sample
                    if abs(target_sec - last_ts) <= abs(timestamp - target_sec):
                        chosen_frame = last_frame
                        chosen_ts = last_ts
                        chosen_idx = last_idx

                out_path = out_dir / (
                    f"{base}_{target_sec:02d}s_f{chosen_idx}_t{chosen_ts:.3f}s.png"
                )
                wrote = save_png(out_path, chosen_frame)
                if wrote:
                    print(
                        f"Saved: {out_path} (target={target_sec}s, "
                        f"chosen_frame={chosen_idx}, chosen_time={chosen_ts:.3f}s)"
                    )
                else:
                    print(f"Failed to write: {out_path}")
                next_idx += 1

            last_sample = (timestamp, frame, frame_idx)

            if next_idx >= len(targets):
                break

        frame_idx += 1

    if next_idx < len(targets):
        for target_sec in targets[next_idx:]:
            print(f"Skip {video_path} @ {target_sec}s (beyond end)")

    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract fixed-time frames from demo videos (no pose, no RULA)."
    )
    parser.add_argument(
        "--input",
        default="demo_videos",
        help="Folder containing video files.",
    )
    parser.add_argument(
        "--output",
        default="demo_videos/frames",
        help="Output folder for extracted frames.",
    )
    parser.add_argument(
        "--pattern",
        default="*.MOV",
        help="Glob pattern for input files.",
    )
    parser.add_argument(
        "--seconds",
        type=parse_seconds,
        default=DEFAULT_SECONDS,
        help="Comma-separated list of seconds, e.g. 15,25,35,45,55.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=5,
        help="Frame interval setting to match system pipeline.",
    )

    args = parser.parse_args()
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")

    pattern = str(input_dir / args.pattern)
    files = sorted(Path(p) for p in glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files matched: {pattern}")

    output_dir = Path(args.output)
    for idx, video_path in enumerate(files, start=1):
        try:
            print(f"Processing: {video_path}")
            grouped_output_dir = categorize_output_dir(output_dir, video_path)
            extract_frames(
                video_path,
                grouped_output_dir,
                args.seconds,
                args.frame_interval,
                idx,
            )
        except Exception as exc:
            print(f"Failed: {video_path} ({exc})")


if __name__ == "__main__":
    main()
