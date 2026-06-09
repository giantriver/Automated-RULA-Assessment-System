from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2


def run(cmd: list[str]) -> None:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"{' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def trim_one_with_opencv(input_path: Path, duration: int, overwrite: bool) -> None:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        capture.release()
        raise RuntimeError(f"Could not determine FPS for: {input_path}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise RuntimeError(f"Could not determine frame size for: {input_path}")

    max_frames = max(1, int(duration * fps))

    if overwrite:
        with tempfile.NamedTemporaryFile(
            dir=str(input_path.parent), suffix=input_path.suffix, delete=False
        ) as tmp:
            temp_path = Path(tmp.name)
    else:
        temp_path = input_path

    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open output file for writing: {temp_path}")

    frame_count = 0
    try:
        while frame_count < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            frame_count += 1
    finally:
        capture.release()
        writer.release()

    if overwrite:
        os.replace(temp_path, input_path)


def trim_one(input_path: Path, duration: int, remove_audio: bool, overwrite: bool) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        if remove_audio:
            print("Warning: --remove-audio was requested, but ffmpeg is unavailable. Audio will be kept only if the fallback codec writes it.")
        trim_one_with_opencv(input_path, duration=duration, overwrite=overwrite)
        return

    if overwrite:
        # Write to a temp file in the same directory, then replace the original.
        with tempfile.NamedTemporaryFile(
            dir=str(input_path.parent), suffix=input_path.suffix, delete=False
        ) as tmp:
            temp_path = Path(tmp.name)
    else:
        temp_path = input_path

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-t",
        str(duration),
        "-c",
        "copy",
    ]
    if remove_audio:
        cmd.append("-an")

    cmd.append(str(temp_path))

    run(cmd)

    if overwrite:
        os.replace(temp_path, input_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim MOV files to first N seconds.")
    parser.add_argument(
        "--input",
        default="demo_videos",
        help="Folder containing MOV files.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=70,
        help="Duration in seconds to keep from the start.",
    )
    parser.add_argument(
        "--pattern",
        default="*.MOV",
        help="Glob pattern for input files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original files in-place.",
    )
    parser.add_argument(
        "--remove-audio",
        action="store_true",
        help="Remove audio stream.",
    )

    args = parser.parse_args()
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")

    pattern = str(input_dir / args.pattern)
    files = sorted(Path(p) for p in glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files matched: {pattern}")

    for file_path in files:
        trim_one(
            file_path,
            duration=args.duration,
            remove_audio=args.remove_audio,
            overwrite=args.overwrite,
        )
        print(f"Trimmed: {file_path}")


if __name__ == "__main__":
    main()
