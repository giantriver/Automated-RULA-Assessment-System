from __future__ import annotations

import argparse
import glob
import os
import subprocess
import tempfile
from pathlib import Path


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


def trim_one(input_path: Path, duration: int, remove_audio: bool, overwrite: bool) -> None:
    if overwrite:
        # Write to a temp file in the same directory, then replace the original.
        with tempfile.NamedTemporaryFile(
            dir=str(input_path.parent), suffix=input_path.suffix, delete=False
        ) as tmp:
            temp_path = Path(tmp.name)
    else:
        temp_path = input_path

    cmd = [
        "ffmpeg",
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
