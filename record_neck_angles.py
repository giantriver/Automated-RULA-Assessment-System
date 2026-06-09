"""
離線記錄影片的脖子角度資料。

預設會分析 `demo_videos/箱子1.MOV`，以 frame interval = 5 取樣，
並輸出左右脖子角度到 CSV。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rula_realtime_app.core.config import POSE_BACKEND  # noqa: E402
from rula_realtime_app.core.pose_detector import PoseDetector  # noqa: E402
from rula_realtime_app.core.rula_calculator import angle_calc  # noqa: E402


DEFAULT_VIDEO = PROJECT_ROOT / "demo_videos" / "箱子1.MOV"


def analyze_video(video_path: Path, backend: str, frame_interval: int, max_sampled_frames: int | None = None):
    detector = PoseDetector(backend_mode=backend)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        detector.close()
        raise FileNotFoundError(f"無法開啟影片檔案：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    records: list[dict[str, object]] = []
    frame_idx = 0
    sampled_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                sampled_count += 1
                timestamp_sec = round(frame_idx / fps, 3)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if detector.process_frame(rgb):
                    pose = detector.get_landmarks_array()
                    left_rula, right_rula = angle_calc(pose)
                else:
                    left_rula, right_rula = {"neck_angle": "NULL", "score": "NULL"}, {"neck_angle": "NULL", "score": "NULL"}

                records.append({
                    "frame": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "left_neck_angle": left_rula.get("neck_angle", "NULL"),
                    "left_neck_score": left_rula.get("score", "NULL"),
                    "right_neck_angle": right_rula.get("neck_angle", "NULL"),
                    "right_neck_score": right_rula.get("score", "NULL"),
                })

                if max_sampled_frames is not None and sampled_count >= max_sampled_frames:
                    break

            frame_idx += 1

    finally:
        cap.release()
        detector.close()

    return records, fps


def write_csv(records: list[dict[str, object]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame",
        "timestamp_sec",
        "left_neck_angle",
        "left_neck_score",
        "right_neck_angle",
        "right_neck_score",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_args():
    parser = argparse.ArgumentParser(description="記錄影片中的脖子角度資料")
    parser.add_argument(
        "video_path",
        nargs="?",
        default=str(DEFAULT_VIDEO),
        help="要分析的影片路徑，預設為 demo_videos/箱子1.MOV",
    )
    parser.add_argument(
        "--backend",
        choices=["MEDIAPIPE", "RTMW3D"],
        default=POSE_BACKEND,
        help="姿勢偵測後端，預設使用目前系統設定",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=5,
        help="每隔幾幀取樣一次，預設 5",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV 輸出路徑，預設輸出到影片同資料夾下的 *_neck_angles.csv",
    )
    parser.add_argument(
        "--max-sampled-frames",
        type=int,
        default=None,
        help="最多只記錄幾個取樣幀，方便快速測試",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"找不到影片：{video_path}")

    output_path = Path(args.output) if args.output else video_path.with_name(f"{video_path.stem}_neck_angles.csv")

    records, fps = analyze_video(
        video_path=video_path,
        backend=args.backend,
        frame_interval=max(1, args.frame_interval),
        max_sampled_frames=args.max_sampled_frames,
    )
    write_csv(records, output_path)

    print(f"影片：{video_path}")
    print(f"後端：{args.backend}")
    print(f"FPS：{fps:.3f}")
    print(f"取樣筆數：{len(records)}")
    print(f"輸出：{output_path}")


if __name__ == "__main__":
    main()