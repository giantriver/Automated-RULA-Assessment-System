"""
配置檔案 - 集中管理所有設定參數
"""

import os
from pathlib import Path

import numpy as np

# === 姿勢辨識後端選擇 ===
# - "MEDIAPIPE": MediaPipe
# - "RTMW3D": RTMW3D
POSE_BACKEND = "RTMW3D"  # 可選: "MEDIAPIPE", "RTMW3D"

# 一般網路攝影機索引
WEBCAM_INDEX = 0

# === 顯示模式選擇 ===
DISPLAY_MODE = "RULA"  # "RULA": 顯示RULA評估分數; "COORDINATES": 顯示關鍵點坐標

# RULA 固定參數設定
RULA_CONFIG = {
    'wrist_twist': 1,        # 手腕扭轉參數
    'legs': 2,               # 腿部姿勢參數
    'muscle_use_a': 0,         # Table A 肌肉使用參數
    'muscle_use_b': 0,         # Table B 肌肉使用參數
    'force_load_a': 0,       # Table A 負荷力量參數
    'force_load_b': 0,       # Table B 負荷力量參數
}

# 角度計算參數
TOLERANCE_ANGLE = 5.0        # 容忍角度（度）
MIN_CONFIDENCE = 0.5         # 最小置信度閾值
USE_PREVIOUS_FRAME_ON_LOW_CONFIDENCE = False  # 低置信度處理策略

# MediaPipe 設定（即時辨識優化）
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'model_complexity': 1,      # 改為 0（最輕量模型，提升速度）
    'smooth_landmarks': True,
    'enable_segmentation': False,
    'smooth_segmentation': False,  # 關閉分割功能
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# RTMW3D 設定（使用一般網路攝影機）
RTMW3D_CONFIG = {
    'backend': 'onnxruntime',
    'device': 'cuda',         # 若 CUDA 不可用，程式會自動降級為 CPU
    'det_frequency': 1,     # 每 10 幀才跑一次人體偵測，其餘幀用 tracking（大幅提升 FPS）
    'tracking': False,         # 開啟 tracking 搭配高 det_frequency 才有意義
    'iou_threshold': 0.05,    # 單目標鎖定 IoU 閾值
    'kpt_threshold': 0.3      # 繪製骨架閾值
}

# RTMW（COCO-WholeBody 133）關鍵點索引（只列出 RULA 需要的點）
RTMW = {
    "NOSE": 0,
    "LEFT_EAR": 3,
    "RIGHT_EAR": 4,
    "LEFT_SHOULDER": 5,
    "RIGHT_SHOULDER": 6,
    "LEFT_ELBOW": 7,
    "RIGHT_ELBOW": 8,
    "LEFT_WRIST": 9,
    "RIGHT_WRIST": 10,
    "LEFT_MIDDLE_FINGER1": 100,
    "RIGHT_MIDDLE_FINGER1": 121,
    "LEFT_HIP": 11,
    "RIGHT_HIP": 12,
}

# 將 RTMW 索引映射到 MediaPipe Pose 33 索引
# RULA 核心使用的是 MediaPipe 33 點語意，故 RTMW3D 先轉成相同格式。
RTMW_TO_MEDIAPIPE = {
    0: RTMW["NOSE"],
    7: RTMW["LEFT_EAR"],
    8: RTMW["RIGHT_EAR"],
    11: RTMW["LEFT_SHOULDER"],
    12: RTMW["RIGHT_SHOULDER"],
    13: RTMW["LEFT_ELBOW"],
    14: RTMW["RIGHT_ELBOW"],
    15: RTMW["LEFT_WRIST"],
    16: RTMW["RIGHT_WRIST"],
    # 讓 RTMW3D 的手腕角度使用 wrist -> middle_finger1 方向。
    # rula_calculator 的 HAND_C = (INDEX + PINKY) / 2，因此將兩者都映射到 middle_finger1。
    17: RTMW["LEFT_MIDDLE_FINGER1"],
    18: RTMW["RIGHT_MIDDLE_FINGER1"],
    19: RTMW["LEFT_MIDDLE_FINGER1"],
    20: RTMW["RIGHT_MIDDLE_FINGER1"],
    23: RTMW["LEFT_HIP"],
    24: RTMW["RIGHT_HIP"],
}


def convert_indexed_keypoints_to_pose33(keypoints_xyz, keypoint_scores, index_map):
    """
    通用索引轉換機制：將任意來源關鍵點陣列轉為 MediaPipe-like 33 點格式。

    Args:
        keypoints_xyz: 來源關鍵點，shape 約為 [K, 3+]，至少包含 x, y, z。
        keypoint_scores: 來源分數，shape 約為 [K]。
        index_map: 目標索引 -> 來源索引 的映射字典。

    Returns:
        list: 33 個關鍵點，每個為 [x, y, z, conf]
    """
    pose = [[0.0, 0.0, 0.0, 0.0] for _ in range(33)]

    if keypoints_xyz is None:
        return pose

    kpts = np.asarray(keypoints_xyz)
    if kpts.ndim != 2 or kpts.shape[1] < 3:
        return pose

    if keypoint_scores is None:
        scores = np.ones((kpts.shape[0],), dtype=np.float32)
    else:
        scores = np.asarray(keypoint_scores).reshape(-1)

    for dst_idx, src_idx in index_map.items():
        if src_idx >= kpts.shape[0]:
            continue

        x, y, z = kpts[src_idx][:3]
        conf = float(scores[src_idx]) if src_idx < scores.shape[0] else 0.0
        conf = max(0.0, min(1.0, conf))

        pose[dst_idx] = [float(x), float(y), float(z), conf]

    return pose


def _register_cuda_dll_dirs() -> None:
    """Register Windows DLL search paths for CUDA/cuDNN packaged in the venv."""
    if os.name != 'nt':
        return

    try:
        import site
    except Exception:
        return

    pkg_root = Path(site.getsitepackages()[-1]) / 'nvidia'
    candidate_dirs = [
        pkg_root / 'cudnn' / 'bin',
        pkg_root / 'cublas' / 'bin',
        pkg_root / 'cuda_nvrtc' / 'bin',
    ]

    for env_name in ('CUDA_PATH', 'CUDA_PATH_V12_4'):
        cuda_root = os.environ.get(env_name)
        if cuda_root:
            candidate_dirs.append(Path(cuda_root) / 'bin')

    add_dll_directory = getattr(os, 'add_dll_directory', None)
    if add_dll_directory is None:
        return

    existing_path = os.environ.get('PATH', '')
    path_parts = existing_path.split(os.pathsep) if existing_path else []

    for dll_dir in candidate_dirs:
        if dll_dir.exists():
            try:
                add_dll_directory(str(dll_dir))
            except OSError:
                pass

            dll_dir_str = str(dll_dir)
            if dll_dir_str not in path_parts:
                path_parts.insert(0, dll_dir_str)

    os.environ['PATH'] = os.pathsep.join(path_parts)


_register_cuda_dll_dirs()

