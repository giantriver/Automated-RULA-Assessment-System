"""
配置檔案 - 集中管理所有設定參數
"""

import os
from pathlib import Path

import numpy as np

# === 姿勢辨識後端選擇 ===
# - "MEDIAPIPE": MediaPipe
# - "RTMW3D": RTMW3D
POSE_BACKEND = "RTMW2D"  # choices: "MEDIAPIPE", "RTMW2D", "RTMW3D"

# 一般網路攝影機索引
WEBCAM_INDEX = 0

# RULA angle coordinate mode:
# - "2D": use image pixel coordinates for MediaPipe pose_landmarks and RTMPoseW 2D keypoints.
# - "3D": use MediaPipe pose_world_landmarks only.
ANALYSIS_MODE = "2D"

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

# RULA 實際分析/使用的骨架連線（RTMW 原生 COCO-WholeBody 索引）。
# rtmlib 預設會畫完整 133 點（含臉、雙手 mesh、腳），但其中只有以下這些點會被
# 映射進 33 點並參與 RULA 角度與異常判定。為了「所見即所判」，繪製時只畫這些連線，
# 避免畫面出現未被分析的指尖/臉部點造成誤導。
RTMW_RULA_CONNECTIONS = [
    (RTMW["LEFT_EAR"],  RTMW["RIGHT_EAR"]),                        # 雙耳
    (RTMW["LEFT_SHOULDER"],  RTMW["LEFT_EAR"]),
    (RTMW["RIGHT_SHOULDER"], RTMW["RIGHT_EAR"]),                   # 肩 → 耳
    (RTMW["LEFT_SHOULDER"],  RTMW["RIGHT_SHOULDER"]),             # 雙肩
    (RTMW["LEFT_SHOULDER"],  RTMW["LEFT_HIP"]),
    (RTMW["RIGHT_SHOULDER"], RTMW["RIGHT_HIP"]),
    (RTMW["LEFT_HIP"],       RTMW["RIGHT_HIP"]),                  # 軀幹
    (RTMW["LEFT_SHOULDER"],  RTMW["LEFT_ELBOW"]),
    (RTMW["LEFT_ELBOW"],     RTMW["LEFT_WRIST"]),
    (RTMW["LEFT_WRIST"],     RTMW["LEFT_MIDDLE_FINGER1"]),        # 左臂 + 左手
    (RTMW["RIGHT_SHOULDER"], RTMW["RIGHT_ELBOW"]),
    (RTMW["RIGHT_ELBOW"],    RTMW["RIGHT_WRIST"]),
    (RTMW["RIGHT_WRIST"],    RTMW["RIGHT_MIDDLE_FINGER1"]),       # 右臂 + 右手
]
# 上述連線涉及的所有關鍵點（畫節點用）
RTMW_RULA_KEYPOINTS = sorted({idx for pair in RTMW_RULA_CONNECTIONS for idx in pair})

# MediaPipe BlazePose 33 點 — 只列出 RULA 角度計算使用的連線（所見即所判）
MP_RULA_CONNECTIONS: list[tuple[int, int]] = [
    ( 7,  8),  # 雙耳
    (11, 12),  # 雙肩
    (11, 23), (12, 24), (23, 24),  # 軀幹
    (11, 13), (13, 15),            # 左臂
    (12, 14), (14, 16),            # 右臂
    (15, 17), (15, 19),            # 左手（pinky / index）
    (16, 18), (16, 20),            # 右手
    (11,  7), (12,  8),            # 肩 → 耳
]
MP_RULA_KEYPOINTS: list[int] = sorted({idx for pair in MP_RULA_CONNECTIONS for idx in pair})

# 骨段名稱 → 原生索引對，供骨長異常紅線繪製使用。
# RTMW 版本用 COCO-WholeBody 原始索引（keypoints_2d_norm）。
BONE_NAME_TO_RTMW_PAIR: dict[str, tuple[int, int]] = {
    # 上肢鏈
    'left_upper_arm':    (RTMW["LEFT_SHOULDER"],  RTMW["LEFT_ELBOW"]),          # (5, 7)
    'left_lower_arm':    (RTMW["LEFT_ELBOW"],     RTMW["LEFT_WRIST"]),          # (7, 9)
    'right_upper_arm':   (RTMW["RIGHT_SHOULDER"], RTMW["RIGHT_ELBOW"]),         # (6, 8)
    'right_lower_arm':   (RTMW["RIGHT_ELBOW"],    RTMW["RIGHT_WRIST"]),         # (8, 10)
    'left_wrist_index':  (RTMW["LEFT_WRIST"],  RTMW["LEFT_MIDDLE_FINGER1"]),    # (9, 100)
    'right_wrist_index': (RTMW["RIGHT_WRIST"], RTMW["RIGHT_MIDDLE_FINGER1"]),   # (10, 121)
    # 軀幹核心
    'shoulder_width':    (RTMW["LEFT_SHOULDER"],  RTMW["RIGHT_SHOULDER"]),      # (5, 6)
    'hip_width':         (RTMW["LEFT_HIP"],        RTMW["RIGHT_HIP"]),           # (11, 12)
    'left_trunk':        (RTMW["LEFT_SHOULDER"],   RTMW["LEFT_HIP"]),            # (5, 11)
    'right_trunk':       (RTMW["RIGHT_SHOULDER"],  RTMW["RIGHT_HIP"]),           # (6, 12)
    # 頭頸鏈
    'left_shoulder_ear':  (RTMW["LEFT_SHOULDER"],  RTMW["LEFT_EAR"]),            # (5, 3)
    'right_shoulder_ear': (RTMW["RIGHT_SHOULDER"], RTMW["RIGHT_EAR"]),           # (6, 4)
}

# MediaPipe 版本用 BlazePose 33 點索引（landmarks_2d）。
BONE_NAME_TO_MP_PAIR: dict[str, tuple[int, int]] = {
    # 上肢鏈
    'left_upper_arm':    (11, 13),
    'left_lower_arm':    (13, 15),
    'right_upper_arm':   (12, 14),
    'right_lower_arm':   (14, 16),
    'left_wrist_index':  (15, 19),
    'left_wrist_pinky':  (15, 17),
    'right_wrist_index': (16, 20),
    'right_wrist_pinky': (16, 18),
    # 軀幹核心
    'shoulder_width':    (11, 12),
    'hip_width':         (23, 24),
    'left_trunk':        (11, 23),
    'right_trunk':       (12, 24),
    # 頭頸鏈
    'left_shoulder_ear':  (11,  7),
    'right_shoulder_ear': (12,  8),
}

# 將 RTMW 索引映射到 MediaPipe Pose 33 索引
# RULA 核心使用的是 MediaPipe 33 點語意，故 RTMW3D 先轉成相同格式。
RTMW_TO_MEDIAPIPE = {
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

