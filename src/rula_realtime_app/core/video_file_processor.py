"""
影片檔案處理器 - 在背景執行緒中對影片檔案進行 RULA 離線分析
"""

import cv2
import csv
import json
import os
import statistics
import time
from datetime import datetime

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from .pose_detector import PoseDetector
from . import angle_calc
from .utils import get_best_rula_score
from .config import RULA_CONFIG, RTMW_TO_MEDIAPIPE


def _score_num(rula_left, rula_right):
    """由左右側 RULA 結果取最佳分數（int），無法計算時回傳 None。"""
    best = get_best_rula_score(rula_left, rula_right)
    s = best.get('final_tableC_score', 'NULL')
    try:
        return int(s) if s != 'NULL' else None
    except (ValueError, TypeError):
        return None

# ── Anomaly detection constants ─────────────────────────────────────────────
# 異常判定設計（簡化版）：
#   2D 分析：僅用 confidence（visibility）→ 只有 low_visibility 會讓關節 invalid。
#   3D 分析：confidence + bone-length → 另含 bone_length_abnormal。
#   速度：不作為異常判定，不影響角度計算；僅在 UI 以三角形提示快速移動的關節。
# 預設值，未來透過實驗校正。
_ANOM_VIS_TH       = 0.50   # visibility 低於此值 → 直接不可靠
_ANOM_MAX_GAP_SECONDS = 1.0  # 速度估計可接受的最大觀測間隔（秒）
_ANOM_MIN_JUMP_RATIO = 0.5  # 自適應閥值的保護下限（避免低動作影片 threshold collapse）

# ── 骨段長度異常門檻（current / ref 落在區間外即異常；僅 3D 分析使用）──────
# body：上臂 / 前臂 / 軀幹，剛性高，門檻較緊。
# hand：腕→指，短且抖動大，門檻較鬆。
# head：肩→耳，頭部旋轉投影變化大，門檻最寬。
_BONE_BODY_LOW,  _BONE_BODY_HIGH = 0.6, 1.4
_BONE_HAND_LOW,  _BONE_HAND_HIGH = 0.5, 1.8
_BONE_HEAD_LOW,  _BONE_HEAD_HIGH = 0.3, 2.5

_BONE_THRESH: dict[str, tuple[float, float]] = {
    'body': (_BONE_BODY_LOW,  _BONE_BODY_HIGH),
    'hand': (_BONE_HAND_LOW,  _BONE_HAND_HIGH),
    'head': (_BONE_HEAD_LOW,  _BONE_HEAD_HIGH),
}

# ── 線性插值補點（Interpolation）─────────────────────────────────────────────
# 目的：修復「短暫」關節追蹤失敗，而非預測長時間遮擋。
# 以前後最近的可信幀對 invalid 關節做線性插值，依時間比例補回位置後重算 RULA。
# 兩端可信幀的時間差（gap）超過此上限就不補，維持 invalid。
_INTERP_MAX_GAP_SECONDS = 1.0

# 只對 RULA 角度實際會用到的關節補點（避免在未使用點上產生多餘的藍色圈）。
_INTERP_JOINTS: tuple[int, ...] = (
    7, 8,            # 左右耳（neck angle）
    11, 12,          # 左右肩
    13, 14,          # 左右肘
    15, 16,          # 左右腕
    17, 18, 19, 20,  # 手指（pinky / index）
    23, 24,          # 左右髖
)

# MediaPipe 33 索引 → 補點統計顯示名稱
_INTERP_JOINT_NAMES: dict[int, str] = {
    7: 'Left Ear',   8: 'Right Ear',
    11: 'Left Shoulder', 12: 'Right Shoulder',
    13: 'Left Elbow',    14: 'Right Elbow',
    15: 'Left Wrist',    16: 'Right Wrist',
    17: 'Left Pinky',    18: 'Right Pinky',
    19: 'Left Index',    20: 'Right Index',
    23: 'Left Hip',      24: 'Right Hip',
}

# ── 骨段定義（MediaPipe 33 點索引）──────────────────────────────────────────
# 'pair'       = (近端, 遠端) 兩端點索引
# 'invalidate' = 異常時要標為不可靠的關節索引
# 'thresh'     = 使用哪組門檻（'body' / 'hand' / 'head'）
#
# 設計原則：「遠端標記」— 骨段異常時標記遠端（distal）關節，而非兩端。
#   上肢鏈 shoulder→elbow→wrist→finger：每段只標遠端。
#     好處：elbow 異常不連帶讓 upper_arm NULL；wrist 異常不讓 lower_arm NULL；
#           finger 異常只影響 wrist_angle，不影響 lower_arm angle。
#   軀幹核心（shoulder-shoulder / hip-hip / shoulder-hip）：
#     不標記任何關節，只畫紅色骨段。
#     shoulder / hip 同時用於 neck / trunk angle，不該因寬度異常就讓角度全 NULL。
#   頭頸鏈 shoulder→ear：只標遠端（ear）。
_LIMB_BONES: dict[str, dict] = {
    # 上臂 / 前臂：只標遠端
    'left_upper_arm':    {'pair': (11, 13), 'invalidate': (13,), 'thresh': 'body'},
    'left_lower_arm':    {'pair': (13, 15), 'invalidate': (15,), 'thresh': 'body'},
    'right_upper_arm':   {'pair': (12, 14), 'invalidate': (14,), 'thresh': 'body'},
    'right_lower_arm':   {'pair': (14, 16), 'invalidate': (16,), 'thresh': 'body'},
    # 手部：只標指端，wrist 保持可靠 → lower_arm angle 仍可計算
    'left_wrist_index':  {'pair': (15, 19), 'invalidate': (19,), 'thresh': 'hand'},
    'left_wrist_pinky':  {'pair': (15, 17), 'invalidate': (17,), 'thresh': 'hand'},
    'right_wrist_index': {'pair': (16, 20), 'invalidate': (20,), 'thresh': 'hand'},
    'right_wrist_pinky': {'pair': (16, 18), 'invalidate': (18,), 'thresh': 'hand'},
}
_CORE_BONES: dict[str, dict] = {
    # 軀幹核心：不標記任何關節，只畫紅色骨段
    'shoulder_width': {'pair': (11, 12), 'invalidate': (), 'thresh': 'body'},
    'hip_width':      {'pair': (23, 24), 'invalidate': (), 'thresh': 'body'},
    'left_trunk':     {'pair': (11, 23), 'invalidate': (), 'thresh': 'body'},
    'right_trunk':    {'pair': (12, 24), 'invalidate': (), 'thresh': 'body'},
}
_HEAD_BONES: dict[str, dict] = {
    # 頭頸鏈：只標遠端（ear）
    'left_shoulder_ear':  {'pair': (11,  7), 'invalidate': ( 7,), 'thresh': 'head'},
    'right_shoulder_ear': {'pair': (12,  8), 'invalidate': ( 8,), 'thresh': 'head'},
}
_ALL_BONES: dict[str, dict] = {**_LIMB_BONES, **_CORE_BONES, **_HEAD_BONES}


def _active_bones(backend_mode: str) -> dict:
    """
    依 backend 選擇要檢查的骨段。

    RTMW 的 RTMW_TO_MEDIAPIPE 把 pinky/index（17/18/19/20）全部映到單一
    MIDDLE_FINGER1，因此 RTMW 手部只用 wrist→index 一條，避免四個疊在同一像素。
    MediaPipe 保留 index + pinky 兩條（wrist_angle HAND_C 所用）。
    軀幹核心（_CORE_BONES）與頭頸鏈（_HEAD_BONES）兩個 backend 均使用。
    """
    backend = str(backend_mode or '').upper()
    arm_names = ('left_upper_arm', 'left_lower_arm',
                 'right_upper_arm', 'right_lower_arm')
    bones = {k: _LIMB_BONES[k] for k in arm_names}
    if backend in ('RTMW2D', 'RTMW3D'):
        hand_names = ('left_wrist_index', 'right_wrist_index')
    else:
        hand_names = ('left_wrist_index', 'left_wrist_pinky',
                      'right_wrist_index', 'right_wrist_pinky')
    for name in hand_names:
        bones[name] = _LIMB_BONES[name]
    bones.update(_CORE_BONES)
    bones.update(_HEAD_BONES)
    return bones

# 建立速度分布的 visibility gate。
# 必須與判定階段 (_ANOM_VIS_TH) 使用同一道門檻，否則門檻會建立在與實際評估
# 不同的關節母體上。實測 RTMPose 的 keypoint 信心分數多落在 [0.5, 0.8)，用
# 0.80 會收集到「零樣本」→ body_scale_ref=0、自適應門檻全數失效並退回未校正的
# 靜態下限，導致幾乎每一幀都被誤判為速度異常。
_THRESHOLD_VIS_HIGH = _ANOM_VIS_TH

# MediaPipe 33 點關節群組（用於建立各群組速度分布）
_JOINT_GROUPS: dict[str, list[int]] = {
    'trunk': [11, 12, 23, 24],          # 左右肩、左右髖
    'head':  [0, 7, 8],                 # 鼻子、左右耳
    'arm':   [13, 14],                  # 左右肘
    'hand':  [15, 16, 17, 18, 19, 20],  # 左右腕、手指點
}

# 每個關節屬於哪個群組（反查表）
_JOINT_TO_GROUP: dict[int, str] = {
    jidx: grp
    for grp, idxs in _JOINT_GROUPS.items()
    for jidx in idxs
}


def _bone_length(arr, a: int, b: int) -> float | None:
    """回傳兩端關節皆通過 visibility gate 時的骨段長度，否則 None。"""
    la, lb = arr[a], arr[b]
    if float(la[3]) < _ANOM_VIS_TH or float(lb[3]) < _ANOM_VIS_TH:
        return None
    length = (
        (float(la[0]) - float(lb[0])) ** 2 +
        (float(la[1]) - float(lb[1])) ** 2 +
        (float(la[2]) - float(lb[2])) ** 2
    ) ** 0.5
    return length if length > 1e-9 else None


def _bone_anomaly_for_frame(arr, bone_ref: dict) -> tuple[set[int], dict[str, float]]:
    """
    以影片內骨長基準 (bone_ref) 判斷單幀的骨段長度異常。

    Returns:
        invalid_joints (set[int]):       需標記為不可靠的關節索引（angle gating 用）
        abnormal (dict[str, float]):     異常骨段 {bone_name: ratio}（紅線繪製用）
    """
    invalid_joints: set[int] = set()
    abnormal: dict[str, float] = {}
    for name, spec in _ALL_BONES.items():
        ref = bone_ref.get(name)
        if not ref or ref <= 1e-9:
            continue
        a, b = spec['pair']
        length = _bone_length(arr, a, b)
        if length is None:
            continue
        ratio = length / ref
        lo, hi = _BONE_THRESH[spec.get('thresh', 'body')]
        if ratio < lo or ratio > hi:
            abnormal[name] = round(ratio, 4)
            invalid_joints.update(spec['invalidate'])
    return invalid_joints, abnormal


def _compute_anomaly_mask(landmarks_arr, prev_reliable, body_scale, dt, current_frame_idx,
                          frame_interval: int,
                          group_thresholds: dict | None = None,
                          bone_ref: dict | None = None):
    """
    判斷 MediaPipe 33 個關節點是否可靠（非異常）。

    異常類型：
      - low_visibility：visibility < _ANOM_VIS_TH → invalid（2D / 3D 皆適用）
      - bone_length_abnormal：骨段長度相對於影片內基準異常 → invalid（僅 3D，bone_ref 非 None 時）
      - speed_candidate：speed_ratio > 全片自適應門檻 → 永不 invalid，僅供 UI 三角形提示

    Args:
        landmarks_arr:    33 × [x, y, z, vis]，MediaPipe world coordinates
        prev_reliable:    list[dict|None]，每個關節上一個可靠幀的 {pos, frame_idx}
        body_scale:       人體尺度參考（肩寬），用來正規化速度
        dt:               兩個分析幀之間的實際時間差（秒）
        current_frame_idx: 目前分析幀索引，用來計算 gap_seconds
        frame_interval:    抽樣間隔（幀）
        group_thresholds: {group_name: th_speed}，由 Pass 1 計算；None 時不做速度候選
        bone_ref:         {bone_name: median_length}，由 _compute_bone_ref 計算；None 時不做骨長檢查

    Returns:
        mask (list[bool]):            True = 可靠，False = 疑似異常（此處僅含 low_vis / bone）
        new_prev (list[dict|None]):   更新後的 prev_reliable（只更新可靠且非候選的關節）
        detail (list[dict]):          {'reason','reasons','visibility','speed_ratio','speed_checked','th_speed'}
        bone_anomaly (dict[str,float]): 異常骨段 {bone_name: ratio}（空 dict = 無骨長異常）
    """
    mask = []
    new_prev = list(prev_reliable)
    detail: list = []

    bone_invalid: set[int] = set()
    bone_anomaly: dict[str, float] = {}
    if bone_ref:
        bone_invalid, bone_anomaly = _bone_anomaly_for_frame(landmarks_arr, bone_ref)

    for i, lm in enumerate(landmarks_arr):
        x, y, z, vis = float(lm[0]), float(lm[1]), float(lm[2]), float(lm[3])
        grp = _JOINT_TO_GROUP.get(i)
        th_speed = group_thresholds.get(grp) if group_thresholds else None
        reasons: list[str] = []
        speed_ratio: float | None = None
        speed_checked = False

        low_visibility = vis < _ANOM_VIS_TH
        bone_abnormal  = i in bone_invalid
        if low_visibility:
            reasons.append('low_visibility')
        if bone_abnormal:
            reasons.append('bone_length_abnormal')

        # 速度候選：低可見度時沒有可信位置可估速，直接略過
        if not low_visibility:
            prev = prev_reliable[i]
            prev_pos = prev.get('pos') if isinstance(prev, dict) else None
            prev_frame_idx = prev.get('frame_idx') if isinstance(prev, dict) else None

            elapsed_seconds = dt
            if prev_frame_idx is not None and prev_frame_idx >= 0:
                elapsed_seconds = ((current_frame_idx - prev_frame_idx) / max(1, frame_interval)) * dt

            if grp is not None and th_speed is not None and prev_pos is not None and elapsed_seconds > 1e-9 and body_scale > 1e-6:
                gap_seconds = elapsed_seconds
                if gap_seconds <= _ANOM_MAX_GAP_SECONDS:
                    jump = ((x - prev_pos[0])**2 + (y - prev_pos[1])**2 + (z - prev_pos[2])**2) ** 0.5
                    speed_ratio = (jump / gap_seconds) / body_scale
                    speed_checked = True

            if th_speed is not None and speed_ratio is not None and speed_ratio > th_speed:
                reasons.append('speed_candidate')

        # invalid = low_visibility 或 bone_length_abnormal；speed_candidate 永不 invalid
        reliable = not (low_visibility or bone_abnormal)
        reason = 'low_visibility' if low_visibility else ('bone_length_abnormal' if bone_abnormal else None)

        mask.append(reliable)
        # 只有「可靠且非速度候選」的位置才能當作後續估速的基準，避免 glitch 汙染 prev
        if reliable and 'speed_candidate' not in reasons:
            new_prev[i] = {'pos': [x, y, z], 'frame_idx': current_frame_idx}
        detail.append({
            'reason':        reason,
            'reasons':       reasons,
            'visibility':    round(vis, 4),
            'speed_ratio':   round(speed_ratio, 4) if speed_ratio is not None else None,
            'speed_checked': speed_checked and th_speed is not None,
            'th_speed':      round(th_speed, 4) if th_speed is not None else None,
        })

    return mask, new_prev, detail, bone_anomaly


class VideoFileProcessor(QObject):
    """
    離線影片 RULA 分析工作器（在 QThread 中執行）

    流程分兩個階段：
      1. 推論階段（_inference_pass）：唯一一次解碼 + 姿勢推論，快取每個取樣幀的
         landmarks 與繪圖資料，同時收集肩寬 / 群組速度以建立自適應門檻。
      2. 計算階段（_build_records）：純 CPU，對快取的取樣幀做異常判定、角度計算與
         記錄組裝。不再重複解碼或推論，避免推論成本加倍，也避免 detector 跨階段
         追蹤狀態互相污染。

    Signals:
        progress_updated(int, str): 進度百分比 + 狀態訊息
        frame_preview(np.ndarray): 目前處理的畫面（RGB）
        analysis_complete(dict): 完整分析結果
        error_occurred(str): 錯誤訊息
    """

    progress_updated = pyqtSignal(int, str)
    frame_preview    = pyqtSignal(object)   # np.ndarray
    analysis_complete = pyqtSignal(dict)
    error_occurred   = pyqtSignal(str)

    def __init__(self, video_path: str, meta: dict,
                 frame_interval: int = 10,
                 backend_mode: str = 'RTMW2D',
                 analysis_mode: str = '2D',
                 rula_params: dict | None = None,
                 enable_speed_anomaly: bool = True,
                 mp_model_complexity: int = 1,
                 use_interpolation: bool = False):
        """
        Args:
            video_path:     影片檔案路徑
            meta:           調查資訊 dict（survey_date, assessor, organization, task_name）
            frame_interval: 每隔幾幀取樣一次（預設 10）
            backend_mode:   姿勢偵測模式（'RTMW2D'、'RTMW3D' 或 'MEDIAPIPE'）
            analysis_mode:  '2D' 或 '3D'
            rula_params:    RULA 固定參數覆寫（wrist_twist, legs, muscle_use_a/b, force_load_a/b）
            enable_speed_anomaly: 是否啟用速度異常偵測（關閉時僅用可信度）
            mp_model_complexity: MediaPipe 模型複雜度（0/1/2，僅 MEDIAPIPE backend 使用）
            use_interpolation: 是否對短暫 invalid 關節做線性插值補點並以補點骨架重算 RULA
        """
        super().__init__()
        self.video_path     = video_path
        self.meta           = meta
        self.frame_interval = max(1, frame_interval)
        self.backend_mode   = backend_mode
        self.analysis_mode  = str(analysis_mode or '2D').upper()
        merged_params = dict(RULA_CONFIG)
        if isinstance(rula_params, dict):
            for key in ('wrist_twist', 'legs', 'muscle_use_a', 'muscle_use_b', 'force_load_a', 'force_load_b'):
                if key in rula_params:
                    merged_params[key] = int(rula_params[key])
        self.rula_params    = merged_params
        self.enable_speed_anomaly = bool(enable_speed_anomaly)
        self.mp_model_complexity  = int(mp_model_complexity)
        self.use_interpolation    = bool(use_interpolation)
        self._cancelled     = False

    def cancel(self):
        """請求取消處理"""
        self._cancelled = True

    @pyqtSlot()
    def run(self):
        """主處理流程（由 QThread.started 觸發）"""
        try:
            self._process()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))

    # ------------------------------------------------------------------
    def _process(self):
        analysis_started_at = time.perf_counter()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error_occurred.emit(f'無法開啟影片檔案：{self.video_path}')
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        self.progress_updated.emit(3, '初始化姿勢偵測模型...')

        original_rula_config = dict(RULA_CONFIG)
        RULA_CONFIG.update(self.rula_params)
        try:
            detector = PoseDetector(backend_mode=self.backend_mode,
                                    mp_model_complexity=self.mp_model_complexity)
            detector.analysis_mode = self.analysis_mode

            # 1) 推論階段：唯一一次解碼 + 推論，建立自適應速度門檻
            samples, body_scale_ref, group_thresholds = self._inference_pass(
                detector, fps, total_frames
            )
            if self._cancelled:
                return

            # 2) 計算階段：純 CPU，組裝記錄
            records = self._build_records(
                samples, fps, body_scale_ref, group_thresholds
            )
            if self._cancelled:
                return

            self.progress_updated.emit(97, '統計資料中...')
            analysis_duration_seconds = max(0.0, time.perf_counter() - analysis_started_at)
            results = self._build_results(
                records, total_frames, fps, analysis_duration_seconds, group_thresholds
            )
            self.progress_updated.emit(100, '分析完成！')
            self.analysis_complete.emit(results)
        finally:
            RULA_CONFIG.clear()
            RULA_CONFIG.update(original_rula_config)

    # ------------------------------------------------------------------
    def _inference_pass(self, detector, fps: float, total_frames: int):
        """
        唯一一次解碼 + 姿勢推論。

        - 快取每個取樣幀的 landmarks 與序列化繪圖資料供計算階段重用
        - 收集肩寬與各群組速度，於掃描結束後計算自適應門檻
        - 週期性發送預覽畫面與進度（5% ~ 55%）

        Returns:
            samples          : list[{'frame_idx', 'landmarks_arr', 'native_draw'}]
            body_scale_ref   : 穩定肩寬中位數（> 0 才有效）
            group_thresholds : {group_name: th_speed}
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error_occurred.emit(f'無法開啟影片檔案：{self.video_path}')
            return [], 0.0, {}

        samples: list[dict] = []
        shoulder_widths: list[float] = []
        group_raw_speeds: dict[str, list[float]] = {g: [] for g in _JOINT_GROUPS}
        # 建立門檻用的追蹤狀態（只採用高 visibility 點，與異常判定分開）
        th_prev_reliable: list[dict | None] = [None] * 33

        preview_every = max(1, self.frame_interval * 5)  # 每 5 個取樣幀更新一次預覽
        frame_idx = 0

        try:
            while not self._cancelled:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.frame_interval == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detected = detector.process_frame(rgb)

                    landmarks_arr = None
                    native_draw   = None
                    if detected:
                        landmarks_arr = detector.get_rula_landmarks(self.analysis_mode)
                        native_draw   = self._serialize_native_draw(
                            detector.get_native_draw_data_2d(), landmarks_arr
                        )

                        # 偶爾發送預覽畫面
                        if frame_idx % preview_every == 0:
                            annotated = detector.draw_landmarks(rgb.copy())
                            self.frame_preview.emit(annotated)

                        # 收集自適應門檻樣本（僅啟用速度異常時）
                        if self.enable_speed_anomaly and landmarks_arr and len(landmarks_arr) == 33:
                            self._collect_threshold_samples(
                                landmarks_arr, th_prev_reliable,
                                shoulder_widths, group_raw_speeds, frame_idx, fps
                            )

                    samples.append({
                        'frame_idx':     frame_idx,
                        'landmarks_arr': landmarks_arr,
                        'native_draw':   native_draw,
                    })

                    pct = int((frame_idx / total_frames) * 50) + 5
                    self.progress_updated.emit(
                        min(pct, 55),
                        f'姿勢分析中… 第 {frame_idx} / {total_frames} 幀'
                    )

                frame_idx += 1
        finally:
            cap.release()

        if self.enable_speed_anomaly:
            body_scale_ref, group_thresholds = self._compute_thresholds(
                shoulder_widths, group_raw_speeds, fps
            )
        else:
            body_scale_ref, group_thresholds = 0.0, {}
        return samples, body_scale_ref, group_thresholds

    # ------------------------------------------------------------------
    def _collect_threshold_samples(self, arr, prev_reliable, shoulder_widths,
                                   group_raw_speeds, frame_idx: int, fps: float):
        """收集肩寬與各群組原始速度（jump/dt），只採用高 visibility 點。"""
        # 肩寬樣本
        L_SHO, R_SHO = arr[11], arr[12]
        if float(L_SHO[3]) >= _THRESHOLD_VIS_HIGH and float(R_SHO[3]) >= _THRESHOLD_VIS_HIGH:
            sw = (
                (L_SHO[0] - R_SHO[0]) ** 2 +
                (L_SHO[1] - R_SHO[1]) ** 2 +
                (L_SHO[2] - R_SHO[2]) ** 2
            ) ** 0.5
            if sw > 1e-6:
                shoulder_widths.append(sw)

        # 各群組速度樣本（先讀 prev，再更新 prev）
        for i, lm in enumerate(arr):
            grp = _JOINT_TO_GROUP.get(i)
            if grp is None or float(lm[3]) < _THRESHOLD_VIS_HIGH:
                continue

            prev_info = prev_reliable[i]
            if not isinstance(prev_info, dict):
                continue
            prev_pos = prev_info.get('pos')
            prev_idx = prev_info.get('frame_idx')
            if prev_pos is None or prev_idx is None or prev_idx < 0:
                continue

            gap_seconds = (frame_idx - prev_idx) / fps if fps > 1e-9 else 0.0
            if gap_seconds <= 1e-9 or gap_seconds > _ANOM_MAX_GAP_SECONDS:
                continue

            jump = (
                (lm[0] - prev_pos[0]) ** 2 +
                (lm[1] - prev_pos[1]) ** 2 +
                (lm[2] - prev_pos[2]) ** 2
            ) ** 0.5
            # 收集 raw world-space 速度（jump/dt），掃描結束後再以 body_scale_ref 正規化
            group_raw_speeds[grp].append(jump / gap_seconds)

        for i, lm in enumerate(arr):
            if float(lm[3]) >= _THRESHOLD_VIS_HIGH:
                prev_reliable[i] = {'pos': [lm[0], lm[1], lm[2]], 'frame_idx': frame_idx}

    # ------------------------------------------------------------------
    def _compute_thresholds(self, shoulder_widths, group_raw_speeds, fps: float):
        """由收集的肩寬與群組速度，計算 body_scale_ref 與各群組自適應速度門檻。"""
        body_scale_ref = statistics.median(shoulder_widths) if shoulder_widths else 0.0

        group_thresholds: dict[str, float] = {}
        sample_dt = (self.frame_interval / fps) if fps > 1e-9 else 0.0
        min_speed_th = (_ANOM_MIN_JUMP_RATIO / sample_dt) if sample_dt > 1e-9 else None
        for grp, raw_speeds in group_raw_speeds.items():
            # 有有效 body_scale_ref 時，把 raw 速度轉成 speed_ratio
            if body_scale_ref and body_scale_ref > 1e-6:
                speeds = [rs / body_scale_ref for rs in raw_speeds]
            else:
                speeds = []

            if len(speeds) >= 20:
                med = statistics.median(speeds)
                mad = statistics.median([abs(v - med) for v in speeds])
                robust_std = 1.4826 * mad
                adaptive_th = med + 5 * robust_std if robust_std > 1e-6 else None
            else:
                adaptive_th = None

            if min_speed_th is None:
                th_speed = adaptive_th
            elif adaptive_th is None:
                th_speed = min_speed_th
            else:
                th_speed = max(adaptive_th, min_speed_th)

            group_thresholds[grp] = th_speed

        return body_scale_ref, group_thresholds

    # ------------------------------------------------------------------
    def _serialize_native_draw(self, native_draw_data, landmarks_arr):
        """
        將 backend 原生繪圖資料轉成可序列化 dict（供歷史 / 結果頁以相同 renderer 重播）。

        注意：本程式 RTMW 後端僅用於 2D 分析（3D 一律改用 MediaPipe），故不再輸出
        RTMW 的 3D 資料。
        """
        if not isinstance(native_draw_data, dict):
            return None

        backend_name = str(native_draw_data.get('backend', self.backend_mode)).upper()

        if backend_name in ('RTMW2D', 'RTMW3D'):
            kpts_norm = native_draw_data.get('keypoints_2d_norm') or []
            scores    = native_draw_data.get('scores') or []
            out = {
                'backend':           backend_name,
                'analysis_mode':     self.analysis_mode,
                'keypoints_2d_norm': [],  # 2D 原始模型關鍵點（用於繪圖）
                'scores':            [],  # 2D 原始模型信心度（用於繪圖）
            }
            for pt in kpts_norm:
                try:
                    out['keypoints_2d_norm'].append([float(pt[0]), float(pt[1])])
                except Exception:
                    out['keypoints_2d_norm'].append([0.0, 0.0])
            for sc in scores:
                try:
                    out['scores'].append(float(sc))
                except Exception:
                    out['scores'].append(0.0)
            return out

        if backend_name == 'MEDIAPIPE':
            lms = native_draw_data.get('landmarks_2d') or []
            out = {
                'backend':       'MEDIAPIPE',
                'analysis_mode': self.analysis_mode,
                'landmarks_2d':  [],  # 2D 正規化座標 [x, y, visibility]（用於繪圖）
                'landmarks_3d':  [],  # 3D 世界座標 [x, y, z, visibility]（用於 3D 繪圖與信心度查詢）
            }
            for lm in lms:
                try:
                    # landmarks_2d 格式為 [x, y, visibility]（3 個值）
                    out['landmarks_2d'].append([float(lm[0]), float(lm[1]), float(lm[2])])
                except Exception:
                    out['landmarks_2d'].append([0.0, 0.0, 0.0])
            if self.analysis_mode == '3D' and landmarks_arr:
                for lm in landmarks_arr:
                    try:
                        out['landmarks_3d'].append([
                            float(lm[0]), float(lm[1]), float(lm[2]), float(lm[3])
                        ])
                    except Exception:
                        out['landmarks_3d'].append([0.0, 0.0, 0.0, 0.0])
            return out

        return None

    # ------------------------------------------------------------------
    def _build_records(self, samples: list, fps: float,
                       body_scale_ref: float, group_thresholds: dict):
        """
        純 CPU 階段：對快取的取樣幀做異常判定、角度計算與記錄組裝（55% ~ 95%）。

        分三個子階段：
          0. （僅 3D）建立 bone_ref（影片內各骨段中位數長度）
          1. 逐幀異常偵測（low_visibility / bone_length_abnormal / speed_candidate）
          2. 以遮罩計算角度並組裝記錄

        speed_candidate 永不視為 invalid（僅 UI 提示），故無需後處理即可直接計算角度。
        """
        # ── Phase 0: 影片內骨長基準（僅 3D 分析；2D 只用 confidence）──────
        if self.analysis_mode == '3D':
            active_bones = _active_bones(self.backend_mode)
            bone_ref = self._compute_bone_ref(samples, active_bones)
        else:
            bone_ref = None

        # ── Phase 1: 逐幀異常偵測 ──────────────────────────────────────
        anomaly_prev_reliable: list = [None] * 33  # 每個關節上一個可靠幀的 {'pos','frame_idx'}
        anomaly_dt = self.frame_interval / fps if fps > 1e-9 else 0.0
        masks: list = [None] * len(samples)
        details: list = [None] * len(samples)
        bone_anomalies: list = [{} for _ in samples]  # 每幀的骨段異常 {bone_name: ratio}

        for s_i, sample in enumerate(samples):
            if self._cancelled:
                return []
            landmarks_arr = sample['landmarks_arr']
            if not (landmarks_arr and len(landmarks_arr) == 33):
                continue

            if body_scale_ref > 1e-6:
                body_scale = body_scale_ref
            else:
                L_SHO, R_SHO = landmarks_arr[11], landmarks_arr[12]
                body_scale = (
                    (L_SHO[0]-R_SHO[0])**2 +
                    (L_SHO[1]-R_SHO[1])**2 +
                    (L_SHO[2]-R_SHO[2])**2
                ) ** 0.5
            gt = group_thresholds if self.enable_speed_anomaly else None
            mask, anomaly_prev_reliable, detail, bone_anom = _compute_anomaly_mask(
                landmarks_arr, anomaly_prev_reliable, body_scale, anomaly_dt,
                sample['frame_idx'], self.frame_interval,
                group_thresholds=gt, bone_ref=bone_ref,
            )
            masks[s_i] = mask
            details[s_i] = detail
            bone_anomalies[s_i] = bone_anom

        # ── Phase 1.5: 線性插值補點（僅 use_interpolation 時）─────────────
        interp_arrs = interp_masks = interp_meta = None
        if self.use_interpolation:
            interp_arrs, interp_masks, interp_meta = self._interpolate(
                samples, masks, details, fps
            )

        # ── Phase 2: 角度計算 + 記錄組裝 ───────────────────────────────
        # 永遠計算「原始」RULA；開啟補點時另計算「補點後」RULA，兩者並存供比較。
        # primary（寫入既有欄位、供圖表/匯出）依 use_interpolation 選擇。
        records: list = []
        prev_left  = None
        prev_right = None
        total = max(1, len(samples))

        for s_i, sample in enumerate(samples):
            if self._cancelled:
                break

            frame_idx     = sample['frame_idx']
            landmarks_arr = sample['landmarks_arr']
            native_draw   = sample['native_draw']

            joint_anomaly        = masks[s_i]
            joint_anomaly_detail = details[s_i]
            bone_anomaly_frame   = bone_anomalies[s_i]

            # 原始 RULA（原始骨架 + 原始遮罩）
            orig_left = orig_right = None
            if landmarks_arr:
                orig_left, orig_right = angle_calc(
                    landmarks_arr, None, None,
                    joint_anomaly=joint_anomaly,
                    analysis_mode=self.analysis_mode,
                )
            orig_score = _score_num(orig_left, orig_right)

            # 補點後 RULA（補點骨架 + 補點遮罩）
            interp_left = interp_right = None
            interp_score = None
            frame_interp = interp_meta[s_i] if interp_meta else None
            if self.use_interpolation and interp_arrs is not None:
                i_arr  = interp_arrs[s_i]
                i_mask = interp_masks[s_i]
                if i_arr:
                    interp_left, interp_right = angle_calc(
                        i_arr, None, None,
                        joint_anomaly=i_mask,
                        analysis_mode=self.analysis_mode,
                    )
                    interp_score = _score_num(interp_left, interp_right)

            # primary 選擇
            if self.use_interpolation:
                rula_left, rula_right, score_num = interp_left, interp_right, interp_score
            else:
                rula_left, rula_right, score_num = orig_left, orig_right, orig_score
            prev_left, prev_right = rula_left, rula_right

            record = {
                'frame':             frame_idx,
                'timestamp':         round(frame_idx / fps, 3) if fps > 1e-9 else 0.0,
                'best_score':        score_num,
                'left_score':        rula_left.get('score', 'NULL')  if rula_left  else 'NULL',
                'right_score':       rula_right.get('score', 'NULL') if rula_right else 'NULL',
                # 左側角度資料
                'left_upper_arm_angle':  rula_left.get('upper_arm_angle', 'NULL') if rula_left else 'NULL',
                'left_lower_arm_angle':  rula_left.get('lower_arm_angle', 'NULL') if rula_left else 'NULL',
                'left_wrist_angle':      rula_left.get('wrist_angle',     'NULL') if rula_left else 'NULL',
                'left_neck_angle':       rula_left.get('neck_angle',      'NULL') if rula_left else 'NULL',
                'left_trunk_angle':      rula_left.get('trunk_angle',     'NULL') if rula_left else 'NULL',
                'left_posture_score_a':  rula_left.get('posture_score_a', 'NULL') if rula_left else 'NULL',
                'left_posture_score_b':  rula_left.get('posture_score_b', 'NULL') if rula_left else 'NULL',
                # 右側角度資料
                'right_upper_arm_angle': rula_right.get('upper_arm_angle', 'NULL') if rula_right else 'NULL',
                'right_lower_arm_angle': rula_right.get('lower_arm_angle', 'NULL') if rula_right else 'NULL',
                'right_wrist_angle':     rula_right.get('wrist_angle',     'NULL') if rula_right else 'NULL',
                'right_neck_angle':      rula_right.get('neck_angle',      'NULL') if rula_right else 'NULL',
                'right_trunk_angle':     rula_right.get('trunk_angle',     'NULL') if rula_right else 'NULL',
                'right_posture_score_a': rula_right.get('posture_score_a', 'NULL') if rula_right else 'NULL',
                'right_posture_score_b': rula_right.get('posture_score_b', 'NULL') if rula_right else 'NULL',
                # 原生繪圖資料（保存所有關節，供歷史重播）
                'native_draw_data':  native_draw,
                # 異常判定結果：33 個 bool，True=可靠 False=疑似異常
                'joint_anomaly':        joint_anomaly,
                # 異常診斷明細：33 個 None | {'reason','reasons','visibility','speed_ratio',...}
                'joint_anomaly_detail': joint_anomaly_detail,
                # 骨段長度異常：{bone_name: ratio}，空 dict = 無骨長異常（繪紅色骨段線用）
                'bone_anomaly':         bone_anomaly_frame,
                # 補點前後分數（供 Original vs Interpolated 比較）
                'original_best_score':  orig_score,
                'interp_best_score':    interp_score,
            }
            # 開啟補點時，額外儲存補點前原始角度（primary 欄位此時已是補點後）
            if self.use_interpolation:
                def _g(obj, key): return obj.get(key, 'NULL') if obj else 'NULL'
                record.update({
                    'orig_left_upper_arm_angle':  _g(orig_left,  'upper_arm_angle'),
                    'orig_left_lower_arm_angle':  _g(orig_left,  'lower_arm_angle'),
                    'orig_left_wrist_angle':      _g(orig_left,  'wrist_angle'),
                    'orig_left_neck_angle':       _g(orig_left,  'neck_angle'),
                    'orig_left_trunk_angle':      _g(orig_left,  'trunk_angle'),
                    'orig_left_posture_score_a':  _g(orig_left,  'posture_score_a'),
                    'orig_left_posture_score_b':  _g(orig_left,  'posture_score_b'),
                    'orig_left_score':            _g(orig_left,  'score'),
                    'orig_right_upper_arm_angle': _g(orig_right, 'upper_arm_angle'),
                    'orig_right_lower_arm_angle': _g(orig_right, 'lower_arm_angle'),
                    'orig_right_wrist_angle':     _g(orig_right, 'wrist_angle'),
                    'orig_right_neck_angle':      _g(orig_right, 'neck_angle'),
                    'orig_right_trunk_angle':     _g(orig_right, 'trunk_angle'),
                    'orig_right_posture_score_a': _g(orig_right, 'posture_score_a'),
                    'orig_right_posture_score_b': _g(orig_right, 'posture_score_b'),
                    'orig_right_score':           _g(orig_right, 'score'),
                })
            # 補點關節資訊（稀疏）：{ str(mp_idx): {gap_sec, reasons, pos_norm} }
            if frame_interp:
                record['interpolation'] = {
                    str(j): info for j, info in frame_interp.items()
                }
            records.append(record)

            pct = int((s_i / total) * 40) + 55
            self.progress_updated.emit(
                min(pct, 95),
                f'計算 RULA 分數中… {s_i + 1} / {len(samples)}'
            )

        return records

    # ------------------------------------------------------------------
    def _compute_bone_ref(self, samples: list, active_bones: dict) -> dict:
        """
        建立影片內骨長基準：對每個骨段，收集所有有效幀的長度後取中位數。

        只處理 active_bones（依 backend 挑選，見 _active_bones）；未列入者不建立
        基準，逐幀檢查 (_bone_anomaly_for_frame) 會因 bone_ref 缺該鍵而自動略過。

        valid 條件（見 _bone_length）：
          - 兩端關節 visibility >= _ANOM_VIS_TH
          - bone_length > 0
        使用 median 而非 mean，避免被少數異常值拉動。
        """
        collected: dict[str, list[float]] = {name: [] for name in active_bones}
        for sample in samples:
            arr = sample.get('landmarks_arr')
            if not (arr and len(arr) == 33):
                continue
            for name, spec in active_bones.items():
                a, b = spec['pair']
                length = _bone_length(arr, a, b)
                if length is not None:
                    collected[name].append(length)

        bone_ref: dict[str, float] = {}
        for name, lengths in collected.items():
            if lengths:
                bone_ref[name] = statistics.median(lengths)
        return bone_ref

    # ------------------------------------------------------------------
    def _native_joint_norm(self, native, mp_idx: int):
        """取得某 MediaPipe 關節在原生繪圖資料中的正規化 [x, y]，無則 None。"""
        if not isinstance(native, dict):
            return None
        backend = str(native.get('backend', '')).upper()
        if backend in ('RTMW2D', 'RTMW3D'):
            rtmw_idx = RTMW_TO_MEDIAPIPE.get(mp_idx)
            if rtmw_idx is None:
                return None
            pts = native.get('keypoints_2d_norm') or []
            if rtmw_idx < len(pts) and len(pts[rtmw_idx]) >= 2:
                return [float(pts[rtmw_idx][0]), float(pts[rtmw_idx][1])]
        else:
            lms = native.get('landmarks_2d') or []
            if mp_idx < len(lms) and len(lms[mp_idx]) >= 2:
                return [float(lms[mp_idx][0]), float(lms[mp_idx][1])]
        return None

    # ------------------------------------------------------------------
    def _interpolate(self, samples: list, masks: list, details: list, fps: float):
        """
        線性插值補點：對每個 RULA 關節，在前後最近可信幀之間補回短暫 invalid 的位置。

        對每段 invalid run（兩端皆為可信且 gap <= _INTERP_MAX_GAP_SECONDS）：
            ratio = (t_cur - t_prev) / (t_next - t_prev)
            p_interp = p_prev + (p_next - p_prev) * ratio
        分析座標（x,y,z）與顯示座標（正規化 x,y）使用同一組括弧分別插值。
        補點關節的 visibility 設為 1.0，使後續 check_confidence / 遮罩視為可用。

        Returns（長度皆為 len(samples)）：
            interp_arrs:  補點後的 33×[x,y,z,vis]（複本）或 None（該幀無姿勢）
            interp_masks: 補點後遮罩（補點關節 → True）或 None
            interp_meta:  {mp_idx: {'gap_sec','reasons','pos_norm'}} 每幀（無補點則空 dict）
        """
        n = len(samples)
        interp_arrs: list = [None] * n
        interp_masks: list = [None] * n
        interp_meta: list = [dict() for _ in range(n)]

        for i in range(n):
            arr = samples[i].get('landmarks_arr')
            if arr and len(arr) == 33:
                interp_arrs[i] = [list(p) for p in arr]
                interp_masks[i] = list(masks[i]) if masks[i] else [True] * 33

        def t(i: int) -> float:
            return samples[i]['frame_idx'] / fps if fps > 1e-9 else 0.0

        def has_pose(i: int) -> bool:
            return interp_arrs[i] is not None

        def reliable(i: int, j: int) -> bool:
            m = masks[i]
            return bool(m) and j < len(m) and bool(m[j])

        for j in _INTERP_JOINTS:
            s = 0
            while s < n:
                # 略過可信幀或無姿勢幀（無姿勢幀視為括弧牆）
                if not has_pose(s) or reliable(s, j):
                    s += 1
                    continue
                # s 為 invalid（且有姿勢）的 run 起點；往後找 run 結束
                run_start = s
                k = s
                while k < n and has_pose(k) and not reliable(k, j):
                    k += 1
                prev_idx = run_start - 1
                left_ok  = prev_idx >= 0 and has_pose(prev_idx) and reliable(prev_idx, j)
                next_ok  = k < n and has_pose(k) and reliable(k, j)
                if left_ok and next_ok:
                    gap = t(k) - t(prev_idx)
                    if 0 < gap <= _INTERP_MAX_GAP_SECONDS:
                        self._fill_interp_run(
                            samples, details, interp_arrs, interp_masks, interp_meta,
                            j, prev_idx, k, run_start, t, gap,
                        )
                s = max(k, run_start + 1)

        return interp_arrs, interp_masks, interp_meta

    def _fill_interp_run(self, samples, details, interp_arrs, interp_masks, interp_meta,
                         j: int, prev_idx: int, next_idx: int, run_start: int, t, gap: float):
        """對單一關節 j 在 [run_start, next_idx) 的 invalid 幀做線性插值填補。"""
        prev_arr = samples[prev_idx]['landmarks_arr']
        next_arr = samples[next_idx]['landmarks_arr']
        prev_disp = self._native_joint_norm(samples[prev_idx]['native_draw'], j)
        next_disp = self._native_joint_norm(samples[next_idx]['native_draw'], j)
        t_prev, t_next = t(prev_idx), t(next_idx)
        span = t_next - t_prev
        if span <= 1e-9:
            return

        for m in range(run_start, next_idx):
            if interp_arrs[m] is None or (interp_masks[m] and interp_masks[m][j]):
                continue
            ratio = (t(m) - t_prev) / span
            for a in range(3):
                interp_arrs[m][j][a] = (
                    prev_arr[j][a] + (next_arr[j][a] - prev_arr[j][a]) * ratio
                )
            interp_arrs[m][j][3] = 1.0          # 視為可用，使 check_confidence 通過
            interp_masks[m][j] = True

            pos_norm = None
            if prev_disp is not None and next_disp is not None:
                pos_norm = [
                    round(prev_disp[0] + (next_disp[0] - prev_disp[0]) * ratio, 6),
                    round(prev_disp[1] + (next_disp[1] - prev_disp[1]) * ratio, 6),
                ]

            # 插值後 3D 世界座標（與 native['landmarks_3d'] 同座標系，供 3D 骨架藍圈用）
            pos_3d = None
            if self.analysis_mode == '3D':
                pos_3d = [
                    round(interp_arrs[m][j][0], 6),
                    round(interp_arrs[m][j][1], 6),
                    round(interp_arrs[m][j][2], 6),
                ]

            d = details[m][j] if (details[m] and j < len(details[m])) else None
            reasons: list = []
            if isinstance(d, dict):
                src = d.get('reasons') or ([d['reason']] if d.get('reason') else [])
                reasons = [r for r in src if r != 'speed_candidate']
            interp_meta[m][j] = {
                'gap_sec':  round(gap, 3),
                'reasons':  reasons,
                'pos_norm': pos_norm,
                'pos_3d':   pos_3d,
            }

    # ------------------------------------------------------------------
    def _build_results(self, records: list, total_frames: int, fps: float,
                       analysis_duration_seconds: float = 0.0,
                       group_thresholds: dict | None = None) -> dict:
        valid_scores = [r['best_score'] for r in records
                        if isinstance(r['best_score'], int)]

        dist = {str(i): 0 for i in range(1, 8)}
        for s in valid_scores:
            k = str(max(1, min(7, s)))
            dist[k] = dist.get(k, 0) + 1

        interpolation_summary = self._build_interpolation_summary(records)
        rula_comparison       = self._build_rula_comparison(records)

        return {
            'meta':             self.meta,
            'video_path':       self.video_path,
            'original_filename': os.path.basename(self.video_path),
            'total_frames':     total_frames,
            'processed_frames': len(records),
            'fps':              fps,
            'frame_interval':   self.frame_interval,
            'analysis_duration_seconds': round(max(0.0, float(analysis_duration_seconds)), 3),
            'backend_mode':     (self.backend_mode or 'MEDIAPIPE').upper(),
            'analysis_mode':    self.analysis_mode,
            'rula_params':      dict(self.rula_params),
            'speed_anomaly_enabled': bool(self.enable_speed_anomaly),
            'use_interpolation':     bool(self.use_interpolation),
            # 速度門檻：每群組一個 th_speed（全片共用一份，不再逐幀重複儲存）
            'joint_group_thresholds': group_thresholds if group_thresholds else None,
            'interpolation_summary': interpolation_summary,
            'rula_comparison':       rula_comparison,
            'records':          records,
            'stats': {
                'max_score':          max(valid_scores)  if valid_scores else None,
                'avg_score':          round(sum(valid_scores) / len(valid_scores), 2)
                                      if valid_scores else None,
                'score_distribution': dist,
            },
            'created_at': datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    def _build_interpolation_summary(self, records: list) -> dict | None:
        """彙整補點統計：總數、各關節數、補點率。無補點時回傳 None。"""
        if not self.use_interpolation:
            return None
        per_joint: dict[str, int] = {}
        total_interp = 0
        frames_with_pose = 0
        for r in records:
            if r.get('joint_anomaly'):
                frames_with_pose += 1
            interp = r.get('interpolation') or {}
            for j_str in interp:
                total_interp += 1
                name = _INTERP_JOINT_NAMES.get(int(j_str), j_str)
                per_joint[name] = per_joint.get(name, 0) + 1
        total_points = frames_with_pose * len(_INTERP_JOINTS)
        rate = (total_interp / total_points) if total_points > 0 else 0.0
        # 依數量排序，方便 UI 由多到少顯示
        per_joint_sorted = dict(sorted(per_joint.items(), key=lambda kv: kv[1], reverse=True))
        return {
            'total_interpolated': total_interp,
            'total_points':       total_points,
            'rate':               round(rate, 4),
            'per_joint':          per_joint_sorted,
        }

    # ------------------------------------------------------------------
    def _build_rula_comparison(self, records: list) -> dict | None:
        """彙整原始 vs 補點後 RULA（平均 / 最大 / delta）。未開啟補點時回傳 None。"""
        if not self.use_interpolation:
            return None
        orig = [r['original_best_score'] for r in records
                if isinstance(r.get('original_best_score'), int)]
        interp = [r['interp_best_score'] for r in records
                  if isinstance(r.get('interp_best_score'), int)]
        # 因補點恢復而可計算的幀數（原始 NULL → 補點後有值）
        recovered = sum(
            1 for r in records
            if not isinstance(r.get('original_best_score'), int)
            and isinstance(r.get('interp_best_score'), int)
        )
        orig_avg   = round(sum(orig) / len(orig), 2) if orig else None
        interp_avg = round(sum(interp) / len(interp), 2) if interp else None
        delta_avg  = (round(interp_avg - orig_avg, 2)
                      if (orig_avg is not None and interp_avg is not None) else None)
        return {
            'original_avg':     orig_avg,
            'interpolated_avg': interp_avg,
            'original_max':     max(orig) if orig else None,
            'interpolated_max': max(interp) if interp else None,
            'delta_avg':        delta_avg,
            'recovered_frames': recovered,
        }


# ------------------------------------------------------------------
# 本機歷史記錄存取
# ------------------------------------------------------------------
HISTORY_DIR = os.path.join(os.path.expanduser('~'), '.rula_analyses')


def ensure_history_dir():
    os.makedirs(HISTORY_DIR, exist_ok=True)


def save_analysis(results: dict) -> str:
    """將分析結果儲存為 JSON，回傳檔案路徑（含 native_draw_data）。"""
    ensure_history_dir()
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = os.path.basename(results.get('video_path', 'unknown'))
    safe = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in name)[:40]
    path = os.path.join(HISTORY_DIR, f'{ts}_{safe}.json')

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return path


def load_history() -> list:
    """載入所有歷史分析記錄（最新優先）"""
    ensure_history_dir()
    items = []
    for fname in sorted(os.listdir(HISTORY_DIR), reverse=True):
        if fname.endswith('.json'):
            fpath = os.path.join(HISTORY_DIR, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data['_json_path'] = fpath
                items.append(data)
            except Exception:
                pass
    return items


def export_csv(results: dict, csv_path: str):
    """將分析記錄匯出為 CSV"""
    records = results.get('records', [])
    if not records:
        return
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'timestamp',
                                               'best_score', 'left_score', 'right_score'],
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
