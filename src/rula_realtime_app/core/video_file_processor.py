"""
影片檔案處理器 - 在背景執行緒中對影片檔案進行 RULA 離線分析
"""

import cv2
import csv
import json
import os
import time
from datetime import datetime

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from .pose_detector import PoseDetector
from . import angle_calc
from .utils import get_best_rula_score
from .config import RULA_CONFIG

# ── Anomaly detection constants (MediaPipe world-coordinate space) ──────────
# 預設值，未來透過實驗校正
_ANOM_VIS_TH       = 0.50   # visibility 低於此值 → 直接不可靠
_ANOM_MAX_GAP_SECONDS = 1.0  # Pass 1 / Pass 2 共同使用的最大觀測間隔（秒）

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


def _compute_anomaly_mask(landmarks_arr, prev_reliable, body_scale, dt, current_frame_idx,
                          frame_interval: int,
                          group_thresholds: dict | None = None):
    """
    判斷 MediaPipe 33 個關節點是否可靠（非異常）。

    Args:
        landmarks_arr:    33 × [x, y, z, vis]，MediaPipe world coordinates
        prev_reliable:    list[dict|None]，每個關節上一個可靠幀的 {pos, frame_idx}
        body_scale:       人體尺度參考（肩寬），用來正規化速度
        dt:               兩個分析幀之間的實際時間差（秒）
        current_frame_idx: 目前分析幀索引，用來計算 gap_seconds
        frame_interval:    抽樣間隔（幀）
        group_thresholds: {group_name: th_speed}，由 Pass 1 計算；None 時退回固定值

    Returns:
        mask (list[bool]):         True = 可靠，False = 疑似異常
        new_prev (list[dict|None]): 更新後的 prev_reliable（只更新可靠的關節）
    """
    mask = []
    new_prev = list(prev_reliable)
    detail: list = []   # {'reason', 'visibility', 'speed_ratio', 'speed_checked', 'th_speed'}

    for i, lm in enumerate(landmarks_arr):
        x, y, z, vis = float(lm[0]), float(lm[1]), float(lm[2]), float(lm[3])
        grp = _JOINT_TO_GROUP.get(i)
        th_speed = None
        if group_thresholds:
            th_speed = group_thresholds.get(grp)
        reliable = True
        reason: str | None = None
        speed_ratio: float | None = None
        speed_checked = False

        if vis < _ANOM_VIS_TH:
            reliable = False
            reason = 'low_visibility'
        else:
            prev = prev_reliable[i]
            prev_pos = None
            prev_frame_idx = None
            if isinstance(prev, dict):
                prev_pos = prev.get('pos')
                prev_frame_idx = prev.get('frame_idx')
            elif isinstance(prev, (list, tuple)) and len(prev) >= 3:
                prev_pos = prev

            elapsed_seconds = dt
            if prev_frame_idx is not None and current_frame_idx is not None and prev_frame_idx >= 0:
                elapsed_seconds = ((current_frame_idx - prev_frame_idx) / max(1, frame_interval)) * dt

            if grp is not None and th_speed is not None and prev_pos is not None and elapsed_seconds > 1e-9 and body_scale > 1e-6:
                gap_seconds = elapsed_seconds
                if gap_seconds <= _ANOM_MAX_GAP_SECONDS:
                    jump = ((x - prev_pos[0])**2 + (y - prev_pos[1])**2 + (z - prev_pos[2])**2) ** 0.5
                    speed_ratio = (jump / gap_seconds) / body_scale
                    speed_checked = True
                else:
                    speed_ratio = None

            if th_speed is not None and speed_ratio is not None and speed_ratio > th_speed:
                reliable = False
                reason = 'speed_jump'

        mask.append(reliable)
        if reliable:
            new_prev[i] = {'pos': [x, y, z], 'frame_idx': current_frame_idx}
        detail.append({
            'reason':        reason,
            'visibility':    round(vis, 4),
            'speed_ratio':   round(speed_ratio, 4) if speed_ratio is not None else None,
            'speed_checked': speed_checked and th_speed is not None,
            'th_speed':      round(th_speed, 4) if th_speed is not None else None,
        })

    return mask, new_prev, detail


def _run_pass1(
    video_path: str,
    detector,
    frame_interval: int,
    fps: float,
    vis_high: float = 0.80,
) -> tuple[float, dict[str, float]]:
    """
    Pass 1：預掃描影片，計算 body_scale_ref 與各關節群組自適應速度門檻。

    Returns:
        body_scale_ref  : 穩定肩寬中位數（> 0 才有效）
        group_thresholds: {group_name: th_speed}
    """
    import statistics

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, {}

    shoulder_widths: list[float] = []
    # Collect raw speeds (jump / dt) during Pass 1, normalize after body_scale_ref is computed
    group_raw_speeds: dict[str, list[float]] = {g: [] for g in _JOINT_GROUPS}

    # 與 Pass 2 一致：每個關節保存 {'pos': [x, y, z], 'frame_idx': int} 或 None
    prev_reliable: list[dict | None] = [None] * 33
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected = detector.process_frame(rgb)
            if detected:
                arr = detector.get_landmarks_array()
                if arr and len(arr) == 33:
                    # 收集肩寬樣本
                    L_SHO, R_SHO = arr[11], arr[12]
                    if float(L_SHO[3]) >= vis_high and float(R_SHO[3]) >= vis_high:
                        sw = (
                            (L_SHO[0] - R_SHO[0]) ** 2 +
                            (L_SHO[1] - R_SHO[1]) ** 2 +
                            (L_SHO[2] - R_SHO[2]) ** 2
                        ) ** 0.5
                        if sw > 1e-6:
                            shoulder_widths.append(sw)

                    # 收集各群組速度樣本（只用高 visibility 點）
                    for i, lm in enumerate(arr):
                        grp = _JOINT_TO_GROUP.get(i)
                        if grp is None:
                            continue
                        if float(lm[3]) < vis_high:
                            continue

                        prev_info = prev_reliable[i]
                        if not isinstance(prev_info, dict):
                            continue

                        prev_pos = prev_info.get('pos')
                        prev_idx = prev_info.get('frame_idx')
                        if prev_pos is None or prev_idx is None or prev_idx < 0:
                            continue

                        gap_seconds = (frame_idx - prev_idx) / fps
                        if gap_seconds > _ANOM_MAX_GAP_SECONDS:
                            continue

                        jump = (
                            (lm[0] - prev_pos[0]) ** 2 +
                            (lm[1] - prev_pos[1]) ** 2 +
                            (lm[2] - prev_pos[2]) ** 2
                        ) ** 0.5
                        # Collect raw world-space speed (jump/dt). We'll normalize
                        # by the final body_scale_ref after the full scan completes.
                        raw_speed = (jump / gap_seconds) if gap_seconds > 1e-9 else 0.0
                        group_raw_speeds[grp].append(raw_speed)

                    # 更新 prev_reliable（Pass 1 全用高 visibility 的點）
                    for i, lm in enumerate(arr):
                        if float(lm[3]) >= vis_high:
                            prev_reliable[i] = {
                                'pos': [lm[0], lm[1], lm[2]],
                                'frame_idx': frame_idx,
                            }

        frame_idx += 1

    cap.release()

    # 計算 body_scale_ref（中位數）
    body_scale_ref = statistics.median(shoulder_widths) if shoulder_widths else 0.0

    # Normalize collected raw speeds by body_scale_ref (if available) and
    # compute per-group adaptive thresholds using robust statistics.
    group_thresholds: dict[str, float] = {}
    for grp, raw_speeds in group_raw_speeds.items():
        # If we have a valid body_scale_ref, convert raw speeds to speed_ratio
        if body_scale_ref and body_scale_ref > 1e-6:
            speeds = [rs / body_scale_ref for rs in raw_speeds]
        else:
            speeds = []

        if len(speeds) >= 20:
            med = statistics.median(speeds)
            abs_devs = [abs(v - med) for v in speeds]
            mad = statistics.median(abs_devs)
            robust_std = 1.4826 * mad
            if robust_std > 1e-6:
                th_speed = med + 5 * robust_std
            else:
                th_speed = None
        else:
            th_speed = None
        group_thresholds[grp] = th_speed

    return body_scale_ref, group_thresholds


class VideoFileProcessor(QObject):
    """
    離線影片 RULA 分析工作器（在 QThread 中執行）

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
                 backend_mode: str = 'RTMW3D',
                 rula_params: dict | None = None):
        """
        Args:
            video_path:     影片檔案路徑
            meta:           調查資訊 dict（survey_date, assessor, organization, task_name）
            frame_interval: 每隔幾幀取樣一次（預設 10）
            backend_mode:   姿勢偵測模式（'RTMW3D' 或 'MEDIAPIPE'）
            rula_params:    RULA 固定參數覆寫（wrist_twist, legs, muscle_use_a/b, force_load_a/b）
        """
        super().__init__()
        self.video_path     = video_path
        self.meta           = meta
        self.frame_interval = max(1, frame_interval)
        self.backend_mode   = backend_mode
        merged_params = dict(RULA_CONFIG)
        if isinstance(rula_params, dict):
            for key in ('wrist_twist', 'legs', 'muscle_use_a', 'muscle_use_b', 'force_load_a', 'force_load_b'):
                if key in rula_params:
                    merged_params[key] = int(rula_params[key])
        self.rula_params    = merged_params
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

        self.progress_updated.emit(3, '初始化姿勢偵測模型...')

        original_rula_config = dict(RULA_CONFIG)
        RULA_CONFIG.update(self.rula_params)

        try:
            detector   = PoseDetector(backend_mode=self.backend_mode)
            prev_left  = None
            prev_right = None
            records    = []
            frame_idx  = 0
            preview_every = max(1, self.frame_interval * 5)  # 每 5 個取樣幀更新一次預覽

            # Anomaly detection state (MediaPipe only)
            _anomaly_prev_reliable = [None] * 33  # 每個關節上一個可靠幀的 {'pos','frame_idx'}
            _anomaly_dt = self.frame_interval / fps  # 兩個分析幀的時間差（秒）
            _body_scale_ref    = 0.0
            _group_thresholds: dict = {}

            # Pass 1：預掃描影片，建立自適應速度門檻（MediaPipe only）
            if self.backend_mode == 'MEDIAPIPE':
                self.progress_updated.emit(4, 'Pass 1：建立速度分布...')
                _body_scale_ref, _group_thresholds = _run_pass1(
                    self.video_path, detector, self.frame_interval, fps
                )
                # 重置影片讀取位置，準備 Pass 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0

            while not self._cancelled:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.frame_interval == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detected = detector.process_frame(rgb)

                    rula_left    = None
                    rula_right   = None
                    landmarks_arr = None

                    native_draw_data = None
                    joint_anomaly        = None
                    joint_anomaly_detail = None
                    if detected:
                        landmarks_arr    = detector.get_landmarks_array()
                        native_draw_data = detector.get_native_draw_data_2d()

                        # Anomaly detection BEFORE angle_calc (MediaPipe only)
                        if self.backend_mode == 'MEDIAPIPE' and landmarks_arr and len(landmarks_arr) == 33:
                            if _body_scale_ref > 1e-6:
                                body_scale = _body_scale_ref
                            else:
                                L_SHO, R_SHO = landmarks_arr[11], landmarks_arr[12]
                                body_scale = (
                                    (L_SHO[0]-R_SHO[0])**2 +
                                    (L_SHO[1]-R_SHO[1])**2 +
                                    (L_SHO[2]-R_SHO[2])**2
                                ) ** 0.5
                            joint_anomaly, _anomaly_prev_reliable, joint_anomaly_detail = _compute_anomaly_mask(
                                landmarks_arr, _anomaly_prev_reliable, body_scale, _anomaly_dt,
                                frame_idx, self.frame_interval,
                                group_thresholds=_group_thresholds,
                            )

                        rula_left, rula_right = angle_calc(
                            landmarks_arr, prev_left, prev_right,
                            joint_anomaly=joint_anomaly,
                        )
                        prev_left  = rula_left
                        prev_right = rula_right

                        # 偶爾發送預覽畫面
                        if frame_idx % preview_every == 0:
                            annotated = detector.draw_landmarks(rgb.copy())
                            self.frame_preview.emit(annotated)

                    best_result = get_best_rula_score(rula_left, rula_right)
                    score_str   = best_result.get('final_tableC_score', 'NULL')
                    score_num   = None
                    try:
                        if score_str != 'NULL':
                            score_num = int(score_str)
                    except (ValueError, TypeError):
                        pass

                    # Store backend-native draw payload so history/result page can
                    # re-render with the same renderer as realtime mode.
                    serializable_native_draw_data = None
                    if isinstance(native_draw_data, dict):
                        backend_name = str(native_draw_data.get('backend', self.backend_mode)).upper()
                        if backend_name == 'RTMW3D':
                            kpts_norm = native_draw_data.get('keypoints_2d_norm') or []
                            scores = native_draw_data.get('scores') or []
                            raw_3d = native_draw_data.get('keypoints_3d_raw') or []
                            serializable_native_draw_data = {
                                'backend': 'RTMW3D',
                                'keypoints_2d_norm': [],  # 2D 原始模型關鍵點（用於繪圖）
                                'scores': [],             # 2D 原始模型信心度（用於繪圖）
                                'landmarks_3d': [],       # 3D 33點 MediaPipe映射（用於角度計算與信心度查詢）
                                'keypoints_3d_raw': [],   # 全身 raw 3D [K × [x,y,z]]（用於3D骨架繪圖）
                            }
                            for pt in kpts_norm:
                                try:
                                    serializable_native_draw_data['keypoints_2d_norm'].append([
                                        float(pt[0]), float(pt[1])
                                    ])
                                except Exception:
                                    serializable_native_draw_data['keypoints_2d_norm'].append([0.0, 0.0])
                            for sc in scores:
                                try:
                                    serializable_native_draw_data['scores'].append(float(sc))
                                except Exception:
                                    serializable_native_draw_data['scores'].append(0.0)
                            for pt3 in raw_3d:
                                try:
                                    serializable_native_draw_data['keypoints_3d_raw'].append([
                                        float(pt3[0]), float(pt3[1]), float(pt3[2])
                                    ])
                                except Exception:
                                    serializable_native_draw_data['keypoints_3d_raw'].append([0.0, 0.0, 0.0])
                        elif backend_name == 'MEDIAPIPE':
                            lms = native_draw_data.get('landmarks_2d') or []
                            serializable_native_draw_data = {
                                'backend': 'MEDIAPIPE',
                                'landmarks_2d': [],  # 2D 正規化座標 [x, y, visibility]（用於繪圖）
                                'landmarks_3d': [],  # 3D 世界座標 [x, y, z, visibility]（用於角度計算與信心度查詢）
                            }
                            for lm in lms:
                                try:
                                    # landmarks_2d 格式為 [x, y, visibility]（3 個值）
                                    serializable_native_draw_data['landmarks_2d'].append([
                                        float(lm[0]), float(lm[1]), float(lm[2]),
                                    ])
                                except Exception:
                                    serializable_native_draw_data['landmarks_2d'].append([0.0, 0.0, 0.0])

                    # 序列化 landmarks_3d（兩個 backend 統一格式：33點 MediaPipe映射，[x,y,z,conf]）
                    if serializable_native_draw_data is not None and landmarks_arr:
                        for lm in landmarks_arr:
                            try:
                                serializable_native_draw_data['landmarks_3d'].append([
                                    float(lm[0]), float(lm[1]), float(lm[2]), float(lm[3])
                                ])
                            except Exception:
                                serializable_native_draw_data['landmarks_3d'].append([0.0, 0.0, 0.0, 0.0])

                    records.append({
                        'frame':             frame_idx,
                        'timestamp':         round(frame_idx / fps, 3),
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
                        'native_draw_data':  serializable_native_draw_data,
                        # 異常判定結果（MediaPipe only）：33 個 bool，True=可靠 False=疑似異常
                        'joint_anomaly':        joint_anomaly,
                        # 異常診斷明細（MediaPipe only）：33 個 None | {'reason','visibility','speed_ratio'}
                        'joint_anomaly_detail': joint_anomaly_detail,
                        # 速度門檻（MediaPipe only）：每群組一個 th_speed（或 None）
                        'joint_group_thresholds': _group_thresholds if self.backend_mode == 'MEDIAPIPE' else None,
                    })

                    # 進度（5% ~ 95%）
                    pct = int((frame_idx / total_frames) * 90) + 5
                    self.progress_updated.emit(
                        min(pct, 94),
                        f'分析中… 第 {frame_idx} / {total_frames} 幀'
                    )

                frame_idx += 1

            if self._cancelled:
                return

            self.progress_updated.emit(97, '統計資料中...')
            analysis_duration_seconds = max(0.0, time.perf_counter() - analysis_started_at)
            results = self._build_results(records, total_frames, fps, analysis_duration_seconds)
            self.progress_updated.emit(100, '分析完成！')
            self.analysis_complete.emit(results)
        finally:
            cap.release()
            RULA_CONFIG.clear()
            RULA_CONFIG.update(original_rula_config)

    # ------------------------------------------------------------------
    def _build_results(self, records: list, total_frames: int, fps: float,
                       analysis_duration_seconds: float = 0.0) -> dict:
        valid_scores = [r['best_score'] for r in records
                        if isinstance(r['best_score'], int)]

        dist = {str(i): 0 for i in range(1, 8)}
        for s in valid_scores:
            k = str(max(1, min(7, s)))
            dist[k] = dist.get(k, 0) + 1

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
            'rula_params':      dict(self.rula_params),
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
