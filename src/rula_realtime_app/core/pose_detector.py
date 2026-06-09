"""
骨架辨識模組
支援 MediaPipe 與 RTMW3D（rtmlib）
"""

import cv2
import numpy as np

# 延遲匯入 mediapipe，避免初始化問題
import mediapipe as mp

from .config import (
    MEDIAPIPE_CONFIG,
    RTMW3D_CONFIG,
    RTMW_TO_MEDIAPIPE,
    RTMW_RULA_CONNECTIONS,
    RTMW_RULA_KEYPOINTS,
    MP_RULA_CONNECTIONS,
    MP_RULA_KEYPOINTS,
    convert_indexed_keypoints_to_pose33,
)


def draw_rula_skeleton(image_rgb, kpts_px, scores, kpt_thr: float = 0.3,
                       line_color=(0, 220, 120), joint_color=(0, 160, 255)):
    """
    只繪製 RULA 實際分析/使用的 RTMW 關鍵點與連線（見 RTMW_RULA_CONNECTIONS）。

    取代 rtmlib 預設的全身 133 點繪製，使「畫出來的骨架」= 「被分析的骨架」，
    避免未被異常判定涵蓋的指尖/臉部點誤導使用者。就地繪製於傳入的 RGB 影像。

    線段門檻（_ANOM_VIS_TH=0.5）比節點門檻（kpt_thr）更嚴格：
    兩端節點都 >= 0.5 才畫線，避免低可信度誤偵測點拉出奇怪的骨架線。
    節點圓點仍以較低的 kpt_thr 繪製（方便與 X 標記對位）。

    Args:
        image_rgb: RGB 影像（numpy array），就地繪製。
        kpts_px:   RTMW 原生關鍵點像素座標，[K, 2]。
        scores:    每點信心度，[K]；低於 kpt_thr 的點不畫節點，低於 0.5 的點不畫連線。
    """
    if kpts_px is None:
        return image_rgb
    n = len(kpts_px)
    from .video_file_processor import _ANOM_VIS_TH

    def _score(i: int) -> float:
        if i < 0 or i >= n:
            return 0.0
        return float(scores[i]) if scores is not None and i < len(scores) else 1.0

    for a, b in RTMW_RULA_CONNECTIONS:
        # 只有兩端信心度都達到異常判定門檻，才畫連線（避免誤偵測點拉出奇怪的骨架線）
        if _score(a) >= _ANOM_VIS_TH and _score(b) >= _ANOM_VIS_TH:
            pa = (int(kpts_px[a][0]), int(kpts_px[a][1]))
            pb = (int(kpts_px[b][0]), int(kpts_px[b][1]))
            cv2.line(image_rgb, pa, pb, line_color, 2, cv2.LINE_AA)
    for i in RTMW_RULA_KEYPOINTS:
        if _score(i) >= kpt_thr:
            cv2.circle(image_rgb, (int(kpts_px[i][0]), int(kpts_px[i][1])),
                       4, joint_color, -1, cv2.LINE_AA)
    return image_rgb


def draw_rula_skeleton_mp(image_rgb: np.ndarray,
                          lms_2d: list,
                          kpt_thr: float = 0.3,
                          vis_thr: float = 0.5,
                          line_color=(0, 220, 120),
                          joint_color=(0, 160, 255)) -> np.ndarray:
    """
    只繪製 RULA 相關的 MediaPipe 關鍵點與連線（所見即所判）。
    lms_2d: list of [x_norm, y_norm, visibility]，至少 25 個元素（需涵蓋 index 24）。
    """
    if not lms_2d or len(lms_2d) < 25:
        return image_rgb
    h, w = image_rgb.shape[:2]

    def _px(i):
        lm = lms_2d[i]
        return (int(lm[0] * w), int(lm[1] * h))

    def _vis(i):
        if i >= len(lms_2d):
            return 0.0
        lm = lms_2d[i]
        return float(lm[2]) if len(lm) >= 3 else 1.0

    for a, b in MP_RULA_CONNECTIONS:
        if _vis(a) >= vis_thr and _vis(b) >= vis_thr:
            cv2.line(image_rgb, _px(a), _px(b), line_color, 2, cv2.LINE_AA)
    for i in MP_RULA_KEYPOINTS:
        if _vis(i) >= kpt_thr:
            cv2.circle(image_rgb, _px(i), 4, joint_color, -1, cv2.LINE_AA)
    return image_rgb


def _bbox_from_keypoints_2d(kpts_2d: np.ndarray) -> np.ndarray:
    """Convert one person's 2D keypoints to bbox [x1, y1, x2, y2]."""
    x = kpts_2d[:, 0]
    y = kpts_2d[:, 1]
    return np.array([x.min(), y.min(), x.max(), y.max()], dtype=np.float32)


def _bbox_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU for bboxes in [x1, y1, x2, y2]."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _select_single_target(keypoints_2d, scores, last_target_bbox, iou_thr=0.05):
    """Keep one target to reduce ID switching in multi-person scenes."""
    if keypoints_2d is None:
        return [], [], last_target_bbox, None

    keypoints_2d = np.asarray(keypoints_2d)
    scores = np.asarray(scores)
    if keypoints_2d.ndim < 3 or len(keypoints_2d) == 0:
        return [], [], last_target_bbox, None

    candidate_bboxes = np.asarray([_bbox_from_keypoints_2d(kpts) for kpts in keypoints_2d])
    candidate_areas = (candidate_bboxes[:, 2] - candidate_bboxes[:, 0]) * (candidate_bboxes[:, 3] - candidate_bboxes[:, 1])
    candidate_scores = np.asarray([float(np.mean(s)) for s in scores])

    if last_target_bbox is None:
        rank = candidate_scores * np.sqrt(np.maximum(candidate_areas, 1.0))
        best_idx = int(np.argmax(rank))
    else:
        ious = np.asarray([_bbox_iou_xyxy(b, last_target_bbox) for b in candidate_bboxes])
        best_iou_idx = int(np.argmax(ious))

        if ious[best_iou_idx] >= iou_thr:
            best_idx = best_iou_idx
        else:
            target_center = np.array([
                (last_target_bbox[0] + last_target_bbox[2]) * 0.5,
                (last_target_bbox[1] + last_target_bbox[3]) * 0.5,
            ])
            centers = np.stack([
                (candidate_bboxes[:, 0] + candidate_bboxes[:, 2]) * 0.5,
                (candidate_bboxes[:, 1] + candidate_bboxes[:, 3]) * 0.5,
            ], axis=1)
            dists = np.linalg.norm(centers - target_center, axis=1)
            best_idx = int(np.argmin(dists))

    selected_keypoints_2d = keypoints_2d[best_idx:best_idx + 1]
    selected_scores = scores[best_idx:best_idx + 1]
    updated_bbox = candidate_bboxes[best_idx]
    return selected_keypoints_2d, selected_scores, updated_bbox, best_idx


class PoseDetector:
    """骨架辨識器：支援 MediaPipe 與 RTMW3D。"""

    def __init__(self, backend_mode='MEDIAPIPE', mp_model_complexity: int | None = None):
        """初始化骨架辨識器。"""
        self.backend_mode = backend_mode.upper()
        self._mp_model_complexity = mp_model_complexity
        if self.backend_mode in ('RTMPOSEW2D', 'RTMW'):
            self.backend_mode = 'RTMW2D'

        self.results = None
        self.last_pose33 = None
        self.last_keypoints_2d = None
        self.last_scores = None
        self.last_target_bbox = None
        self.last_raw_3d = None       # raw RTMW3D 3D keypoints [K, 3+] for selected person

        self.pose = None
        self.pose_tracker = None
        self.draw_skeleton = None

        if self.backend_mode in ('RTMW2D', 'RTMW3D'):
            self._init_rtmw3d()
        else:
            self._init_mediapipe()

    def _init_mediapipe(self):
        """初始化 MediaPipe Pose。"""
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles

        complexity = self._mp_model_complexity
        if complexity is None:
            complexity = MEDIAPIPE_CONFIG['model_complexity']
        self.pose = mp_pose.Pose(
            static_image_mode=MEDIAPIPE_CONFIG['static_image_mode'],
            model_complexity=int(complexity),
            smooth_landmarks=MEDIAPIPE_CONFIG['smooth_landmarks'],
            enable_segmentation=MEDIAPIPE_CONFIG['enable_segmentation'],
            smooth_segmentation=MEDIAPIPE_CONFIG['smooth_segmentation'],
            min_detection_confidence=MEDIAPIPE_CONFIG['min_detection_confidence'],
            min_tracking_confidence=MEDIAPIPE_CONFIG['min_tracking_confidence']
        )

    def _init_rtmw3d(self):
        """初始化 RTMW3D PoseTracker。"""
        from rtmlib import Wholebody, Wholebody3d, draw_skeleton
        from rtmlib.tools.solution.pose_tracker import (
            PoseTracker as RTMLibPoseTracker,
            pose_to_bbox,
        )

        class PatchedPoseTracker(RTMLibPoseTracker):
            """Fix tracking reorder by using matched detection indices."""

            def __call__(self, image: np.ndarray):
                pose_model_name = type(self.pose_model).__name__
                keypoints_simcc = None
                keypoints2d = None

                if self.det_model:
                    if self.frame_cnt % self.det_frequency == 0:
                        try:
                            if self.det_categories or self.det_mode == 'multiclass':
                                if self.det_categories:
                                    bboxes, classes = self.det_model(image)
                                    bboxes = [
                                        bbox for bbox, cls in zip(bboxes, classes)
                                        if cls in self.det_categories
                                    ]
                                else:
                                    bboxes, _ = self.det_model(image)
                            else:
                                bboxes = self.det_model(image)
                        except Exception:
                            return [], []
                    else:
                        bboxes = self.bboxes_last_frame

                    if pose_model_name == 'RTMPose3d':
                        keypoints, scores, keypoints_simcc, keypoints2d = self.pose_model(
                            image, bboxes=bboxes)
                    else:
                        keypoints, scores = self.pose_model(image, bboxes=bboxes)
                else:
                    keypoints, scores = self.pose_model(image)

                if not self.tracking and self.det_frequency != 1:
                    bboxes_current_frame = []
                    if pose_model_name == 'RTMPose3d':
                        for kpts in keypoints2d:
                            bboxes_current_frame.append(pose_to_bbox(kpts))
                    else:
                        for kpts in keypoints:
                            bboxes_current_frame.append(pose_to_bbox(kpts))
                else:
                    if len(self.track_ids_last_frame) == 0:
                        self.next_id = len(self.bboxes_last_frame)
                        self.track_ids_last_frame = list(range(self.next_id))

                    bboxes_current_frame = []
                    track_ids_current_frame = []
                    matched_indices = []

                    for det_idx, kpts in enumerate(keypoints):
                        bbox = pose_to_bbox(kpts)
                        track_id, _ = self.track_by_iou(bbox)

                        if track_id > -1:
                            matched_indices.append(det_idx)
                            track_ids_current_frame.append(track_id)
                            bboxes_current_frame.append(bbox)

                    self.track_ids_last_frame = track_ids_current_frame

                    keypoints = np.asarray(keypoints)
                    scores = np.asarray(scores)
                    if matched_indices:
                        keypoints = keypoints[matched_indices]
                        scores = scores[matched_indices]
                        if pose_model_name == 'RTMPose3d':
                            keypoints2d = np.asarray(keypoints2d)[matched_indices]
                            keypoints_simcc = np.asarray(keypoints_simcc)[matched_indices]
                    else:
                        keypoints = keypoints[:0]
                        scores = scores[:0]
                        if pose_model_name == 'RTMPose3d':
                            keypoints2d = np.asarray(keypoints2d)[:0]
                            keypoints_simcc = np.asarray(keypoints_simcc)[:0]

                self.bboxes_last_frame = bboxes_current_frame
                self.frame_cnt += 1

                if pose_model_name == 'RTMPose3d':
                    return keypoints, scores, keypoints_simcc, keypoints2d

                return keypoints, scores

        self.draw_skeleton = draw_skeleton

        tracker_device = RTMW3D_CONFIG.get('device', 'cuda')
        if tracker_device == 'auto':
            tracker_device = 'cuda'

        pose_model = Wholebody3d if self.backend_mode == 'RTMW3D' else Wholebody

        try:
            self.pose_tracker = PatchedPoseTracker(
                pose_model,
                det_frequency=RTMW3D_CONFIG.get('det_frequency', 1),
                tracking=RTMW3D_CONFIG.get('tracking', True),
                backend=RTMW3D_CONFIG.get('backend', 'onnxruntime'),
                device=tracker_device,
            )
        except Exception:
            self.pose_tracker = PatchedPoseTracker(
                pose_model,
                det_frequency=RTMW3D_CONFIG.get('det_frequency', 1),
                tracking=RTMW3D_CONFIG.get('tracking', True),
                backend=RTMW3D_CONFIG.get('backend', 'onnxruntime'),
                device='cpu',
            )

    def process_frame(self, frame):
        """
        處理單一影像幀。

        Args:
            frame: RGB 格式的影像（numpy array）

        Returns:
            bool: 是否成功偵測到骨架
        """
        h, w = frame.shape[:2]
        self._last_image_wh = (w, h)

        if self.backend_mode in ('RTMW2D', 'RTMW3D'):
            return self._process_rtmw3d(frame)

        self.results = self.pose.process(frame)
        return self.results.pose_landmarks is not None

    def _process_rtmw3d(self, frame_rgb):
        """RTMW3D 單幀推理，並轉成 MediaPipe-like 33 點。"""
        if self.pose_tracker is None:
            return False

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        result = self.pose_tracker(frame_bgr)
        keypoints_3d = None
        keypoints_2d = None
        scores = None

        if isinstance(result, tuple) and len(result) == 4:
            keypoints_3d, scores, _, keypoints_2d = result
        elif isinstance(result, tuple) and len(result) == 2:
            keypoints_3d, scores = result
            if hasattr(keypoints_3d, 'ndim') and keypoints_3d.ndim >= 3 and keypoints_3d.shape[-1] >= 2:
                keypoints_2d = keypoints_3d[..., :2]
            else:
                keypoints_2d = []
        else:
            self.last_pose33 = None
            self.last_keypoints_2d = None
            self.last_scores = None
            return False

        keypoints_2d, scores, self.last_target_bbox, selected_idx = _select_single_target(
            keypoints_2d,
            scores,
            self.last_target_bbox,
            iou_thr=RTMW3D_CONFIG.get('iou_threshold', 0.05),
        )

        if len(keypoints_2d) == 0:
            self.last_pose33 = None
            self.last_keypoints_2d = None
            self.last_scores = None
            return False

        selected_2d = np.asarray(keypoints_2d[0])
        selected_scores = np.asarray(scores[0])

        selected_3d = None
        if keypoints_3d is not None and selected_idx is not None:
            keypoints_3d = np.asarray(keypoints_3d)
            if (
                keypoints_3d.ndim == 3
                and keypoints_3d.shape[0] > selected_idx
                and keypoints_3d.shape[1] == selected_2d.shape[0]
            ):
                selected_3d = keypoints_3d[selected_idx]

        if selected_3d is not None and selected_3d.ndim == 2 and selected_3d.shape[1] >= 3:
            source_xyz = selected_3d[:, :3]
        else:
            source_xyz = np.concatenate(
                [selected_2d[:, :2], np.zeros((selected_2d.shape[0], 1), dtype=np.float32)],
                axis=1,
            )

        self.last_pose33 = convert_indexed_keypoints_to_pose33(
            source_xyz,
            selected_scores,
            RTMW_TO_MEDIAPIPE,
        )

        self.last_keypoints_2d = np.asarray(keypoints_2d)
        self.last_scores = np.asarray(scores)
        # Store raw 3D for full-body rendering (all K keypoints, not just 33 mapped ones)
        if (
            self.backend_mode == 'RTMW3D'
            and selected_3d is not None
            and selected_3d.ndim == 2
            and selected_3d.shape[1] >= 3
        ):
            self.last_raw_3d = selected_3d
        else:
            self.last_raw_3d = None
        return True

    def get_native_draw_data_2d(self):
        """
        取得可供「原生繪圖器」重現骨架的 2D 資料（含完整關節）。

        Returns:
            dict | None:
                - RTMW3D:
                    {
                        'backend': 'RTMW3D',
                        'keypoints_2d_norm': [[x_norm, y_norm], ...],
                        'scores': [conf, ...]
                    }
                - MEDIAPIPE:
                    {
                        'backend': 'MEDIAPIPE',
                        'landmarks_2d': [[x_norm, y_norm, visibility], ...]
                    }
        """
        if self.backend_mode in ('RTMW2D', 'RTMW3D'):
            if self.last_keypoints_2d is None or self.last_scores is None:
                return None

            w, h = getattr(self, '_last_image_wh', (1, 1))
            if w <= 0 or h <= 0:
                return None

            selected_2d = np.asarray(self.last_keypoints_2d[0])
            selected_sc = np.asarray(self.last_scores[0]).reshape(-1)

            kpts_norm = []
            for i in range(selected_2d.shape[0]):
                x_px, y_px = float(selected_2d[i, 0]), float(selected_2d[i, 1])
                x_norm = x_px / w
                y_norm = y_px / h
                kpts_norm.append([x_norm, y_norm])

            scores = []
            for i in range(selected_sc.shape[0]):
                conf = max(0.0, min(1.0, float(selected_sc[i])))
                scores.append(conf)

            # Build raw_3d list: [[x, y, z], ...] for all K joints
            raw_3d = []
            if self.last_raw_3d is not None:
                r3d = np.asarray(self.last_raw_3d)
                for i in range(r3d.shape[0]):
                    raw_3d.append([float(r3d[i, 0]), float(r3d[i, 1]), float(r3d[i, 2])])

            return {
                'backend': self.backend_mode,
                'keypoints_2d_norm': kpts_norm,
                'scores': scores,
                'keypoints_3d_raw': raw_3d,  # full raw 3D from model (len == K)
            }

        if self.results is None or self.results.pose_landmarks is None:
            return None

        return {
            'backend': 'MEDIAPIPE',
            'landmarks_2d': [
                [float(lm.x), float(lm.y), float(lm.visibility)]
                for lm in self.results.pose_landmarks.landmark
            ],
        }

    def get_landmarks_array(self):
        """取得關鍵點陣列（用於 RULA 計算）。"""
        return self.get_rula_landmarks('3D')

    def get_rula_landmarks(self, analysis_mode='2D'):
        """Return MediaPipe-like 33 landmarks for RULA angle calculation."""
        mode = str(analysis_mode or '2D').upper()

        if self.backend_mode in ('RTMW2D', 'RTMW3D'):
            if mode == '3D':
                return None
            return self._get_rtmw_pose33_2d_pixels()

        if mode == '2D':
            return self._get_mediapipe_pose33_2d_pixels()

        if self.results is None or self.results.pose_world_landmarks is None:
            return None

        landmarks = []
        for lm in self.results.pose_world_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z, lm.visibility])

        return landmarks

    def _get_mediapipe_pose33_2d_pixels(self):
        if self.results is None or self.results.pose_landmarks is None:
            return None

        w, h = getattr(self, '_last_image_wh', (1, 1))
        if w <= 0 or h <= 0:
            return None

        landmarks = []
        for lm in self.results.pose_landmarks.landmark:
            landmarks.append([
                float(lm.x) * float(w),
                float(lm.y) * float(h),
                0.0,
                float(lm.visibility),
            ])
        return landmarks

    def _get_rtmw_pose33_2d_pixels(self):
        if self.last_keypoints_2d is None or self.last_scores is None:
            return None

        selected_2d = np.asarray(self.last_keypoints_2d[0])
        if selected_2d.ndim != 2 or selected_2d.shape[1] < 2:
            return None

        selected_scores = np.asarray(self.last_scores[0]).reshape(-1)
        source_xyz = np.concatenate(
            [
                selected_2d[:, :2].astype(np.float32),
                np.zeros((selected_2d.shape[0], 1), dtype=np.float32),
            ],
            axis=1,
        )
        return convert_indexed_keypoints_to_pose33(
            source_xyz,
            selected_scores,
            RTMW_TO_MEDIAPIPE,
        )

    def draw_landmarks(self, image):
        """
        在影像上繪製骨架關鍵點。

        Args:
            image: RGB 格式的影像（numpy array）

        Returns:
            numpy.ndarray: 繪製後的影像
        """
        if self.backend_mode in ('RTMW2D', 'RTMW3D'):
            if self.last_keypoints_2d is None or self.last_scores is None:
                return image

            # 只畫 RULA 實際分析/使用的點，而非 rtmlib 的全身 133 點，
            # 讓畫面與異常判定範圍一致。
            annotated = image.copy()
            kpts_px = np.asarray(self.last_keypoints_2d[0])
            scores  = np.asarray(self.last_scores[0]).reshape(-1)
            return draw_rula_skeleton(
                annotated, kpts_px, scores,
                kpt_thr=RTMW3D_CONFIG.get('kpt_threshold', 0.3),
            )

        if self.results is None or self.results.pose_landmarks is None:
            return image

        lms_2d = [
            [lm.x, lm.y, lm.visibility]
            for lm in self.results.pose_landmarks.landmark
        ]
        return draw_rula_skeleton_mp(image.copy(), lms_2d)

    def close(self):
        """釋放資源。"""
        if self.pose:
            self.pose.close()
