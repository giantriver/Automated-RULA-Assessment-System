"""
Frame processor worker - offloads heavy ML inference to a background thread.

Architecture:
    CameraHandler (QThread)
        │ frame_ready (DirectConnection)
        ▼
    RealtimeProcessorWorker.set_latest_frame()   ← just stores latest frame, no queue
        │ QTimer polling
        ▼
    _poll() → _do_process()                    ← runs on worker thread
        │ frame_processed signal
        ▼
    MainWindow.on_frame_processed()            ← UI update only, on main thread

Key design: camera uses DirectConnection so frames are written directly into a
threading.Lock-protected buffer instead of accumulating in the Qt event queue.
The timer polls this buffer at a fixed rate, naturally rate-limiting processing.
"""

import threading

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from . import angle_calc
from .pose_detector import PoseDetector


class RealtimeProcessorWorker(QObject):
    """
    Worker that runs ML inference and RULA calculation on a background thread.

    Frames from the camera are stored in a single-slot buffer (latest frame only).
    A QTimer polls this buffer at `poll_interval_ms` and processes one frame per tick.
    If the model is slower than the timer interval, the _busy flag skips the tick
    and the timer will try again on the next interval.
    """

    # (annotated_frame, rula_left|None, rula_right|None, landmarks|None)
    frame_processed = pyqtSignal(np.ndarray, object, object, object)
    fps_updated = pyqtSignal(float)

    def __init__(self, pose_detector_or_mode, display_mode: str,
                 rula_calc_every_n_frames: int, poll_interval_ms: int = 33,
                 analysis_mode: str = '2D'):
        """
        Args:
            pose_detector_or_mode: 已建立的 PoseDetector，或 backend_mode 字串
                ('MEDIAPIPE' / 'RTMW3D')。傳入字串時會在 worker 執行緒延遲初始化，
                避免在主執行緒載入大型 ONNX 模型。
        """
        super().__init__()
        if isinstance(pose_detector_or_mode, str):
            self._backend_mode = pose_detector_or_mode
            self.pose_detector = None  # 延遲到 start_timer() 建立
        else:
            self._backend_mode = None
            self.pose_detector = pose_detector_or_mode
        self.display_mode = display_mode
        self.analysis_mode = str(analysis_mode or '2D').upper()
        self.rula_calc_every_n_frames = rula_calc_every_n_frames

        # Thread-safe latest-frame buffer
        self._latest_frame: np.ndarray | None = None
        self._lock = threading.Lock()

        self._busy = False
        self._frame_counter = 0
        self.prev_left = None
        self.prev_right = None

        # FPS tracking
        self._fps_counter = 0
        self._fps_timer = cv2.getTickCount()

        self._poll_interval_ms = poll_interval_ms
        # 不在這裡建立 QTimer — 在 start_timer()（worker 執行緒）建立，確保執行緒歸屬正確

    # ------------------------------------------------------------------
    # Called from camera thread via Qt.ConnectionType.DirectConnection
    # Must be fast and thread-safe (only writes to _latest_frame).
    # ------------------------------------------------------------------
    def set_latest_frame(self, frame: np.ndarray):
        with self._lock:
            self._latest_frame = frame

    # ------------------------------------------------------------------
    # Timer management (called on worker thread via thread.started signal)
    # ------------------------------------------------------------------
    @pyqtSlot()
    def start_timer(self):
        # 如果傳入的是 backend_mode 字串，在 worker 執行緒這裡才初始化（避免阻塞主執行緒）
        if self._backend_mode is not None and self.pose_detector is None:
            self.pose_detector = PoseDetector(backend_mode=self._backend_mode)
        # 在 worker 執行緒建立 QTimer，確保執行緒歸屬正確
        self._timer = QTimer()
        self._timer.setInterval(self._poll_interval_ms)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    @pyqtSlot()
    def stop_timer(self):
        if hasattr(self, '_timer') and self._timer is not None:
            self._timer.stop()

    # ------------------------------------------------------------------
    # Timer tick - runs on worker thread
    # ------------------------------------------------------------------
    def _poll(self):
        if self._busy or self.pose_detector is None:
            return  # Model still initializing or running; skip this tick

        with self._lock:
            frame = self._latest_frame
            self._latest_frame = None

        if frame is None:
            return  # No new frame since last poll

        self._busy = True
        try:
            self._do_process(frame)
        finally:
            self._busy = False

    def _do_process(self, frame: np.ndarray):
        self._frame_counter += 1

        # Track processed FPS（每秒更新一次，不論幀率高低）
        self._fps_counter += 1
        current_time = cv2.getTickCount()
        elapsed = (current_time - self._fps_timer) / cv2.getTickFrequency()
        if elapsed >= 1.0:
            self.fps_updated.emit(self._fps_counter / elapsed)
            self._fps_counter = 0
            self._fps_timer = current_time

        detected = self.pose_detector.process_frame(frame)

        rula_left = None
        rula_right = None
        landmarks = None

        if detected:
            annotated = self.pose_detector.draw_landmarks(frame)

            if self.display_mode == "RULA":
                if self._frame_counter % self.rula_calc_every_n_frames == 0:
                    landmarks_arr = self.pose_detector.get_rula_landmarks(self.analysis_mode)
                    rula_left, rula_right = angle_calc(
                        landmarks_arr, self.prev_left, self.prev_right,
                        analysis_mode=self.analysis_mode,
                    )
                    self.prev_left = rula_left
                    self.prev_right = rula_right
            else:  # COORDINATES
                landmarks = self.pose_detector.get_rula_landmarks(self.analysis_mode)
        else:
            annotated = frame

        self.frame_processed.emit(annotated, rula_left, rula_right, landmarks)
