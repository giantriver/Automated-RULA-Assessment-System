"""
影片分析結果視窗模組。

負責呈現離線分析完成後的結果頁，包含：
- 取樣幀播放與骨架疊加檢視
- 分數折線圖與分布長條圖
- 關鍵統計資訊卡片
- 匯出 CSV
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QSizePolicy, QCheckBox,
    QFileDialog, QMessageBox, QScrollArea, QSplitter, QSlider, QTabWidget,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont

from ..core.video_file_processor import export_csv
from .styles import (
    UPLOAD_BG_STYLE, CONTENT_CARD_STYLE, HEADER_CARD_STYLE,
    BACK_BTN_STYLE, EMERALD_BTN_STYLE,
)
from .language import language_manager, t
from .dialogs import FrameMetricsDialog


_SCORE_COLORS = {
    1: ('#d1fae5', '#065f46'), 2: ('#d1fae5', '#065f46'),
    3: ('#fef3c7', '#92400e'), 4: ('#fef3c7', '#92400e'),
    5: ('#fee2e2', '#991b1b'), 6: ('#fee2e2', '#991b1b'),
    7: ('#fca5a5', '#7c2d12'),
}


def _setup_matplotlib_cjk_font() -> None:
    """Set a CJK-capable font so Chinese labels render in matplotlib."""
    candidates = [
        'Microsoft JhengHei',
        'Microsoft YaHei',
        'Noto Sans CJK TC',
        'PingFang TC',
        'Heiti TC',
        'SimHei',
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available:
            matplotlib.rcParams['font.family'] = [font_name, 'DejaVu Sans']
            break
    else:
        matplotlib.rcParams['font.family'] = ['DejaVu Sans']

    # Avoid garbled minus signs when using CJK fonts.
    matplotlib.rcParams['axes.unicode_minus'] = False


_setup_matplotlib_cjk_font()


# BlazePose 33-keypoint connections (MediaPipe index pairs + color)
_SKELETON_CONNECTIONS = [
    # head
    (0, 1, 'cyan'), (1, 2, 'cyan'), (2, 3, 'cyan'), (3, 7, 'cyan'),
    (0, 4, 'cyan'), (4, 5, 'cyan'), (5, 6, 'cyan'), (6, 8, 'cyan'),
    (9, 10, 'cyan'),
    # torso
    (11, 12, 'yellow'), (11, 23, 'yellow'), (12, 24, 'yellow'), (23, 24, 'yellow'),
    # left arm
    (11, 13, '#4dabf7'), (13, 15, '#4dabf7'), (15, 17, '#4dabf7'),
    (15, 19, '#4dabf7'), (15, 21, '#4dabf7'), (17, 19, '#4dabf7'),
    # right arm
    (12, 14, '#ff6b6b'), (14, 16, '#ff6b6b'), (16, 18, '#ff6b6b'),
    (16, 20, '#ff6b6b'), (16, 22, '#ff6b6b'), (18, 20, '#ff6b6b'),
    # left leg
    (23, 25, '#4dabf7'), (25, 27, '#4dabf7'), (27, 29, '#4dabf7'),
    (27, 31, '#4dabf7'), (29, 31, '#4dabf7'),
    # right leg
    (24, 26, '#ff6b6b'), (26, 28, '#ff6b6b'), (28, 30, '#ff6b6b'),
    (28, 32, '#ff6b6b'), (30, 32, '#ff6b6b'),
]

# RTMW / COCO-WholeBody body-only connections (indices 0-16 = COCO-17)
# COCO-17: 0=nose,1=l_eye,2=r_eye,3=l_ear,4=r_ear,5=l_sho,6=r_sho,
#          7=l_elb,8=r_elb,9=l_wri,10=r_wri,11=l_hip,12=r_hip,
#          13=l_kne,14=r_kne,15=l_ank,16=r_ank
_RTMW_BODY_CONNECTIONS = [
    # head
    (0, 1, 'cyan'), (0, 2, 'cyan'), (1, 3, 'cyan'), (2, 4, 'cyan'),
    # torso
    (5, 6, 'yellow'), (5, 11, 'yellow'), (6, 12, 'yellow'), (11, 12, 'yellow'),
    # left arm
    (5, 7, '#4dabf7'), (7, 9, '#4dabf7'),
    # right arm
    (6, 8, '#ff6b6b'), (8, 10, '#ff6b6b'),
    # left leg
    (11, 13, '#4dabf7'), (13, 15, '#4dabf7'),
    # right leg
    (12, 14, '#ff6b6b'), (14, 16, '#ff6b6b'),
]


def _render_3d_skeleton_pixmap(landmarks_3d: list,
                               width: int = 340,
                               height: int = 380,
                               native: dict = None) -> 'QPixmap | None':
    """
    Render a 3D skeleton into a QPixmap.

    For RTMW3D, uses ``native['keypoints_3d_raw']`` (all model joints + scores
    from ``native['scores']``) with COCO-17 body connections.
    Falls back to ``landmarks_3d`` (33-point MediaPipe format) otherwise.
    Returns None when no usable data is present.
    """
    from io import BytesIO

    # ── Choose data source ────────────────────────────────────────────────────
    # For RTMW3D use the full raw 3D keypoints (all K model joints) so the
    # skeleton shows the whole body, not just the 15 RULA-mapped points.
    raw_3d = (native or {}).get('keypoints_3d_raw') or []
    raw_scores = (native or {}).get('scores') or []

    if raw_3d and len(raw_3d) >= 17:
        # RTMW3D path — use raw model outputs
        arr3 = np.asarray(raw_3d, dtype=np.float64)          # [K, 3]
        sc   = np.asarray(raw_scores, dtype=np.float64)
        if sc.shape[0] < arr3.shape[0]:
            sc = np.pad(sc, (0, arr3.shape[0] - sc.shape[0]))

        xs, ys, zs, vs = arr3[:, 0], arr3[:, 1], arr3[:, 2], sc
        connections = _RTMW_BODY_CONNECTIONS
    else:
        # MediaPipe fallback
        if not landmarks_3d or len(landmarks_3d) < 33:
            return None
        arr = np.asarray(landmarks_3d, dtype=np.float64)
        xs, ys, zs, vs = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        connections = _SKELETON_CONNECTIONS

    # ── Normalise pixel-scale coordinates ────────────────────────────────────
    # RTMW3D X/Y are in pixel space; Z uses a different (smaller) unit.
    # Normalise so the plot is proportional regardless of backend.
    _vis = vs > 0.3
    if _vis.any() and max(np.ptp(xs[_vis]), np.ptp(ys[_vis])) > 5.0:
        _xy_span = max(np.ptp(xs[_vis]), np.ptp(ys[_vis]))
        _cx, _cy, _cz = xs[_vis].mean(), ys[_vis].mean(), zs[_vis].mean()
        xs = (xs - _cx) / _xy_span
        ys = (ys - _cy) / _xy_span
        zs = (zs - _cz) / _xy_span
        _z_rng  = np.ptp(zs[_vis])
        _xy_rng = max(np.ptp(xs[_vis]), np.ptp(ys[_vis]))
        if _z_rng < _xy_rng * 0.1 and _z_rng > 1e-6:
            zs = zs * min(_xy_rng / _z_rng * 0.3, 10.0)

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    fig.subplots_adjust(left=0.0, right=0.82, top=0.99, bottom=0.08)

    for i, j, color in connections:
        if i < len(vs) and j < len(vs) and vs[i] > 0.3 and vs[j] > 0.3:
            # darken cyan/yellow for white bg readability
            c = {'cyan': '#0891b2', 'yellow': '#b45309'}.get(color, color)
            ax.plot(
                [xs[i], xs[j]], [zs[i], zs[j]], [-ys[i], -ys[j]],
                color=c, linewidth=2.2, alpha=0.9,
            )

    cmap = plt.cm.plasma
    norm = plt.Normalize(0.0, 1.0)
    for k in range(len(vs)):
        if vs[k] > 0.3:
            ax.scatter(xs[k], zs[k], -ys[k],
                       color=cmap(norm(vs[k])), s=18, zorder=5, depthshade=False,
                       edgecolors='#334155', linewidths=0.4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.05, fraction=0.03, shrink=0.55)
    cbar.set_label('Confidence', color='#334155', fontsize=7)
    cbar.ax.yaxis.set_tick_params(color='#334155')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#334155', fontsize=6)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(['Low', '0.5', 'High'])

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.label.set_color('#334155')
        axis.set_tick_params(colors='#64748b', labelsize=5)
        axis.pane.fill = False
        axis.pane.set_edgecolor('#cbd5e1')
        axis._axinfo['grid']['color'] = '#e2e8f0'

    ax.set_xlabel('X', fontsize=6, color='#334155', labelpad=2)
    ax.set_ylabel('Z', fontsize=6, color='#334155', labelpad=2)
    ax.set_zlabel('Y', fontsize=6, color='#334155', labelpad=2)

    # auto-fit axis limits from visible joints with a small margin
    _vis_mask = vs > 0.3
    if _vis_mask.any():
        _pad = 0.15
        x_min = xs[_vis_mask].min() - _pad
        x_max = xs[_vis_mask].max() + _pad
        y_min = zs[_vis_mask].min() - _pad
        y_max = zs[_vis_mask].max() + _pad
        z_min = -ys[_vis_mask].max() - _pad
        z_max = -ys[_vis_mask].min() + _pad
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_box_aspect([x_max - x_min, y_max - y_min, z_max - z_min])
    else:
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(-1.2, 0.4)
        ax.set_box_aspect([1.2, 1.2, 1.6])
    ax.view_init(elev=10, azim=-65)

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2,
                facecolor=fig.get_facecolor(), dpi=dpi)
    plt.close(fig)
    buf.seek(0)

    pixmap = QPixmap()
    pixmap.loadFromData(buf.getvalue(), 'PNG')
    return pixmap if not pixmap.isNull() else None


def _draw_mediapipe_skeleton(frame_rgb: np.ndarray,
                             landmarks_2d: list,
                             min_visibility: float = 0.0) -> np.ndarray:
    """Draw MediaPipe skeleton with MediaPipe's native drawer."""
    if not landmarks_2d or len(landmarks_2d) < 33:
        return frame_rgb

    annotated = frame_rgb.copy()
    lm_proto = landmark_pb2.NormalizedLandmarkList()
    for lm in landmarks_2d:
        if len(lm) < 3:
            continue
        lm_proto.landmark.add(
            x=float(lm[0]),
            y=float(lm[1]),
            z=0.0,
            visibility=float(lm[2]),
            presence=float(lm[2]),
        )

    mp.solutions.drawing_utils.draw_landmarks(
        annotated,
        lm_proto,
        mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
    )
    return annotated


def _draw_rtmw_skeleton(frame_rgb: np.ndarray,
                        keypoints_2d_norm: list,
                        scores: list,
                        kpt_threshold: float = 0.3) -> np.ndarray:
    """Draw RTMW skeleton with rtmlib native drawer."""
    if not keypoints_2d_norm or not scores:
        return frame_rgb

    try:
        from rtmlib import draw_skeleton
    except Exception:
        return frame_rgb

    h, w = frame_rgb.shape[:2]
    kpts_px = []
    for pt in keypoints_2d_norm:
        if len(pt) < 2:
            kpts_px.append([0.0, 0.0])
            continue
        kpts_px.append([float(pt[0]) * w, float(pt[1]) * h])

    kpts_arr = np.asarray([kpts_px], dtype=np.float32)
    scores_arr = np.asarray([scores], dtype=np.float32)

    image_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    drawn_bgr = draw_skeleton(
        image_bgr,
        kpts_arr,
        scores_arr,
        kpt_thr=kpt_threshold,
    )
    return cv2.cvtColor(drawn_bgr, cv2.COLOR_BGR2RGB)


def _frame_rgb_from_video(cap: cv2.VideoCapture,
                           frame_idx: int) -> np.ndarray | None:
    """Seek to `frame_idx` and return a decoded RGB frame, or None."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, bgr = cap.read()
    if not ret or bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ──────────────────────────────────────────────────────────────────────────────
class ResultWindow(QMainWindow):
    """
    獨立的分析結果視窗。

    Signals:
        back_requested: 使用者按「關閉」或視窗關閉
    """

    back_requested = pyqtSignal()

    def __init__(self, results: dict):
        super().__init__()
        language_manager.add_observer(self.on_language_changed)

        self._results   = results
        self._records   = results.get('records', [])
        self._video_path = results.get('video_path', '')
        self._backend_mode = str(results.get('backend_mode', 'MEDIAPIPE')).upper()
        self._cap: cv2.VideoCapture | None = None

        # playback state
        self._current_idx = 0
        self._show_skeleton = True
        self._play_timer    = QTimer()
        self._play_timer.setInterval(800)
        self._play_timer.timeout.connect(self._playback_tick)

        fname = results.get('original_filename', 'Result')
        self.setMinimumSize(1280, 780)
        self.resize(1440, 860)
        self.setStyleSheet(UPLOAD_BG_STYLE)

        self._open_video()
        self._init_ui()
        self._retranslate_ui()
        self._show_frame(0)

    # ── Video open/close ──────────────────────────────────────────────────────
    def _open_video(self):
        if self._video_path and os.path.exists(self._video_path):
            self._cap = cv2.VideoCapture(self._video_path)

    def _close_video(self):
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self._cap = None

    # ── UI ────────────────────────────────────────────────────────────────────
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        outer = QVBoxLayout(central)
        outer.setSpacing(16)
        outer.setContentsMargins(32, 24, 32, 24)

        outer.addWidget(self._build_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([640, 500])

        outer.addWidget(splitter, stretch=1)

    # ── Header ────────────────────────────────────────────────────────────────
    def _build_header(self) -> QFrame:
        card = QFrame()
        card.setObjectName('header_card')
        card.setStyleSheet(HEADER_CARD_STYLE)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        row = QHBoxLayout(card)
        row.setContentsMargins(20, 14, 20, 14)
        row.setSpacing(14)

        icon = QLabel('🎬')
        icon.setFixedSize(44, 44)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #3b82f6, stop:1 #2563eb);
                border-radius: 10px; font-size: 20px;
            }
        """)
        row.addWidget(icon)

        text_col = QVBoxLayout()
        meta = self._results.get('meta', {})

        self._header_title_lbl = QLabel()
        self._header_title_lbl.setFont(QFont('Microsoft JhengHei', 14, QFont.Weight.Bold))
        self._header_title_lbl.setStyleSheet('color: #0f172a;')
        text_col.addWidget(self._header_title_lbl)

        sub_parts = []
        if meta.get('survey_date'): sub_parts.append(meta['survey_date'])
        if meta.get('assessor'):    sub_parts.append(meta['assessor'])
        if meta.get('task_name'):   sub_parts.append(meta['task_name'])
        self._header_sub_lbl = QLabel('  |  '.join(sub_parts) if sub_parts else '')
        self._header_sub_lbl.setStyleSheet('color: #64748b; font-size: 12px;')
        text_col.addWidget(self._header_sub_lbl)

        row.addLayout(text_col)
        row.addStretch()

        self._export_btn = QPushButton()
        self._export_btn.setStyleSheet(EMERALD_BTN_STYLE)
        self._export_btn.setFixedWidth(130)
        self._export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_btn.clicked.connect(self._export_csv)
        row.addWidget(self._export_btn)

        self._close_btn = QPushButton()
        self._close_btn.setStyleSheet(BACK_BTN_STYLE)
        self._close_btn.setFixedWidth(90)
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_btn.clicked.connect(self.close)
        row.addWidget(self._close_btn)

        return card

    # ── Left panel: frame viewer + controls ───────────────────────────────────
    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet('background: transparent;')
        col = QVBoxLayout(panel)
        col.setSpacing(12)
        col.setContentsMargins(0, 0, 8, 0)

        # Frame display card
        frame_card = QFrame()
        frame_card.setObjectName('content_card')
        frame_card.setStyleSheet(CONTENT_CARD_STYLE)
        fc_layout = QVBoxLayout(frame_card)
        fc_layout.setContentsMargins(14, 14, 14, 14)
        fc_layout.setSpacing(10)

        self._frame_lbl = QLabel()
        self._frame_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_lbl.setMinimumHeight(340)
        self._frame_lbl.setStyleSheet(
            'background: #1a1a1a; border-radius: 8px; color: #94a3b8; font-size: 14px;'
        )
        fc_layout.addWidget(self._frame_lbl, stretch=1)

        skel_row = QHBoxLayout()
        self._skel_cb = QCheckBox()
        self._skel_cb.setChecked(True)
        self._skel_cb.setStyleSheet('color: #0f172a; font-size: 13px;')
        self._skel_cb.toggled.connect(self._on_skeleton_toggle)
        skel_row.addWidget(self._skel_cb)

        self._no_video_lbl = QLabel()
        self._no_video_lbl.setStyleSheet('color: #94a3b8; font-size: 11px;')
        skel_row.addWidget(self._no_video_lbl)
        skel_row.addStretch()
        fc_layout.addLayout(skel_row)

        col.addWidget(frame_card, stretch=1)

        # Playback controls card
        ctrl_card = QFrame()
        ctrl_card.setObjectName('content_card')
        ctrl_card.setStyleSheet(CONTENT_CARD_STYLE)
        cc = QVBoxLayout(ctrl_card)
        cc.setContentsMargins(14, 12, 14, 12)
        cc.setSpacing(10)

        self._ctrl_title_lbl = QLabel()
        self._ctrl_title_lbl.setFont(QFont('Microsoft JhengHei', 13, QFont.Weight.Bold))
        self._ctrl_title_lbl.setStyleSheet('color: #0f172a;')
        cc.addWidget(self._ctrl_title_lbl)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(8)

        self._prev_btn = QPushButton()
        self._prev_btn.setStyleSheet(BACK_BTN_STYLE)
        self._prev_btn.clicked.connect(self._prev_frame)
        nav_row.addWidget(self._prev_btn)

        self._play_btn = QPushButton()
        self._play_btn.setStyleSheet(EMERALD_BTN_STYLE)
        self._play_btn.clicked.connect(self._toggle_play)
        nav_row.addWidget(self._play_btn)

        self._next_btn = QPushButton()
        self._next_btn.setStyleSheet(BACK_BTN_STYLE)
        self._next_btn.clicked.connect(self._next_frame)
        nav_row.addWidget(self._next_btn)

        self._score_badge = QLabel('RULA: —')
        self._score_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._score_badge.setFixedHeight(24)
        self._score_badge.setStyleSheet(
            'background:#e2e8f0; color:#475569; border-radius:10px;'
            'padding:0 12px; font-size:12px; font-weight:bold;'
        )
        nav_row.addWidget(self._score_badge)

        self._metrics_btn = QPushButton()
        self._metrics_btn.setStyleSheet(BACK_BTN_STYLE)
        self._metrics_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._metrics_btn.clicked.connect(self._show_frame_metrics_dialog)
        nav_row.addWidget(self._metrics_btn)

        nav_row.addStretch()
        cc.addLayout(nav_row)

        # Frame scrub slider
        n = max(1, len(self._records) - 1)
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(n)
        self._frame_slider.setValue(0)
        self._frame_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px; background: #e2e8f0; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 14px; height: 14px; margin: -5px 0;
                background: #2563eb; border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #2563eb; border-radius: 2px;
            }
        """)
        self._frame_slider.valueChanged.connect(self._on_slider_changed)
        cc.addWidget(self._frame_slider)

        info_row = QHBoxLayout()
        self._frame_counter_lbl = QLabel()
        self._frame_counter_lbl.setStyleSheet('color: #64748b; font-size: 12px;')
        info_row.addWidget(self._frame_counter_lbl)

        self._ts_lbl = QLabel()
        self._ts_lbl.setStyleSheet('color: #64748b; font-size: 12px;')
        info_row.addWidget(self._ts_lbl)
        info_row.addStretch()
        cc.addLayout(info_row)

        col.addWidget(ctrl_card)
        return panel

    # ── Right panel: stats + line chart + bar chart ───────────────────────────
    def _build_right_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet('QScrollArea { background: transparent; border: none; }')

        inner = QWidget()
        inner.setStyleSheet('background: transparent;')
        col = QVBoxLayout(inner)
        col.setSpacing(12)
        col.setContentsMargins(8, 0, 0, 0)

        col.addWidget(self._build_stat_cards())
        col.addWidget(self._build_chart_tabs())
        col.addStretch()

        scroll.setWidget(inner)
        return scroll

    # ── Stats ─────────────────────────────────────────────────────────────────
    def _build_stat_cards(self) -> QFrame:
        card = QFrame()
        card.setObjectName('content_card')
        card.setStyleSheet(CONTENT_CARD_STYLE)
        col = QVBoxLayout(card)
        col.setContentsMargins(14, 12, 14, 12)
        col.setSpacing(10)

        self._stat_title_lbl = QLabel()
        self._stat_title_lbl.setFont(QFont('Microsoft JhengHei', 13, QFont.Weight.Bold))
        self._stat_title_lbl.setStyleSheet('color: #0f172a;')
        col.addWidget(self._stat_title_lbl)

        stats = self._results.get('stats', {})
        fps   = self._results.get('fps', 1)
        total = self._results.get('total_frames', 0)
        dur   = total / fps if fps else 0
        analysis_dur = self._results.get('analysis_duration_seconds')
        valid = sum(1 for r in self._records if isinstance(r.get('best_score'), int))
        invalid = len(self._records) - valid
        anom_frames = sum(
            1 for r in self._records
            if r.get('joint_anomaly') and not all(r['joint_anomaly'])
        )
        anom_text = str(anom_frames) if anom_frames > 0 else '—'
        analysis_dur_text = '—'
        if isinstance(analysis_dur, (int, float)):
            analysis_dur_text = f'{analysis_dur:.1f} s'

        # Store value/label pairs so we can update label text on language change
        # Each entry: (value_str, key, text_color, bg_color)
        self._stat_items = [
            (str(total),                          'result_stat_total',    '#1e40af', '#dbeafe'),
            (str(valid),                           'result_stat_valid',    '#065f46', '#d1fae5'),
            (str(invalid),                         'result_stat_invalid',  '#991b1b', '#fee2e2'),
            (f'{dur:.1f} s',                       'result_stat_duration', '#6d28d9', '#ede9fe'),
            (analysis_dur_text,                   'result_stat_analysis_duration', '#0f766e', '#ccfbf1'),
            (str(stats.get('max_score') or '—'),   'result_stat_max',      '#991b1b', '#fee2e2'),
            (f"{stats.get('avg_score') or '—'}",   'result_stat_avg',      '#92400e', '#fef3c7'),
            (anom_text,                            'result_stat_anom',     '#7c3aed', '#ede9fe'),
        ]

        self._stat_label_widgets = []  # keep refs to label QLabels for retranslation
        stat_row = QHBoxLayout()
        stat_row.setSpacing(6)
        for value, _key, tc, bg in self._stat_items:
            cell = QFrame()
            cell.setStyleSheet(f'QFrame {{ background:{bg}; border-radius:8px; }}'
                               'QLabel { background:transparent; }')
            cell.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(8, 6, 8, 6)
            cl.setSpacing(2)
            vl = QLabel(value)
            vl.setFont(QFont('Microsoft JhengHei', 13, QFont.Weight.Bold))
            vl.setStyleSheet(f'color:{tc};')
            vl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ll = QLabel()
            ll.setStyleSheet(f'color:{tc}; font-size:10px;')
            ll.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cl.addWidget(vl)
            cl.addWidget(ll)
            stat_row.addWidget(cell)
            self._stat_label_widgets.append(ll)

        col.addLayout(stat_row)
        return card

    # ── Chart tabs (Trend / Bar / Pie) ───────────────────────────────────────
    def _build_chart_tabs(self) -> QFrame:
        card = QFrame()
        card.setObjectName('content_card')
        card.setStyleSheet(CONTENT_CARD_STYLE)
        col = QVBoxLayout(card)
        col.setContentsMargins(14, 12, 14, 12)
        col.setSpacing(8)

        self._chart_tabs = QTabWidget()
        self._chart_tabs.setStyleSheet("""
            QTabWidget::pane { border: none; background: transparent; }
            QTabBar::tab {
                padding: 6px 18px; font-size: 12px; color: #64748b;
                background: #f1f5f9; border-radius: 6px; margin-right: 4px;
            }
            QTabBar::tab:selected { background: #2563eb; color: #ffffff; font-weight: bold; }
            QTabBar::tab:hover    { background: #e2e8f0; }
        """)

        self._chart_tabs.addTab(self._build_line_tab(), '')
        self._chart_tabs.addTab(self._build_bar_tab(),  '')
        self._chart_tabs.addTab(self._build_pie_tab(),  '')

        col.addWidget(self._chart_tabs)
        return card

    def _build_line_tab(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet('background: transparent;')
        col = QVBoxLayout(w)
        col.setContentsMargins(0, 8, 0, 0)
        col.setSpacing(0)

        xs = [r['timestamp']  for r in self._records if isinstance(r.get('best_score'), int)]
        ys = [r['best_score'] for r in self._records if isinstance(r.get('best_score'), int)]

        self._line_fig, self._line_ax = plt.subplots(figsize=(5.5, 3.2))
        self._line_fig.patch.set_facecolor('#ffffff')
        self._line_ax.set_facecolor('#ffffff')

        if xs and ys:
            self._line_ax.plot(xs, ys, color='#2563eb', linewidth=1.8,
                               marker='o', markersize=5, alpha=0.85, picker=6)
            self._line_ax.fill_between(xs, ys, alpha=0.10, color='#2563eb')
            self._line_ax.axhspan(5, 8, alpha=0.06, color='#ef4444')
            self._line_ax.axhspan(3, 5, alpha=0.06, color='#f59e0b')
            self._line_ax.set_ylim(0.5, 7.5)
            self._line_ax.set_yticks(range(1, 8))

        self._vline = self._line_ax.axvline(x=0, color='#ef4444',
                                             linewidth=1.5, linestyle='--', alpha=0.7)
        self._line_ax.tick_params(colors='#64748b', labelsize=8)
        for spine in self._line_ax.spines.values():
            spine.set_visible(False)
        self._line_ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#e2e8f0')
        self._line_ax.set_axisbelow(True)
        # Keep extra margins so translated axis labels are not clipped.
        self._line_fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.18)

        self._line_canvas = FigureCanvas(self._line_fig)
        self._line_canvas.setFixedHeight(300)
        self._line_canvas.mpl_connect('button_press_event', self._on_chart_click)
        col.addWidget(self._line_canvas)
        return w

    def _build_bar_tab(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet('background: transparent;')
        col = QVBoxLayout(w)
        col.setContentsMargins(0, 8, 0, 0)
        col.setSpacing(0)

        dist   = self._results.get('stats', {}).get('score_distribution', {})
        labels = [str(i) for i in range(1, 8)]
        values = [dist.get(str(i), 0) for i in range(1, 8)]
        colors = ['#10b981', '#10b981', '#f59e0b', '#f59e0b',
                  '#ef4444', '#ef4444', '#7c2d12']

        self._bar_fig, self._bar_ax = plt.subplots(figsize=(5, 3.0))
        self._bar_fig.patch.set_facecolor('#ffffff')
        self._bar_ax.set_facecolor('#ffffff')
        self._bar_ax.bar(labels, values, color=colors, width=0.6, edgecolor='none')
        self._bar_ax.tick_params(colors='#64748b', labelsize=8)
        self._bar_ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        for spine in self._bar_ax.spines.values():
            spine.set_visible(False)
        self._bar_ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#e2e8f0')
        self._bar_ax.set_axisbelow(True)
        # Keep extra margins so translated axis labels are not clipped.
        self._bar_fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.18)

        self._bar_canvas = FigureCanvas(self._bar_fig)
        self._bar_canvas.setFixedHeight(300)
        col.addWidget(self._bar_canvas)
        return w

    def _build_pie_tab(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet('background: transparent;')
        col = QVBoxLayout(w)
        col.setContentsMargins(0, 8, 0, 0)
        col.setSpacing(0)

        dist   = self._results.get('stats', {}).get('score_distribution', {})
        pie_data = {i: dist.get(str(i), 0) for i in range(1, 8)}
        pie_data = {k: v for k, v in pie_data.items() if v > 0}

        self._pie_fig, self._pie_ax = plt.subplots(figsize=(5, 3.2))
        self._pie_fig.patch.set_facecolor('#ffffff')
        self._pie_ax.set_facecolor('#ffffff')

        if pie_data:
            _score_bar_colors = {
                1: '#10b981', 2: '#10b981', 3: '#f59e0b', 4: '#f59e0b',
                5: '#ef4444', 6: '#ef4444', 7: '#7c2d12',
            }
            pie_labels = [str(k) for k in pie_data]
            pie_values = list(pie_data.values())
            pie_colors = [_score_bar_colors[k] for k in pie_data]
            wedges, texts, autotexts = self._pie_ax.pie(
                pie_values, labels=pie_labels, colors=pie_colors,
                autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 9, 'color': '#374151'},
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
            )
            for at in autotexts:
                at.set_fontsize(8)
        else:
            self._pie_ax.text(0.5, 0.5, '—', ha='center', va='center',
                              fontsize=20, color='#94a3b8',
                              transform=self._pie_ax.transAxes)
        self._pie_fig.subplots_adjust(left=0.06, right=0.96, top=0.96, bottom=0.10)

        self._pie_canvas = FigureCanvas(self._pie_fig)
        self._pie_canvas.setFixedHeight(300)
        col.addWidget(self._pie_canvas)
        return w

    # ── Language ──────────────────────────────────────────────────────────────
    def on_language_changed(self, _lang_code):
        self._retranslate_ui()

    def _retranslate_ui(self):
        fname = self._results.get('original_filename', '')
        self.setWindowTitle(f'{fname} {t("result_window_title_suffix")}')

        self._header_title_lbl.setText(
            t('result_header_title_prefix') + fname
        )
        # sub is metadata, stays as-is; but update empty fallback
        if not self._header_sub_lbl.text():
            self._header_sub_lbl.setText(t('result_header_sub_local'))

        self._export_btn.setText(t('result_export_btn'))
        self._close_btn.setText(t('result_close_btn'))

        self._frame_lbl.setText(t('result_no_video_text'))
        self._skel_cb.setText(t('result_skel_checkbox'))
        self._ctrl_title_lbl.setText(t('result_ctrl_title'))

        self._prev_btn.setText(t('result_prev_btn'))
        self._next_btn.setText(t('result_next_btn'))
        self._metrics_btn.setText(t('result_metrics_btn'))
        # Play/pause button: set based on current state
        if self._play_timer.isActive():
            self._play_btn.setText(t('result_pause_btn'))
        else:
            self._play_btn.setText(t('result_play_btn'))

        # Stat section title
        self._stat_title_lbl.setText(t('result_stat_title'))
        for lbl, (_val, key, _tc, _bg) in zip(self._stat_label_widgets, self._stat_items):
            lbl.setText(t(key))

        # Chart tab labels
        self._chart_tabs.setTabText(0, t('result_tab_trend'))
        self._chart_tabs.setTabText(1, t('result_tab_bar'))
        self._chart_tabs.setTabText(2, t('result_tab_pie'))

        # Update matplotlib axis labels
        self._line_ax.set_xlabel(t('result_chart_x'), fontsize=9, color='#64748b')
        self._line_ax.set_ylabel(t('result_chart_y'), fontsize=9, color='#64748b')
        try:
            self._line_canvas.draw_idle()
        except Exception:
            pass

        self._bar_ax.set_xlabel(t('result_bar_x'), fontsize=9, color='#64748b')
        self._bar_ax.set_ylabel(t('result_bar_y'), fontsize=9, color='#64748b')
        try:
            self._bar_canvas.draw_idle()
        except Exception:
            pass

        # Refresh current frame info labels (without reloading the frame image)
        if self._records:
            rec   = self._records[self._current_idx]
            total = len(self._records)
            self._frame_counter_lbl.setText(
                t('result_frame_counter').format(self._current_idx + 1, total)
            )
            self._ts_lbl.setText(
                t('result_time_label').format(rec.get('timestamp', 0))
            )

    # ── Frame navigation ──────────────────────────────────────────────────────
    def _show_frame(self, idx: int):
        if not self._records:
            return
        idx = max(0, min(idx, len(self._records) - 1))
        self._current_idx = idx
        rec = self._records[idx]

        frame_rgb = None
        if self._cap and self._cap.isOpened():
            frame_rgb = _frame_rgb_from_video(self._cap, rec['frame'])

        if frame_rgb is not None:
            native = rec.get('native_draw_data')

            if self._show_skeleton and isinstance(native, dict):
                backend = str(native.get('backend', self._backend_mode)).upper()
                if backend == 'RTMW3D':
                    keypoints_2d_norm = native.get('keypoints_2d_norm') or []
                    scores = native.get('scores') or []
                    if keypoints_2d_norm and scores:
                        frame_rgb = _draw_rtmw_skeleton(
                            frame_rgb,
                            keypoints_2d_norm,
                            scores,
                            kpt_threshold=0.3,
                        )
                elif backend == 'MEDIAPIPE':
                    lms = native.get('landmarks_2d') or []
                    if lms:
                        frame_rgb = _draw_mediapipe_skeleton(frame_rgb, lms)

            # ── Joint anomaly overlay (MediaPipe only) ────────────────────
            joint_anomaly = rec.get('joint_anomaly')
            print(f'[ANOM DEBUG] idx={idx} backend={native.get("backend") if isinstance(native,dict) else None} '
                  f'joint_anomaly type={type(joint_anomaly)} len={len(joint_anomaly) if joint_anomaly else None} '
                  f'false_count={joint_anomaly.count(False) if isinstance(joint_anomaly,list) else None}')
            if (joint_anomaly and isinstance(native, dict)
                    and str(native.get('backend', '')).upper() == 'MEDIAPIPE'):
                lms_2d = native.get('landmarks_2d') or []
                h_fr, w_fr = frame_rgb.shape[:2]
                for i, reliable in enumerate(joint_anomaly):
                    if not reliable and i < len(lms_2d) and len(lms_2d[i]) >= 2:
                        cx = int(lms_2d[i][0] * w_fr)
                        cy = int(lms_2d[i][1] * h_fr)
                        # 橘紅色 X 標記（外框白色增加對比）
                        d = 8
                        cv2.line(frame_rgb, (cx-d, cy-d), (cx+d, cy+d), (255, 255, 255), 4)
                        cv2.line(frame_rgb, (cx+d, cy-d), (cx-d, cy+d), (255, 255, 255), 4)
                        cv2.line(frame_rgb, (cx-d, cy-d), (cx+d, cy+d), (255, 80, 0),   2)
                        cv2.line(frame_rgb, (cx+d, cy-d), (cx-d, cy+d), (255, 80, 0),   2)
            score = rec.get('best_score')
            txt   = f"RULA: {score if score is not None else 'NULL'}"
            cv2.putText(frame_rgb, txt, (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(frame_rgb, txt, (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0),       1)
            self._display_frame(frame_rgb)
            self._no_video_lbl.setText('')
        else:
            if self._cap is None:
                self._no_video_lbl.setText(t('result_no_video_note'))
            self._frame_lbl.setText(
                f'Frame #{rec.get("frame", idx)}'
                f'\n{t("result_time_label").format(rec.get("timestamp", 0))}'
                f'\nRULA: {rec.get("best_score") or "NULL"}'
            )

        ts    = rec.get('timestamp', 0)
        score = rec.get('best_score')
        total = len(self._records)
        self._frame_counter_lbl.setText(
            t('result_frame_counter').format(idx + 1, total)
        )
        self._ts_lbl.setText(t('result_time_label').format(ts))
        self._update_score_badge(score)

        # Sync slider without re-triggering _on_slider_changed
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(idx)
        self._frame_slider.blockSignals(False)

        self._vline.set_xdata([ts, ts])
        try:
            self._line_canvas.draw_idle()
        except Exception:
            pass

    def _display_frame(self, rgb: np.ndarray):
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self._frame_lbl.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._frame_lbl.setPixmap(pix)

    def _update_score_badge(self, score):
        if score is None:
            self._score_badge.setText('RULA: —')
            self._score_badge.setStyleSheet(
                'background:#e2e8f0; color:#475569; border-radius:10px;'
                'padding:0 12px; font-size:12px; font-weight:bold;'
            )
        else:
            bg, fg = _SCORE_COLORS.get(int(score), ('#e2e8f0', '#475569'))
            self._score_badge.setText(f'RULA: {score}')
            self._score_badge.setStyleSheet(
                f'background:{bg}; color:{fg}; border-radius:10px;'
                'padding:0 12px; font-size:12px; font-weight:bold;'
            )

    # ── Frame Metrics Dialog ──────────────────────────────────────────────────
    def _show_frame_metrics_dialog(self):
        if not self._records:
            return
        rec = self._records[self._current_idx]
        frame_label = (
            t('result_frame_counter').format(self._current_idx + 1, len(self._records))
            + '   '
            + t('result_time_label').format(rec.get('timestamp', 0))
        )
        dlg = FrameMetricsDialog(
            rec=rec,
            frame_label=frame_label,
            render_3d_fn=_render_3d_skeleton_pixmap,
            parent=self,
        )
        dlg.exec()

    # ── Controls ──────────────────────────────────────────────────────────────
    def _prev_frame(self):
        self._play_timer.stop()
        self._play_btn.setText(t('result_play_btn'))
        self._show_frame(self._current_idx - 1)

    def _next_frame(self):
        self._play_timer.stop()
        self._play_btn.setText(t('result_play_btn'))
        self._show_frame(self._current_idx + 1)

    def _toggle_play(self):
        if self._play_timer.isActive():
            self._play_timer.stop()
            self._play_btn.setText(t('result_play_btn'))
        else:
            self._play_btn.setText(t('result_pause_btn'))
            self._play_timer.start()

    def _playback_tick(self):
        if self._current_idx >= len(self._records) - 1:
            self._play_timer.stop()
            self._play_btn.setText(t('result_play_btn'))
            return
        self._show_frame(self._current_idx + 1)

    def _on_slider_changed(self, value: int):
        self._play_timer.stop()
        self._play_btn.setText(t('result_play_btn'))
        self._show_frame(value)

    def _on_skeleton_toggle(self, checked: bool):
        self._show_skeleton = checked
        self._show_frame(self._current_idx)

    # ── Chart click → jump to frame ───────────────────────────────────────────
    def _on_chart_click(self, event):
        if event.inaxes is not self._line_ax:
            return
        click_x = event.xdata
        if click_x is None:
            return

        best_dist = float('inf')
        best_idx  = self._current_idx
        for i, rec in enumerate(self._records):
            ts = rec.get('timestamp', 0)
            d  = abs(ts - click_x)
            if d < best_dist:
                best_dist = d
                best_idx  = i

        self._play_timer.stop()
        self._play_btn.setText(t('result_play_btn'))
        self._show_frame(best_idx)

    # ── Export ────────────────────────────────────────────────────────────────
    def _export_csv(self):
        meta = self._results.get('meta', {})
        default = (f"rula_{meta.get('task_name','analysis')}"
                   f"_{meta.get('survey_date','')}.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, t('result_export_dialog_title'), default,
            'CSV Files (*.csv);;All Files (*)'
        )
        if path:
            try:
                export_csv(self._results, path)
                QMessageBox.information(
                    self, t('result_export_success_title'),
                    t('result_export_success_msg').format(path)
                )
            except Exception as e:
                QMessageBox.critical(self, t('result_export_fail_title'), str(e))

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self._play_timer.stop()
        self._close_video()
        plt.close(self._line_fig)
        plt.close(self._bar_fig)
        plt.close(self._pie_fig)
        self.back_requested.emit()
        super().closeEvent(event)
