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
from ..core.config import RTMW_TO_MEDIAPIPE, BONE_NAME_TO_RTMW_PAIR, BONE_NAME_TO_MP_PAIR, MP_RULA_KEYPOINTS
from ..core.pose_detector import draw_rula_skeleton, draw_rula_skeleton_mp
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


# ── RULA-only 3D skeleton connections (MediaPipe 33-pt indices) ──────────────
# 3D 分析模式僅使用 MediaPipe，故只保留 MediaPipe 的 RULA 連線定義。
# 僅含參與 RULA 角度計算的關節。
_MP_RULA_3D_CONNECTIONS = [
    ( 7,  8, '#0891b2'),  # 雙耳
    (11, 12, '#b45309'), (11, 23, '#b45309'), (12, 24, '#b45309'), (23, 24, '#b45309'),
    (11, 13, '#16a34a'), (13, 15, '#16a34a'),  # 左臂
    (12, 14, '#16a34a'), (14, 16, '#16a34a'),  # 右臂
    (15, 17, '#16a34a'), (15, 19, '#16a34a'),  # 左手
    (16, 18, '#16a34a'), (16, 20, '#16a34a'),  # 右手
    (11,  7, '#0891b2'), (12,  8, '#0891b2'),  # 肩→耳
]


def _render_3d_skeleton_pixmap(landmarks_3d: list,
                               width: int = 340,
                               height: int = 380,
                               native: dict = None,
                               rec: dict = None) -> 'QPixmap | None':
    """Render RULA-only 3D skeleton with anomaly markers into a QPixmap."""
    from io import BytesIO

    if str((native or {}).get('analysis_mode', '')).upper() == '2D':
        return None

    # 3D 分析模式僅使用 MediaPipe，故只處理 33 點 world landmarks。
    if not landmarks_3d or len(landmarks_3d) < 33:
        return None
    arr = np.asarray(landmarks_3d, dtype=np.float64)
    xs, ys, zs, vs = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    connections  = _MP_RULA_3D_CONNECTIONS
    rula_joints  = set(MP_RULA_KEYPOINTS)

    def _tx(p3):
        """world 座標 [x,y,z] → 繪圖座標 (px, py, pz) = (x, z, -y)。"""
        return p3[0], p3[2], -p3[1]

    # ── Bone anomaly lookup ────────────────────────────────────────────────────
    bone_anomaly_pairs: set = set()
    if rec:
        for bone_name in (rec.get('bone_anomaly') or {}):
            pair = BONE_NAME_TO_MP_PAIR.get(bone_name)
            if pair and len(pair) == 2:
                bone_anomaly_pairs.add((pair[0], pair[1]))
                bone_anomaly_pairs.add((pair[1], pair[0]))

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor='white')
    ax  = fig.add_subplot(111, projection='3d', facecolor='white')
    fig.subplots_adjust(left=0.0, right=0.82, top=0.99, bottom=0.08)

    # ── Draw RULA bones ────────────────────────────────────────────────────────
    for i, j, color in connections:
        if i >= len(vs) or j >= len(vs) or vs[i] <= 0.3 or vs[j] <= 0.3:
            continue
        c  = '#dc2626' if ((i, j) in bone_anomaly_pairs) else color
        lw = 3.0      if c == '#dc2626'                  else 2.2
        ax.plot([xs[i], xs[j]], [zs[i], zs[j]], [-ys[i], -ys[j]],
                color=c, linewidth=lw, alpha=0.9)

    # ── Draw RULA joints (confidence-coloured) ─────────────────────────────────
    cmap = plt.cm.plasma
    norm = plt.Normalize(0.0, 1.0)
    for k in rula_joints:
        if k < len(vs) and vs[k] > 0.3:
            ax.scatter(xs[k], zs[k], -ys[k],
                       color=cmap(norm(vs[k])), s=18, zorder=5, depthshade=False,
                       edgecolors='#334155', linewidths=0.4)

    # ── Anomaly markers ────────────────────────────────────────────────────────
    if rec:
        joint_anomaly = rec.get('joint_anomaly') or []
        detail_list   = rec.get('joint_anomaly_detail') or []
        interp        = rec.get('interpolation') or {}

        for mp_idx in rula_joints:
            if mp_idx >= len(xs):
                continue
            detail  = detail_list[mp_idx] if mp_idx < len(detail_list) else None
            reliable = joint_anomaly[mp_idx] if mp_idx < len(joint_anomaly) else True

            reasons: set = set()
            if isinstance(detail, dict):
                rl = detail.get('reasons') or []
                reasons = set(rl) if rl else ({detail['reason']} if detail.get('reason') else set())

            has_interp = str(mp_idx) in interp
            if reliable and not reasons and not has_interp:
                continue

            px, py, pz = float(xs[mp_idx]), float(zs[mp_idx]), float(-ys[mp_idx])

            if 'low_visibility' in reasons:
                ax.scatter(px, py, pz, marker='x', c='#ff5000', s=150,
                           linewidths=2.5, zorder=7, depthshade=False)
            if 'bone_length_abnormal' in reasons:
                ax.scatter(px, py, pz, marker='s', c='#bef264', s=90,
                           zorder=7, depthshade=False, alpha=0.9)
            if 'speed_candidate' in reasons or 'speed_jump' in reasons:
                ax.scatter(px, py, pz, marker='^', c='#dc2626', s=90,
                           zorder=7, depthshade=False, alpha=0.9)
            if has_interp:
                # 藍圈畫在「插值後」3D 座標；無 pos_3d 時退回原始位置
                info = interp.get(str(mp_idx))
                p3 = info.get('pos_3d') if isinstance(info, dict) else None
                if p3 and len(p3) >= 3:
                    bx, by, bz = _tx(p3)
                else:
                    bx, by, bz = px, py, pz
                ax.scatter(bx, by, bz, marker='o', c='#2563eb', s=110,
                           zorder=8, depthshade=False, alpha=0.85,
                           edgecolors='white', linewidths=1.2)

    # ── Colorbar ───────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.05, fraction=0.03, shrink=0.55)
    cbar.set_label('Confidence', color='#334155', fontsize=7)
    cbar.ax.yaxis.set_tick_params(color='#334155')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#334155', fontsize=6)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(['Low', '0.5', 'High'])

    # ── Axes styling ───────────────────────────────────────────────────────────
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.label.set_color('#334155')
        axis.set_tick_params(colors='#64748b', labelsize=5)
        axis.pane.fill = False
        axis.pane.set_edgecolor('#cbd5e1')
        axis._axinfo['grid']['color'] = '#e2e8f0'
    ax.set_xlabel('X', fontsize=6, color='#334155', labelpad=2)
    ax.set_ylabel('Z', fontsize=6, color='#334155', labelpad=2)
    ax.set_zlabel('Y', fontsize=6, color='#334155', labelpad=2)

    # ── 固定對稱立方體：以所有可見 RULA 關節質心為中心，box_aspect=1:1:1 ────────
    # 用 RULA 關節質心（而非臀部）作為中心，讓骨架自然填滿框內。
    # _CUBE_HALF 單位：公尺；手臂舉高被裁切時調大到 0.75。
    _CUBE_HALF = 0.65
    # 繪圖座標：(x, y, z)_plot = (xs, zs, -ys)
    px_plot, py_plot, pz_plot = xs, zs, -ys
    rula_vis = [k for k in rula_joints if k < len(vs) and vs[k] > 0.3]
    if rula_vis:
        cx = float(np.mean(px_plot[rula_vis]))
        cy = float(np.mean(py_plot[rula_vis]))
        cz = float(np.mean(pz_plot[rula_vis]))
    else:
        cx = cy = cz = 0.0
    ax.set_xlim(cx - _CUBE_HALF, cx + _CUBE_HALF)
    ax.set_ylim(cy - _CUBE_HALF, cy + _CUBE_HALF)
    ax.set_zlim(cz - _CUBE_HALF, cz + _CUBE_HALF)
    ax.set_box_aspect([1, 1, 1])
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
    """Draw MediaPipe RULA skeleton (only joints used in angle calculation)."""
    return draw_rula_skeleton_mp(frame_rgb.copy(), landmarks_2d)


def _draw_rtmw_skeleton(frame_rgb: np.ndarray,
                        keypoints_2d_norm: list,
                        scores: list,
                        kpt_threshold: float = 0.3) -> np.ndarray:
    """
    只繪製 RULA 實際分析/使用的 RTMW 點與連線（與即時預覽共用 draw_rula_skeleton），
    而非 rtmlib 的全身 133 點，使畫面與異常判定範圍一致。
    """
    if not keypoints_2d_norm or not scores:
        return frame_rgb

    h, w = frame_rgb.shape[:2]
    kpts_px = []
    for pt in keypoints_2d_norm:
        if len(pt) < 2:
            kpts_px.append([0.0, 0.0])
            continue
        kpts_px.append([float(pt[0]) * w, float(pt[1]) * h])

    return draw_rula_skeleton(
        frame_rgb,
        np.asarray(kpts_px, dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
        kpt_thr=kpt_threshold,
    )


def _apply_interp_to_native(native: dict, interp: dict) -> dict:
    """
    回傳一份套用補點位置的 native 複本（不改動原始記錄）。

    將每個補點關節的顯示座標覆蓋為插值後的 pos_norm，並把該點的可信度/分數
    拉到 1.0，使骨架繪製器（以可見度 / score 為門檻）會畫出修復後的點。
    """
    if not interp or not isinstance(native, dict):
        return native
    backend = str(native.get('backend', '')).upper()
    patched = dict(native)

    if backend in ('RTMW2D', 'RTMW3D'):
        pts = [list(p) for p in (native.get('keypoints_2d_norm') or [])]
        sc  = list(native.get('scores') or [])
        for j_str, info in interp.items():
            pos = info.get('pos_norm') if isinstance(info, dict) else None
            if not pos:
                continue
            rtmw_idx = RTMW_TO_MEDIAPIPE.get(int(j_str))
            if rtmw_idx is None:
                continue
            if rtmw_idx < len(pts) and len(pts[rtmw_idx]) >= 2:
                pts[rtmw_idx][0], pts[rtmw_idx][1] = pos[0], pos[1]
            if rtmw_idx < len(sc):
                sc[rtmw_idx] = 1.0
        patched['keypoints_2d_norm'] = pts
        patched['scores'] = sc
    else:
        lms = [list(p) for p in (native.get('landmarks_2d') or [])]
        for j_str, info in interp.items():
            pos = info.get('pos_norm') if isinstance(info, dict) else None
            if not pos:
                continue
            j = int(j_str)
            if j < len(lms) and len(lms[j]) >= 2:
                lms[j][0], lms[j][1] = pos[0], pos[1]
                if len(lms[j]) >= 3:
                    lms[j][2] = 1.0
        patched['landmarks_2d'] = lms
    return patched


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
        self._show_interp   = True
        self._has_interp    = bool(results.get('use_interpolation'))
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

        # 「顯示補點」只在有補點資料時出現；切換藍色圓圈顯示
        self._interp_cb = QCheckBox()
        self._interp_cb.setChecked(True)
        self._interp_cb.setStyleSheet('color: #2563eb; font-size: 13px;')
        self._interp_cb.toggled.connect(self._on_interp_toggle)
        self._interp_cb.setVisible(self._has_interp)
        skel_row.addWidget(self._interp_cb)

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
        speed_anomaly_enabled = self._results.get('speed_anomaly_enabled', True)
        speed_anomaly_text = t('common_on') if speed_anomaly_enabled else t('common_off')

        # Store value/label pairs so we can update label text on language change
        # Each entry: (value_str, key, text_color, bg_color)
        self._stat_items = [
            (str(total),                          'result_stat_total',    '#1e40af', '#dbeafe'),
            (str(valid),                           'result_stat_valid',    '#065f46', '#d1fae5'),
            (str(invalid),                         'result_stat_invalid',  '#991b1b', '#fee2e2'),
            (f'{dur:.1f} s',                       'result_stat_duration', '#6d28d9', '#ede9fe'),
            (analysis_dur_text,                   'result_stat_analysis_duration', '#0f766e', '#ccfbf1'),
            (speed_anomaly_text,                   'result_stat_speed_anomaly', '#0f766e', '#ccfbf1'),
            (str(stats.get('max_score') or '—'),   'result_stat_max',      '#991b1b', '#fee2e2'),
            (f"{stats.get('avg_score') or '—'}",   'result_stat_avg',      '#92400e', '#fef3c7'),
            (anom_text,                            'result_stat_anom',     '#7c3aed', '#ede9fe'),
        ]

        # 補點相關統計（僅在開啟補點時加入）
        interp_summary = self._results.get('interpolation_summary') or {}
        comparison     = self._results.get('rula_comparison') or {}
        if self._results.get('use_interpolation'):
            interp_total = interp_summary.get('total_interpolated', 0)
            rate_pct = round(interp_summary.get('rate', 0.0) * 100, 1)
            orig_avg   = comparison.get('original_avg')
            interp_avg = comparison.get('interpolated_avg')
            delta_avg  = comparison.get('delta_avg')
            delta_text = ('—' if delta_avg is None
                          else (f'+{delta_avg}' if delta_avg >= 0 else f'{delta_avg}'))
            self._stat_items += [
                (str(interp_total),                'result_stat_interp_count', '#1d4ed8', '#dbeafe'),
                (f'{rate_pct}%',                   'result_stat_interp_rate',  '#1d4ed8', '#dbeafe'),
                (f"{orig_avg if orig_avg is not None else '—'}",
                                                   'result_stat_orig_avg',     '#92400e', '#fef3c7'),
                (f"{interp_avg if interp_avg is not None else '—'}",
                                                   'result_stat_interp_avg',   '#1d4ed8', '#dbeafe'),
                (delta_text,                       'result_stat_interp_delta', '#7c3aed', '#ede9fe'),
            ]

        self._stat_label_widgets = []  # keep refs to label QLabels for retranslation
        self._speed_anomaly_value_lbl = None
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
            if _key == 'result_stat_speed_anomaly':
                self._speed_anomaly_value_lbl = vl
            # 補點總數格：tooltip 顯示各關節明細（Interpolation Summary）
            if _key == 'result_stat_interp_count':
                per_joint = (self._results.get('interpolation_summary') or {}).get('per_joint') or {}
                if per_joint:
                    lines = [f'{name}: {cnt}' for name, cnt in per_joint.items()]
                    tip = t('result_interp_summary_title') + '\n' + '\n'.join(lines)
                    cell.setToolTip(tip)
                    vl.setToolTip(tip)

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

        use_interp = bool(self._results.get('use_interpolation'))

        if xs and ys:
            self._line_ax.axhspan(5, 8, alpha=0.06, color='#ef4444')
            self._line_ax.axhspan(3, 5, alpha=0.06, color='#f59e0b')

            if use_interp:
                # 原始 RULA（灰色虛線，畫在底層供對照）
                ox = [r['timestamp'] for r in self._records
                      if isinstance(r.get('original_best_score'), int)]
                oy = [r['original_best_score'] for r in self._records
                      if isinstance(r.get('original_best_score'), int)]
                if ox and oy:
                    self._line_ax.plot(ox, oy, color='#94a3b8', linewidth=1.4,
                                       linestyle='--', marker='o', markersize=3,
                                       alpha=0.8, label=t('result_chart_legend_orig'),
                                       zorder=2)
                interp_label = t('result_chart_legend_interp')
            else:
                interp_label = None

            # primary（補點後 / 原始）藍色實線
            self._line_ax.plot(xs, ys, color='#2563eb', linewidth=1.8,
                               marker='o', markersize=5, alpha=0.85, picker=6,
                               label=interp_label, zorder=3)
            self._line_ax.fill_between(xs, ys, alpha=0.10, color='#2563eb', zorder=1)
            self._line_ax.set_ylim(0.5, 7.5)
            self._line_ax.set_yticks(range(1, 8))

            if use_interp:
                self._line_ax.legend(loc='upper right', fontsize=8, framealpha=0.85,
                                     ncol=2, handlelength=1.6)

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
        self._interp_cb.setText(t('result_interp_checkbox'))
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
        if self._speed_anomaly_value_lbl is not None:
            speed_anomaly_enabled = self._results.get('speed_anomaly_enabled', True)
            self._speed_anomaly_value_lbl.setText(
                t('common_on') if speed_anomaly_enabled else t('common_off')
            )

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
            interp = rec.get('interpolation') or {}

            if self._show_skeleton and isinstance(native, dict):
                backend = str(native.get('backend', self._backend_mode)).upper()
                if backend in ('RTMW2D', 'RTMW3D'):
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

            # ── Bone anomaly overlay (紅色骨段線，疊在綠色骨架之上) ──────
            joint_anomaly = rec.get('joint_anomaly')
            joint_anomaly_detail = rec.get('joint_anomaly_detail') or []
            backend_name = str(native.get('backend', '')).upper() if isinstance(native, dict) else ''

            bone_anomaly = rec.get('bone_anomaly') or {}
            if bone_anomaly and isinstance(native, dict) and backend_name in ('MEDIAPIPE', 'RTMW2D', 'RTMW3D'):
                h_fr, w_fr = frame_rgb.shape[:2]
                if backend_name in ('RTMW2D', 'RTMW3D'):
                    lms_bone = native.get('keypoints_2d_norm') or []
                    bone_pair_map = BONE_NAME_TO_RTMW_PAIR
                else:
                    lms_bone = native.get('landmarks_2d') or []
                    bone_pair_map = BONE_NAME_TO_MP_PAIR
                for bone_name in bone_anomaly:
                    pair = bone_pair_map.get(bone_name)
                    if not pair:
                        continue
                    a, b = pair
                    if a >= len(lms_bone) or b >= len(lms_bone):
                        continue
                    if len(lms_bone[a]) < 2 or len(lms_bone[b]) < 2:
                        continue
                    px_a = (int(lms_bone[a][0] * w_fr), int(lms_bone[a][1] * h_fr))
                    px_b = (int(lms_bone[b][0] * w_fr), int(lms_bone[b][1] * h_fr))
                    cv2.line(frame_rgb, px_a, px_b, (255, 255, 255), 5)  # 白色描邊
                    cv2.line(frame_rgb, px_a, px_b, (220,   0,   0), 3)  # 紅色骨段

            # ── Joint anomaly overlay (X / □ / △ 標記) ──────────────────
            # 遍歷 detail（含可靠但 speed_candidate 的關節），逐關節依 reasons 疊加形狀：
            #   low_visibility       → 橘色 X（invalid）
            #   bone_length_abnormal → 黃色 □（invalid，僅 3D；core bone 不會出現）
            #   speed_candidate      → 紅色 △（非 invalid，純速度提示）
            if isinstance(native, dict) and backend_name in ('MEDIAPIPE', 'RTMW2D', 'RTMW3D') \
                    and (joint_anomaly or joint_anomaly_detail):
                h_fr, w_fr = frame_rgb.shape[:2]
                if backend_name in ('RTMW2D', 'RTMW3D'):
                    lms_2d = native.get('keypoints_2d_norm') or []
                    idx_mapper = RTMW_TO_MEDIAPIPE
                else:
                    lms_2d = native.get('landmarks_2d') or []
                    idx_mapper = {i: i for i in range(len(lms_2d))}

                n_joints = max(len(joint_anomaly or []), len(joint_anomaly_detail))
                _mp_rula_set = set(MP_RULA_KEYPOINTS)
                for i in range(n_joints):
                    # MediaPipe：只標記有參與角度計算的關節
                    if backend_name == 'MEDIAPIPE' and i not in _mp_rula_set:
                        continue
                    detail = joint_anomaly_detail[i] if i < len(joint_anomaly_detail) else None
                    reliable = joint_anomaly[i] if (joint_anomaly and i < len(joint_anomaly)) else True

                    # reasons 清單（優先）；向後相容僅有 reason 欄位的舊記錄
                    reasons_set: set[str] = set()
                    if isinstance(detail, dict):
                        reasons_list = detail.get('reasons') or []
                        if reasons_list:
                            reasons_set = set(reasons_list)
                        elif detail.get('reason'):
                            reasons_set = {detail['reason']}

                    # 沒有任何標記要畫就略過（可靠且無 reasons）
                    if reliable and not reasons_set:
                        continue

                    src_idx = idx_mapper.get(i)
                    if src_idx is None or src_idx >= len(lms_2d) or len(lms_2d[src_idx]) < 2:
                        continue
                    cx = int(lms_2d[src_idx][0] * w_fr)
                    cy = int(lms_2d[src_idx][1] * h_fr)

                    d = 10
                    if 'low_visibility' in reasons_set:
                        cv2.line(frame_rgb, (cx-d, cy-d), (cx+d, cy+d), (255,255,255), 5)
                        cv2.line(frame_rgb, (cx+d, cy-d), (cx-d, cy+d), (255,255,255), 5)
                        cv2.line(frame_rgb, (cx-d, cy-d), (cx+d, cy+d), (255, 80,  0), 3)
                        cv2.line(frame_rgb, (cx+d, cy-d), (cx-d, cy+d), (255, 80,  0), 3)
                    if 'bone_length_abnormal' in reasons_set:
                        cv2.rectangle(frame_rgb, (cx-d, cy-d), (cx+d, cy+d), (255,255,255), 4)
                        cv2.rectangle(frame_rgb, (cx-d, cy-d), (cx+d, cy+d), (220,200,  0), 2)
                    # 三角形 = 速度提示（speed_candidate）；相容舊記錄的 speed_jump
                    if 'speed_candidate' in reasons_set or 'speed_jump' in reasons_set:
                        p1 = (cx,    cy - d)
                        p2 = (cx - d, cy + d)
                        p3 = (cx + d, cy + d)
                        cv2.polylines(frame_rgb, [np.array([p1, p2, p3])], True, (255,255,255), 4)
                        cv2.polylines(frame_rgb, [np.array([p1, p2, p3])], True, (220,  0,  0), 2)
                    # 不可靠但 reasons 為空（舊格式無 reasons 欄位）fallback 橘色 X
                    if not reasons_set and not reliable:
                        cv2.line(frame_rgb, (cx-d, cy-d), (cx+d, cy+d), (255,255,255), 5)
                        cv2.line(frame_rgb, (cx+d, cy-d), (cx-d, cy+d), (255,255,255), 5)
                        cv2.line(frame_rgb, (cx-d, cy-d), (cx+d, cy+d), (255, 80,  0), 3)
                        cv2.line(frame_rgb, (cx+d, cy-d), (cx-d, cy+d), (255, 80,  0), 3)

            # ── Interpolated joints (藍色圓圈) ──────────────────────────
            if self._show_interp and interp:
                h_fr, w_fr = frame_rgb.shape[:2]
                for info in interp.values():
                    pos = info.get('pos_norm') if isinstance(info, dict) else None
                    if not pos or len(pos) < 2:
                        continue
                    cx = int(pos[0] * w_fr)
                    cy = int(pos[1] * h_fr)
                    cv2.circle(frame_rgb, (cx, cy), 9, (255, 255, 255), 3)  # 白色描邊
                    cv2.circle(frame_rgb, (cx, cy), 9, ( 40, 120, 255), 2)  # 藍色圓圈

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
            group_thresholds=self._results.get('joint_group_thresholds'),
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

    def _on_interp_toggle(self, checked: bool):
        self._show_interp = checked
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
