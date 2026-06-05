"""
影片上傳分析視窗模組。

提供離線分析入口，負責蒐集調查資訊與分析參數，啟動背景執行緒處理影片，
並在完成後發送 analysis_done 給控制器開啟結果視窗。
"""

import os

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFrame, QLineEdit, QSpinBox, QComboBox,
    QProgressBar, QFileDialog, QGroupBox, QSizePolicy, QCheckBox,
    QDateEdit, QMessageBox, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDate
from PyQt6.QtGui import QPixmap, QImage, QFont

from ..core.video_file_processor import VideoFileProcessor, save_analysis
from .styles import (
    UPLOAD_BG_STYLE, CONTENT_CARD_STYLE, HEADER_CARD_STYLE,
    BACK_BTN_STYLE, EMERALD_BTN_STYLE, BLUE_BTN_STYLE, RED_BTN_STYLE,
)
from .language import language_manager, t


class UploadWindow(QMainWindow):
    """
    上傳影片 → 送出分析

    Signals:
        back_requested:     使用者按「回到主頁」
        analysis_done(dict): 分析完成，附帶完整 results dict（含 native_draw_data）
    """

    back_requested  = pyqtSignal()
    analysis_done   = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        language_manager.add_observer(self.on_language_changed)
        self.setMinimumSize(820, 620)
        self.resize(940, 700)
        self.setStyleSheet(UPLOAD_BG_STYLE)

        self._processor   = None
        self._proc_thread = None
        self._video_path  = ''

        self._init_ui()
        self._retranslate_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        outer = QVBoxLayout(central)
        outer.setSpacing(20)
        outer.setContentsMargins(40, 28, 40, 28)

        outer.addWidget(self._build_header())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet('QScrollArea { background: transparent; border: none; }')

        content = QWidget()
        content.setStyleSheet('background: transparent;')
        self._content_layout = QVBoxLayout(content)
        self._content_layout.setSpacing(16)
        self._content_layout.setContentsMargins(0, 0, 0, 0)

        scroll.setWidget(content)
        outer.addWidget(scroll)

        self._build_form_panel()

    # ── Header ────────────────────────────────────────────────────────────────
    def _build_header(self) -> QFrame:
        card = QFrame()
        card.setObjectName('header_card')
        card.setStyleSheet(HEADER_CARD_STYLE)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        row = QHBoxLayout(card)
        row.setContentsMargins(24, 16, 24, 16)
        row.setSpacing(14)

        icon = QLabel('📂')
        icon.setFixedSize(48, 48)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #3b82f6, stop:1 #2563eb);
                border-radius: 12px; font-size: 22px;
            }
        """)
        row.addWidget(icon)

        tc = QVBoxLayout()
        self._header_title = QLabel()
        self._header_title.setFont(QFont('Microsoft JhengHei', 16, QFont.Weight.Bold))
        self._header_title.setStyleSheet('color: #0f172a;')
        tc.addWidget(self._header_title)

        self._header_sub = QLabel()
        self._header_sub.setStyleSheet('color: #64748b; font-size: 13px;')
        tc.addWidget(self._header_sub)

        row.addLayout(tc)
        row.addStretch()

        self._back_btn = QPushButton()
        self._back_btn.setStyleSheet(BACK_BTN_STYLE)
        self._back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._back_btn.clicked.connect(self.back_requested.emit)
        row.addWidget(self._back_btn)

        return card

    # ── Form ──────────────────────────────────────────────────────────────────
    def _build_form_panel(self):
        card = QFrame()
        card.setObjectName('content_card')
        card.setStyleSheet(CONTENT_CARD_STYLE)

        col = QVBoxLayout(card)
        col.setContentsMargins(28, 24, 28, 24)
        col.setSpacing(18)

        # File selection
        self._file_group = QGroupBox()
        fl = QHBoxLayout(self._file_group)
        fl.setSpacing(10)

        self._file_label = QLabel()
        self._file_label.setStyleSheet('color: #64748b; font-size: 13px;')
        self._file_label.setWordWrap(True)
        fl.addWidget(self._file_label, stretch=1)

        self._choose_btn = QPushButton()
        self._choose_btn.setStyleSheet(BLUE_BTN_STYLE)
        self._choose_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._choose_btn.setFixedWidth(120)
        self._choose_btn.clicked.connect(self._choose_file)
        fl.addWidget(self._choose_btn)

        col.addWidget(self._file_group)

        # Survey metadata
        self._meta_group = QGroupBox()
        grid = QGridLayout(self._meta_group)
        grid.setSpacing(10)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        self._date_label = QLabel()
        grid.addWidget(self._date_label, 0, 0)
        self._date_edit = QDateEdit(QDate.currentDate())
        self._date_edit.setCalendarPopup(True)
        self._date_edit.setDisplayFormat('yyyy-MM-dd')
        grid.addWidget(self._date_edit, 0, 1)

        self._assessor_label = QLabel()
        grid.addWidget(self._assessor_label, 0, 2)
        self._assessor_edit = QLineEdit()
        grid.addWidget(self._assessor_edit, 0, 3)

        self._org_label = QLabel()
        grid.addWidget(self._org_label, 1, 0)
        self._org_edit = QLineEdit()
        grid.addWidget(self._org_edit, 1, 1)

        self._task_label = QLabel()
        grid.addWidget(self._task_label, 1, 2)
        self._task_edit = QLineEdit()
        grid.addWidget(self._task_edit, 1, 3)

        col.addWidget(self._meta_group)

        # Analysis settings
        self._settings_group = QGroupBox()
        sl = QGridLayout(self._settings_group)
        sl.setSpacing(12)
        sl.setColumnStretch(1, 1)
        sl.setColumnStretch(3, 1)

        self._interval_label = QLabel()
        sl.addWidget(self._interval_label, 0, 0)
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(1, 60)
        self._interval_spin.setValue(10)
        sl.addWidget(self._interval_spin, 0, 1)

        self._backend_label = QLabel()
        sl.addWidget(self._backend_label, 0, 2)
        self._backend_combo = QComboBox()
        self._backend_combo.addItem('', 'RTMW3D')
        self._backend_combo.addItem('', 'MEDIAPIPE')
        sl.addWidget(self._backend_combo, 0, 3)

        self._wrist_twist_label = QLabel()
        sl.addWidget(self._wrist_twist_label, 1, 0)
        self._wrist_twist_combo = QComboBox()
        self._wrist_twist_combo.addItem('', 1)
        self._wrist_twist_combo.addItem('', 2)
        sl.addWidget(self._wrist_twist_combo, 1, 1)

        self._legs_label = QLabel()
        sl.addWidget(self._legs_label, 1, 2)
        self._legs_combo = QComboBox()
        self._legs_combo.addItem('', 1)
        self._legs_combo.addItem('', 2)
        self._legs_combo.setCurrentIndex(1)
        sl.addWidget(self._legs_combo, 1, 3)

        self._muscle_use_a_label = QLabel()
        sl.addWidget(self._muscle_use_a_label, 2, 0)
        self._muscle_use_a_combo = QComboBox()
        self._muscle_use_a_combo.addItem('', 0)
        self._muscle_use_a_combo.addItem('', 1)
        sl.addWidget(self._muscle_use_a_combo, 2, 1)

        self._muscle_use_b_label = QLabel()
        sl.addWidget(self._muscle_use_b_label, 2, 2)
        self._muscle_use_b_combo = QComboBox()
        self._muscle_use_b_combo.addItem('', 0)
        self._muscle_use_b_combo.addItem('', 1)
        sl.addWidget(self._muscle_use_b_combo, 2, 3)

        self._force_load_a_label = QLabel()
        sl.addWidget(self._force_load_a_label, 3, 0)
        self._force_load_a_combo = QComboBox()
        self._force_load_a_combo.addItem('', 0)
        self._force_load_a_combo.addItem('', 1)
        self._force_load_a_combo.addItem('', 2)
        self._force_load_a_combo.addItem('', 3)
        sl.addWidget(self._force_load_a_combo, 3, 1)

        self._force_load_b_label = QLabel()
        sl.addWidget(self._force_load_b_label, 3, 2)
        self._force_load_b_combo = QComboBox()
        self._force_load_b_combo.addItem('', 0)
        self._force_load_b_combo.addItem('', 1)
        self._force_load_b_combo.addItem('', 2)
        self._force_load_b_combo.addItem('', 3)
        sl.addWidget(self._force_load_b_combo, 3, 3)

        self._speed_anomaly_checkbox = QCheckBox()
        self._speed_anomaly_checkbox.setChecked(True)
        sl.addWidget(self._speed_anomaly_checkbox, 4, 0, 1, 4)

        col.addWidget(self._settings_group)

        # Analyze button
        self._analyze_btn = QPushButton()
        self._analyze_btn.setStyleSheet(EMERALD_BTN_STYLE)
        self._analyze_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._analyze_btn.clicked.connect(self._start_analysis)
        col.addWidget(self._analyze_btn)

        self._content_layout.addWidget(card)

        # Preview label
        self._preview_lbl = QLabel()
        self._preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_lbl.setVisible(False)
        self._preview_lbl.setStyleSheet(
            'border: 2px solid #3b82f6; border-radius:10px; background:#1a1a1a;'
        )
        self._preview_lbl.setFixedHeight(200)
        self._content_layout.addWidget(self._preview_lbl)

        # Progress card
        self._progress_card = self._build_progress_card()
        self._progress_card.setVisible(False)
        self._content_layout.addWidget(self._progress_card)

        self._content_layout.addStretch()

    def _build_progress_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName('content_card')
        card.setStyleSheet(CONTENT_CARD_STYLE)

        col = QVBoxLayout(card)
        col.setContentsMargins(24, 20, 24, 20)
        col.setSpacing(12)

        self._status_lbl = QLabel()
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet('color: #0f172a; font-size: 14px;')
        col.addWidget(self._status_lbl)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        col.addWidget(self._progress_bar)

        h = QHBoxLayout()
        h.addStretch()
        self._cancel_btn = QPushButton()
        self._cancel_btn.setStyleSheet(RED_BTN_STYLE)
        self._cancel_btn.setFixedWidth(100)
        self._cancel_btn.clicked.connect(self._cancel_analysis)
        h.addWidget(self._cancel_btn)
        col.addLayout(h)

        return card

    # ── Language ──────────────────────────────────────────────────────────────
    def on_language_changed(self, _lang_code):
        self._retranslate_ui()

    def _retranslate_ui(self):
        self.setWindowTitle(t('upload_window_title'))
        self._header_title.setText(t('upload_header_title'))
        self._header_sub.setText(t('upload_header_sub'))
        self._back_btn.setText(t('upload_back_btn'))

        self._file_group.setTitle(t('upload_file_group'))
        # Only reset file label text if no file chosen yet
        if not self._video_path:
            self._file_label.setText(t('upload_no_file'))
        self._choose_btn.setText(t('upload_choose_btn'))

        self._meta_group.setTitle(t('upload_meta_group'))
        self._date_label.setText(t('upload_date_label'))
        self._assessor_label.setText(t('upload_assessor_label'))
        self._assessor_edit.setPlaceholderText(t('upload_assessor_placeholder'))
        self._org_label.setText(t('upload_org_label'))
        self._org_edit.setPlaceholderText(t('upload_org_placeholder'))
        self._task_label.setText(t('upload_task_label'))
        self._task_edit.setPlaceholderText(t('upload_task_placeholder'))

        self._settings_group.setTitle(t('upload_settings_group'))
        self._interval_label.setText(t('upload_interval_label'))
        self._interval_spin.setToolTip(t('upload_interval_tooltip'))
        self._backend_label.setText(t('upload_backend_label'))
        self._backend_combo.setItemText(0, t('upload_backend_rtmw3d'))
        self._backend_combo.setItemText(1, t('upload_backend_mediapipe'))
        self._wrist_twist_label.setText(t('upload_wrist_twist_label'))
        self._wrist_twist_combo.setItemText(0, t('upload_wrist_twist_option_neutral'))
        self._wrist_twist_combo.setItemText(1, t('upload_wrist_twist_option_twisted'))

        self._legs_label.setText(t('upload_legs_label'))
        self._legs_combo.setItemText(0, t('upload_legs_option_supported'))
        self._legs_combo.setItemText(1, t('upload_legs_option_unstable'))

        self._muscle_use_a_label.setText(t('upload_muscle_use_a_label'))
        self._muscle_use_a_combo.setItemText(0, t('upload_muscle_use_option_no'))
        self._muscle_use_a_combo.setItemText(1, t('upload_muscle_use_option_yes'))

        self._muscle_use_b_label.setText(t('upload_muscle_use_b_label'))
        self._muscle_use_b_combo.setItemText(0, t('upload_muscle_use_option_no'))
        self._muscle_use_b_combo.setItemText(1, t('upload_muscle_use_option_yes'))

        self._force_load_a_label.setText(t('upload_force_load_a_label'))
        self._force_load_a_combo.setItemText(0, t('upload_load_option_0'))
        self._force_load_a_combo.setItemText(1, t('upload_load_option_1'))
        self._force_load_a_combo.setItemText(2, t('upload_load_option_2'))
        self._force_load_a_combo.setItemText(3, t('upload_load_option_3'))

        self._force_load_b_label.setText(t('upload_force_load_b_label'))
        self._force_load_b_combo.setItemText(0, t('upload_load_option_0'))
        self._force_load_b_combo.setItemText(1, t('upload_load_option_1'))
        self._force_load_b_combo.setItemText(2, t('upload_load_option_2'))
        self._force_load_b_combo.setItemText(3, t('upload_load_option_3'))

        self._speed_anomaly_checkbox.setText(t('upload_speed_anomaly_label'))
        self._speed_anomaly_checkbox.setToolTip(t('upload_speed_anomaly_tooltip'))

        self._analyze_btn.setText(t('upload_analyze_btn'))
        self._cancel_btn.setText(t('upload_cancel_btn'))
        self._status_lbl.setText(t('upload_status_ready'))

    # ── File chooser ──────────────────────────────────────────────────────────
    def _choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, t('upload_choose_dialog_title'), '',
            'Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All Files (*)'
        )
        if path:
            self._video_path = path
            self._file_label.setText(os.path.basename(path))
            self._file_label.setStyleSheet('color: #0f172a; font-size: 13px;')

    # ── Start / cancel analysis ───────────────────────────────────────────────
    def _start_analysis(self):
        if not self._video_path:
            QMessageBox.warning(self, t('upload_no_file_title'), t('upload_no_file_msg'))
            return

        # Reuse one processing slot at a time. If a previous analysis is still
        # around, stop it first to avoid destroying a running QThread.
        self._cleanup_processing_thread(wait_ms=5000)

        meta = {
            'survey_date':  self._date_edit.date().toString('yyyy-MM-dd'),
            'assessor':     self._assessor_edit.text().strip(),
            'organization': self._org_edit.text().strip(),
            'task_name':    self._task_edit.text().strip(),
        }

        rula_params = {
            'wrist_twist': int(self._wrist_twist_combo.currentData() or 1),
            'legs': int(self._legs_combo.currentData() or 1),
            'muscle_use_a': int(self._muscle_use_a_combo.currentData() or 0),
            'muscle_use_b': int(self._muscle_use_b_combo.currentData() or 0),
            'force_load_a': int(self._force_load_a_combo.currentData() or 0),
            'force_load_b': int(self._force_load_b_combo.currentData() or 0),
        }

        self._analyze_btn.setEnabled(False)
        self._preview_lbl.setVisible(True)
        self._progress_card.setVisible(True)
        self._progress_bar.setValue(0)
        self._status_lbl.setText(t('upload_status_init'))

        self._proc_thread = QThread()
        self._processor   = VideoFileProcessor(
            video_path     = self._video_path,
            meta           = meta,
            frame_interval = self._interval_spin.value(),
            backend_mode   = self._backend_combo.currentData(),
            rula_params    = rula_params,
            enable_speed_anomaly = self._speed_anomaly_checkbox.isChecked(),
        )
        self._processor.moveToThread(self._proc_thread)

        self._proc_thread.started.connect(self._processor.run)
        self._processor.progress_updated.connect(self._on_progress)
        self._processor.frame_preview.connect(self._on_preview)
        self._processor.analysis_complete.connect(self._on_complete)
        self._processor.error_occurred.connect(self._on_error)

        self._proc_thread.start()

    def _cancel_analysis(self):
        self._cleanup_processing_thread(wait_ms=3000)
        self._reset_form_state()

    def _cleanup_processing_thread(self, wait_ms: int = 3000):
        processor = self._processor
        thread = self._proc_thread

        if processor is not None:
            try:
                processor.cancel()
            except Exception:
                pass

        if thread is not None:
            try:
                if thread.isRunning():
                    thread.quit()
                    thread.wait(wait_ms)
            finally:
                try:
                    thread.deleteLater()
                except Exception:
                    pass

        self._processor = None
        self._proc_thread = None

    # ── Worker signals ────────────────────────────────────────────────────────
    def _on_progress(self, pct: int, msg: str):
        self._progress_bar.setValue(pct)
        self._status_lbl.setText(msg)

    def _on_preview(self, frame):
        try:
            h, w, ch = frame.shape
            img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(img).scaled(
                self._preview_lbl.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._preview_lbl.setPixmap(pix)
        except Exception:
            pass

    def _on_complete(self, results: dict):
        self._cleanup_processing_thread(wait_ms=5000)

        save_analysis(results)
        self._reset_form_state()
        self.analysis_done.emit(results)

    def _on_error(self, msg: str):
        self._cleanup_processing_thread(wait_ms=5000)
        self._reset_form_state()
        QMessageBox.critical(self, t('upload_error_title'),
                             t('upload_error_msg').format(msg))

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _reset_form_state(self):
        self._analyze_btn.setEnabled(True)
        self._progress_card.setVisible(False)
        self._preview_lbl.setVisible(False)

    def closeEvent(self, event):
        self._cancel_analysis()
        super().closeEvent(event)
