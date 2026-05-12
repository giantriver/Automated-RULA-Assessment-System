"""
對話框模組。

提供系統設定相關對話框，主要包含：
- RULA 固定參數設定（Table A/B 參數）
- 即時姿勢辨識後端切換（MediaPipe / RTMW3D）
- 語言切換與套用
- FrameMetricsDialog：單幀角度、Table 分數與 3D 骨架
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                             QGridLayout, QPushButton, QRadioButton, QButtonGroup,
                             QScrollArea, QWidget, QFrame, QSizePolicy)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from ..core import config
from ..core.config import RTMW_TO_MEDIAPIPE
from .styles import RULA_CONFIG_DIALOG_STYLE, BACK_BTN_STYLE
from .language import language_manager, t


class RULAConfigDialog(QDialog):
    """Configuration dialog for RULA parameters with dropdown controls"""
    
    def __init__(self, parent=None, current_backend_mode='MEDIAPIPE'):
        super().__init__(parent)
        self.lang = language_manager
        self.lang.add_observer(self.on_language_changed)
        self.backend_modes = ['MEDIAPIPE', 'RTMW3D']
        self.selected_backend_mode = current_backend_mode if current_backend_mode in self.backend_modes else 'MEDIAPIPE'
        
        self.setWindowTitle(t('config_title'))
        self.setMinimumSize(420, 360)
        self.resize(500, 560)
        self.setStyleSheet(RULA_CONFIG_DIALOG_STYLE)
        
        # Store references to combo boxes for retrieval
        self.combos = {}
        
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        scroll = QScrollArea()
        scroll.setObjectName('configScroll')
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea#configScroll {
                background-color: rgba(52, 73, 94, 0.35);
                border: 1px solid #3f5b73;
                border-radius: 10px;
            }
            QScrollArea#configScroll > QWidget > QWidget {
                background-color: #2c3e50;
                border-radius: 8px;
            }
            QScrollBar:vertical {
                background: #34495e;
                width: 10px;
                margin: 8px 2px 8px 2px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #5dade2;
                min-height: 28px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #85c1e9;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)

        content_widget = QWidget()
        content_widget.setObjectName('configContent')
        content_widget.setStyleSheet("""
            QWidget#configContent {
                background-color: rgba(44, 62, 80, 0.92);
                border-radius: 8px;
            }
        """)
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(12)
        content_layout.setContentsMargins(0, 0, 8, 0)
        
        # 標題
        self.title_label = QLabel(t('config_subtitle'))
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #3498db; margin-bottom: 10px;")
        content_layout.addWidget(self.title_label)

        # 語言選擇
        language_layout = QHBoxLayout()
        self.language_label = QLabel(t('config_language'))
        self.language_label.setStyleSheet("font-weight: bold; color: #ecf0f1;")

        self.language_combo = QComboBox()
        self.language_codes = ['zh_TW', 'en']
        for lang_code in self.language_codes:
            self.language_combo.addItem(t('lang_chinese') if lang_code == 'zh_TW' else t('lang_english'), lang_code)

        current_lang = self.lang.get_language()
        current_lang_index = self.language_combo.findData(current_lang)
        if current_lang_index >= 0:
            self.language_combo.setCurrentIndex(current_lang_index)

        language_layout.addWidget(self.language_label)
        language_layout.addWidget(self.language_combo)
        content_layout.addLayout(language_layout)

        # 姿勢偵測後端選擇
        backend_layout = QHBoxLayout()
        self.backend_label = QLabel(t('config_pose_backend'))
        self.backend_label.setStyleSheet("font-weight: bold; color: #ecf0f1;")

        self.backend_combo = QComboBox()
        self.backend_combo.addItem(t('config_option_backend_mediapipe'), 'MEDIAPIPE')
        self.backend_combo.addItem(t('config_option_backend_rtmw3d'), 'RTMW3D')

        backend_index = self.backend_combo.findData(self.selected_backend_mode)
        if backend_index >= 0:
            self.backend_combo.setCurrentIndex(backend_index)

        backend_layout.addWidget(self.backend_label)
        backend_layout.addWidget(self.backend_combo)
        content_layout.addLayout(backend_layout)

        self.backend_desc_label = QLabel(t('config_pose_backend_desc'))
        self.backend_desc_label.setStyleSheet("font-size: 11px; color: #95a5a6; margin-bottom: 8px;")
        self.backend_desc_label.setWordWrap(True)
        content_layout.addWidget(self.backend_desc_label)

        # 參數網格
        grid_layout = QGridLayout()
        grid_layout.setSpacing(12)
        grid_layout.setColumnStretch(0, 2)
        grid_layout.setColumnStretch(1, 1)
        
        # 保存標籤引用以便語言切換時更新
        self.param_name_labels = {}
        self.param_desc_labels = {}
        
        # Define parameter options with translation keys
        param_config = [
            ("wrist_twist", "config_wrist_twist", "config_wrist_twist_desc", 
             ["config_option_wrist_1", "config_option_wrist_2"]),
            ("legs", "config_legs", "config_legs_desc",
             ["config_option_legs_1", "config_option_legs_2"]),
            ("muscle_use_a", "config_muscle_use_a", "config_muscle_use_a_desc",
             ["config_option_muscle_0", "config_option_muscle_1"]),
            ("muscle_use_b", "config_muscle_use_b", "config_muscle_use_b_desc",
             ["config_option_muscle_0", "config_option_muscle_1"]),
            ("force_load_a", "config_force_load_a", "config_force_load_a_desc",
             ["config_option_force_0", "config_option_force_1", "config_option_force_2"]),
            ("force_load_b", "config_force_load_b", "config_force_load_b_desc",
             ["config_option_force_0", "config_option_force_1", "config_option_force_2"]),
        ]
        
        row = 0
        for param_key, name_key, desc_key, option_keys in param_config:
            # 參數名稱
            name_label = QLabel(t(name_key))
            name_label.setStyleSheet("font-weight: bold; color: #ecf0f1;")
            grid_layout.addWidget(name_label, row, 0)
            self.param_name_labels[param_key] = (name_label, name_key)
            
            # 下拉選單
            combo = QComboBox()
            # 添加翻譯的選項文本
            for opt_key in option_keys:
                combo.addItem(t(opt_key))
            
            # 獲取當前值並轉換為正確的索引
            current_value = config.RULA_CONFIG[param_key]
            # wrist_twist 和 legs 的值是 1-2，需要轉換為索引 0-1
            if param_key in ['wrist_twist', 'legs']:
                current_index = current_value - 1
            else:
                # muscle_use 和 force_load 的值直接對應索引
                current_index = current_value
            
            combo.setCurrentIndex(current_index)
            self.combos[param_key] = (combo, option_keys)  # 保存combo和翻譯鍵
            grid_layout.addWidget(combo, row, 1)
            row += 1
            
            # 參數說明
            desc_label = QLabel(t(desc_key))
            desc_label.setStyleSheet("font-size: 11px; color: #95a5a6; margin-bottom: 8px;")
            desc_label.setWordWrap(True)
            grid_layout.addWidget(desc_label, row, 0, 1, 2)
            self.param_desc_labels[param_key] = (desc_label, desc_key)
            row += 1
        
        content_layout.addLayout(grid_layout)
        content_layout.addStretch()

        scroll.setWidget(content_widget)
        layout.addWidget(scroll, 1)
        
        # 按鈕佈局
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # 保存按鈕
        self.save_button = QPushButton(t('config_save'))
        self.save_button.clicked.connect(self.save_config)
        button_layout.addWidget(self.save_button)
        
        # 關閉按鈕
        self.close_button = QPushButton(t('config_cancel'))
        self.close_button.clicked.connect(self.reject)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def on_language_changed(self, lang_code):
        """語言改變時更新對話框文本"""
        self.setWindowTitle(t('config_title'))
        self.title_label.setText(t('config_subtitle'))
        self.language_label.setText(t('config_language'))
        self.backend_label.setText(t('config_pose_backend'))
        self.backend_desc_label.setText(t('config_pose_backend_desc'))
        self.save_button.setText(t('config_save'))
        self.close_button.setText(t('config_cancel'))

        # 更新語言下拉選單顯示文字，保留目前選中的語言代碼
        selected_lang = self.language_combo.currentData()
        self.language_combo.clear()
        for code in self.language_codes:
            self.language_combo.addItem(t('lang_chinese') if code == 'zh_TW' else t('lang_english'), code)
        selected_index = self.language_combo.findData(selected_lang)
        if selected_index >= 0:
            self.language_combo.setCurrentIndex(selected_index)

        # 更新姿勢後端下拉選單顯示文字，保留目前選中的後端
        selected_backend = self.backend_combo.currentData()
        self.backend_combo.clear()
        self.backend_combo.addItem(t('config_option_backend_mediapipe'), 'MEDIAPIPE')
        self.backend_combo.addItem(t('config_option_backend_rtmw3d'), 'RTMW3D')
        selected_backend_index = self.backend_combo.findData(selected_backend)
        if selected_backend_index >= 0:
            self.backend_combo.setCurrentIndex(selected_backend_index)

        # 更新參數名稱標籤
        for param_key, (label, name_key) in self.param_name_labels.items():
            label.setText(t(name_key))
        
        # 更新參數描述標籤
        for param_key, (label, desc_key) in self.param_desc_labels.items():
            label.setText(t(desc_key))
        
        # 更新下拉選單選項
        for param_key, (combo, option_keys) in self.combos.items():
            current_index = combo.currentIndex()
            combo.clear()
            for opt_key in option_keys:
                combo.addItem(t(opt_key))
            combo.setCurrentIndex(current_index)
    
    def save_config(self):
        """Save the current parameter values back to config"""
        selected_lang = self.language_combo.currentData()

        for param_key, (combo, option_keys) in self.combos.items():
            # 獲取選中的索引
            index = combo.currentIndex()

            # wrist_twist 和 legs 的索引 0-1 需要轉換為值 1-2
            if param_key in ['wrist_twist', 'legs']:
                value = index + 1
            else:
                # muscle_use 和 force_load 的索引直接對應值
                value = index

            config.RULA_CONFIG[param_key] = value

        # 套用語言（會透過 observer 通知主視窗與各元件）
        if selected_lang in ('en', 'zh_TW'):
            self.lang.set_language(selected_lang)

        selected_backend = self.backend_combo.currentData()
        if selected_backend in self.backend_modes:
            self.selected_backend_mode = selected_backend

        self.accept()

    def get_selected_backend_mode(self):
        """取得使用者選擇的姿勢偵測後端。"""
        return self.selected_backend_mode


class LanguageSelectionDialog(QDialog):
    """Language selection dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_language = language_manager.get_language()
        
        self.setWindowTitle(t('lang_dialog_title'))
        self.setMinimumSize(400, 200)
        self.setStyleSheet(RULA_CONFIG_DIALOG_STYLE)
        
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 标题
        title_label = QLabel(t('lang_dialog_subtitle'))
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #3498db;")
        layout.addWidget(title_label)
        
        # 语言选项
        self.button_group = QButtonGroup(self)
        
        # 英文选项
        self.en_radio = QRadioButton(t('lang_english'))
        self.en_radio.setStyleSheet("font-size: 14px; color: #ecf0f1; padding: 10px;")
        if self.selected_language == 'en':
            self.en_radio.setChecked(True)
        self.button_group.addButton(self.en_radio)
        layout.addWidget(self.en_radio)
        
        # 繁体中文选项
        self.zh_radio = QRadioButton(t('lang_chinese'))
        self.zh_radio.setStyleSheet("font-size: 14px; color: #ecf0f1; padding: 10px;")
        if self.selected_language == 'zh_TW':
            self.zh_radio.setChecked(True)
        self.button_group.addButton(self.zh_radio)
        layout.addWidget(self.zh_radio)
        
        layout.addStretch()
        
        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # 确认按钮
        confirm_button = QPushButton(t('lang_confirm'))
        confirm_button.clicked.connect(self.confirm_selection)
        button_layout.addWidget(confirm_button)
        
        # 取消按钮
        cancel_button = QPushButton(t('lang_cancel'))
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def confirm_selection(self):
        """确认语言选择"""
        if self.en_radio.isChecked():
            self.selected_language = 'en'
        else:
            self.selected_language = 'zh_TW'
        self.accept()
    
    def get_selected_language(self):
        """获取选择的语言"""
        return self.selected_language


# ──────────────────────────────────────────────────────────────────────────────
# FrameMetricsDialog
# ──────────────────────────────────────────────────────────────────────────────

#: Joint indices required for each body part confidence check
_JOINT_GROUPS = {
    ('left',  'Upper Arm'): [(11, 'L Shoulder'), (13, 'L Elbow'),
                              (23, 'L Hip'), (24, 'R Hip')],
    ('right', 'Upper Arm'): [(12, 'R Shoulder'), (14, 'R Elbow'),
                              (23, 'L Hip'), (24, 'R Hip')],
    ('left',  'Lower Arm'): [(11, 'L Shoulder'), (13, 'L Elbow'), (15, 'L Wrist')],
    ('right', 'Lower Arm'): [(12, 'R Shoulder'), (14, 'R Elbow'), (16, 'R Wrist')],
    ('left',  'Wrist'):     [(15, 'L Wrist'), (17, 'L Pinky'), (19, 'L Index')],
    ('right', 'Wrist'):     [(16, 'R Wrist'), (18, 'R Pinky'), (20, 'R Index')],
    ('left',  'Neck'):      [(7, 'L Ear'), (8, 'R Ear'), (11, 'L Shoulder'),
                              (12, 'R Shoulder'), (23, 'L Hip'), (24, 'R Hip')],
    ('right', 'Neck'):      [(7, 'L Ear'), (8, 'R Ear'), (11, 'L Shoulder'),
                              (12, 'R Shoulder'), (23, 'L Hip'), (24, 'R Hip')],
    ('left',  'Trunk'):     [(11, 'L Shoulder'), (12, 'R Shoulder'),
                              (23, 'L Hip'), (24, 'R Hip')],
    ('right', 'Trunk'):     [(11, 'L Shoulder'), (12, 'R Shoulder'),
                              (23, 'L Hip'), (24, 'R Hip')],
}

_MIN_CONF = 0.5


class FrameMetricsDialog(QDialog):
    """
    單幀角度、Table A/B 分數與 3D 骨架對話框。

    Parameters
    ----------
    rec : dict
        records 清單中的單一幀資料字典（含 native_draw_data、角度、分數欄位）。
    frame_label : str
        顯示在頂部的幀資訊文字（例如 "Frame 3 / 615   Time: 0.10 s"）。
    render_3d_fn : callable
        簽名：(landmarks_3d: list, width: int, height: int) -> QPixmap | None
        由 result_window 傳入，避免在此模組重複引入 matplotlib 相關依賴。
    parent : QWidget, optional
    """

    # ── Layout constants (easy to tweak) ─────────────────────────────────────
    SKL_W: int = 420        # 3D 骨架圖寬度（px）
    SKL_H: int = 420        # 3D 骨架圖高度（px）
    DLG_MIN_W: int = 1080   # 對話框最小寬度
    RIGHT_MIN_W: int = 580  # 右側面板最小寬度
    SECTION_MIN_W: int = 240  # 每個 section card 最小寬度

    def __init__(self, rec: dict, frame_label: str, render_3d_fn, parent=None):
        super().__init__(parent)
        self._rec = rec
        self._frame_label = frame_label
        self._render_3d_fn = render_3d_fn

        self.setWindowTitle(t('result_metrics_title'))
        self.setMinimumWidth(self.DLG_MIN_W)
        self.setStyleSheet('QDialog { background: #f8fafc; }')

        native = rec.get('native_draw_data') or {}
        self._native = native
        self._backend = str(native.get('backend', '')).upper()

        outer = QVBoxLayout(self)
        outer.setSpacing(8)
        outer.setContentsMargins(20, 10, 20, 12)

        # ── Frame info ────────────────────────────────────────────────────────
        info_lbl = QLabel(frame_label)
        info_lbl.setStyleSheet('color: #64748b; font-size: 12px;')
        outer.addWidget(info_lbl)

        # ── Body row ──────────────────────────────────────────────────────────
        body_row = QHBoxLayout()
        body_row.setSpacing(14)
        body_row.addWidget(self._build_skeleton_panel())
        body_row.addWidget(self._build_right_panel(), 1)
        outer.addLayout(body_row, 1)

        # ── Close button ──────────────────────────────────────────────────────
        close_btn = QPushButton(t('result_close_btn'))
        close_btn.setStyleSheet(BACK_BTN_STYLE)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

    # ── Private builders ──────────────────────────────────────────────────────

    def _build_skeleton_panel(self) -> QFrame:
        w, h = self.SKL_W, self.SKL_H
        panel = QFrame()
        panel.setFixedSize(w, h + 28)
        panel.setStyleSheet(
            'QFrame { background: white; border-radius: 8px; border: 1px solid #e2e8f0; }'
            'QLabel { background: transparent; }'
        )
        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 4, 0, 0)
        col.setSpacing(2)

        title = QLabel('3D Skeleton')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('color: #475569; font-size: 11px;')
        col.addWidget(title)

        img_lbl = QLabel()
        img_lbl.setFixedSize(w, h)
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        col.addWidget(img_lbl)

        lm3d = self._native.get('landmarks_3d', [])
        pixmap = self._render_3d_fn(lm3d, w, h, self._native)
        if pixmap is not None:
            img_lbl.setPixmap(
                pixmap.scaled(w, h,
                              Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
            )
        else:
            img_lbl.setText('No 3D data')
            img_lbl.setStyleSheet('color: #94a3b8; font-size: 12px;')

        return panel

    def _build_right_panel(self) -> QWidget:
        rec = self._rec
        widget = QWidget()
        widget.setMinimumWidth(self.RIGHT_MIN_W)
        col = QVBoxLayout(widget)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)

        # angles row (left + right)
        angles_row = QHBoxLayout()
        angles_row.setSpacing(10)
        angles_row.addWidget(self._make_section(
            t('result_metrics_left_angles'),
            [
                (t('upper_arm'), self._fmt_angle(rec.get('left_upper_arm_angle')),  'left',  'Upper Arm'),
                (t('lower_arm'), self._fmt_angle(rec.get('left_lower_arm_angle')),  'left',  'Lower Arm'),
                (t('wrist'),     self._fmt_angle(rec.get('left_wrist_angle')),      'left',  'Wrist'),
                (t('neck'),      self._fmt_angle(rec.get('left_neck_angle')),       'left',  'Neck'),
                (t('trunk'),     self._fmt_angle(rec.get('left_trunk_angle')),      'left',  'Trunk'),
            ],
            '#f0f9ff',
        ))
        angles_row.addWidget(self._make_section(
            t('result_metrics_right_angles'),
            [
                (t('upper_arm'), self._fmt_angle(rec.get('right_upper_arm_angle')), 'right', 'Upper Arm'),
                (t('lower_arm'), self._fmt_angle(rec.get('right_lower_arm_angle')), 'right', 'Lower Arm'),
                (t('wrist'),     self._fmt_angle(rec.get('right_wrist_angle')),     'right', 'Wrist'),
                (t('neck'),      self._fmt_angle(rec.get('right_neck_angle')),      'right', 'Neck'),
                (t('trunk'),     self._fmt_angle(rec.get('right_trunk_angle')),     'right', 'Trunk'),
            ],
            '#f0fdf4',
        ))
        col.addLayout(angles_row, 3)

        # scores row (left + right)
        scores_row = QHBoxLayout()
        scores_row.setSpacing(10)
        final_key = t('final_score')
        scores_row.addWidget(self._make_section(
            t('result_metrics_left_scores'),
            [
                ('Table A',  self._fmt_score(rec.get('left_posture_score_a'))),
                ('Table B',  self._fmt_score(rec.get('left_posture_score_b'))),
                (final_key,  self._fmt_score(rec.get('left_score'))),
            ],
            '#eff6ff',
        ))
        scores_row.addWidget(self._make_section(
            t('result_metrics_right_scores'),
            [
                ('Table A',  self._fmt_score(rec.get('right_posture_score_a'))),
                ('Table B',  self._fmt_score(rec.get('right_posture_score_b'))),
                (final_key,  self._fmt_score(rec.get('right_score'))),
            ],
            '#fefce8',
        ))
        col.addLayout(scores_row, 2)

        return widget

    def _make_section(self, title: str, rows: list, bg: str = '#f1f5f9') -> QFrame:
        """
        Build a labelled card.
        rows: list of (label, value) or (label, value, side, part_name)
        """
        frame = QFrame()
        frame.setStyleSheet(
            f'QFrame {{ background: {bg}; border-radius: 8px; }}'
            'QLabel { background: transparent; }'
            'QPushButton { background: transparent; }'
        )
        frame.setMinimumWidth(self.SECTION_MIN_W)
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        col = QVBoxLayout(frame)
        col.setContentsMargins(12, 8, 12, 8)
        col.setSpacing(4)

        title_lbl = QLabel(title)
        title_lbl.setFont(QFont('Microsoft JhengHei', 11, QFont.Weight.Bold))
        title_lbl.setStyleSheet('color: #0f172a;')
        col.addWidget(title_lbl)

        for row_item in rows:
            label, value = row_item[0], row_item[1]
            click_side = row_item[2] if len(row_item) > 2 else None
            click_part = row_item[3] if len(row_item) > 3 else None
            is_null = (value == 'NULL')
            r = QHBoxLayout()

            if click_side is not None:
                lbl = QPushButton(label + ' ›')
                lbl.setFlat(True)
                lbl.setCursor(Qt.CursorShape.PointingHandCursor)
                lbl.setStyleSheet(
                    f'QPushButton {{ color: {"#ef4444" if is_null else "#0369a1"};'
                    'font-size: 12px; text-align: left; padding: 0; border: none; }}'
                    f'QPushButton:hover {{ color: {"#b91c1c" if is_null else "#0284c7"};'
                    'text-decoration: underline; }}'
                )
                _s, _p = click_side, click_part
                lbl.clicked.connect(
                    lambda checked=False, s=_s, p=_p: self._show_joint_popup(s, p)
                )
            else:
                lbl = QLabel(label)
                lbl.setStyleSheet('color: #475569; font-size: 12px;')

            val_lbl = QLabel(value)
            val_lbl.setStyleSheet(
                f'color: {"#ef4444" if is_null else "#0f172a"};'
                'font-size: 12px; font-family: Consolas, monospace;'
            )
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            r.addWidget(lbl)
            r.addStretch()
            r.addWidget(val_lbl)
            col.addLayout(r)

        col.addStretch()
        return frame

    def _show_joint_popup(self, side: str, part_name: str):
        joints = _JOINT_GROUPS.get((side, part_name), [])
        joint_anomaly_detail = self._rec.get('joint_anomaly_detail') or []

        popup = QDialog(self)
        popup.setWindowTitle(f'{part_name} — Joint Confidence')
        popup.setMinimumWidth(380)
        popup.setStyleSheet(
            'QDialog { background: #1e293b; } QLabel { background: transparent; }'
        )
        layout = QVBoxLayout(popup)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        title_lbl = QLabel(f'<b>{part_name}</b> required keypoints')
        title_lbl.setStyleSheet('color: #94a3b8; font-size: 12px;')
        layout.addWidget(title_lbl)

        # Header row
        header_row = QHBoxLayout()
        for txt, stretch in [('Joint', 3), ('Confidence', 2), ('Anomaly', 3), ('Speed ratio', 2)]:
            h = QLabel(txt)
            h.setStyleSheet('color: #64748b; font-size: 10px;')
            if txt in ('Confidence', 'Speed ratio', 'Anomaly'):
                h.setAlignment(Qt.AlignmentFlag.AlignRight)
            header_row.addWidget(h, stretch)
        layout.addLayout(header_row)

        for idx, name in joints:
            conf = self._get_conf(idx)
            det  = joint_anomaly_detail[idx] if idx < len(joint_anomaly_detail) else None

            # ── Confidence ──────────────────────────────────────────────
            if conf is None:
                conf_text, conf_color = 'N/A', '#94a3b8'
            else:
                passes = conf >= _MIN_CONF
                conf_text  = f'{conf:.3f}  {"✓" if passes else "✗"}'
                conf_color = '#4ade80' if passes else '#f87171'

            # ── Anomaly reason ──────────────────────────────────────────
            if det is None:
                reason_text  = '—'
                reason_color = '#4ade80'
            else:
                reason_map = {
                    'low_visibility':    'low_vis',
                    'speed_jump':        'speed_jump',
                    'low_vis_speed_jump':'low_vis+spd',
                }
                reason_text  = reason_map.get(det.get('reason', ''), det.get('reason', '?'))
                reason_color = '#fb923c'

            # ── Speed ratio ─────────────────────────────────────────────
            sr = det.get('speed_ratio') if det else None
            sr_text  = f'{sr:.3f}' if sr is not None else '—'
            sr_color = '#fb923c' if sr is not None else '#94a3b8'

            row = QHBoxLayout()
            name_lbl = QLabel(f'[{idx:2d}] {name}')
            name_lbl.setStyleSheet(
                'color: #e2e8f0; font-size: 12px; font-family: Consolas, monospace;'
            )
            conf_lbl = QLabel(conf_text)
            conf_lbl.setStyleSheet(
                f'color: {conf_color}; font-size: 12px; font-family: Consolas, monospace;'
            )
            conf_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            reason_lbl = QLabel(reason_text)
            reason_lbl.setStyleSheet(
                f'color: {reason_color}; font-size: 12px; font-family: Consolas, monospace;'
            )
            reason_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            sr_lbl = QLabel(sr_text)
            sr_lbl.setStyleSheet(
                f'color: {sr_color}; font-size: 12px; font-family: Consolas, monospace;'
            )
            sr_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)

            row.addWidget(name_lbl, 3)
            row.addWidget(conf_lbl, 2)
            row.addWidget(reason_lbl, 3)
            row.addWidget(sr_lbl, 2)
            layout.addLayout(row)

        close = QPushButton('Close')
        close.setStyleSheet(
            'QPushButton { background:#334155; color:#e2e8f0; border-radius:6px;'
            'padding:4px 14px; font-size:12px; }'
            'QPushButton:hover { background:#475569; }'
        )
        close.clicked.connect(popup.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close)
        layout.addLayout(btn_row)
        popup.exec()

    def _get_conf(self, idx: int):
        native = self._native
        backend = self._backend
        lm3d = native.get('landmarks_3d', [])
        if idx < len(lm3d) and len(lm3d[idx]) >= 4:
            return float(lm3d[idx][3])
        if backend == 'MEDIAPIPE':
            lms = native.get('landmarks_2d', [])
            if idx < len(lms) and len(lms[idx]) >= 3:
                return float(lms[idx][2])
        if backend == 'RTMW3D':
            rtmw_idx = RTMW_TO_MEDIAPIPE.get(idx)
            if rtmw_idx is not None:
                scores_list = native.get('scores', [])
                if rtmw_idx < len(scores_list):
                    return float(scores_list[rtmw_idx])
        return None

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_angle(v) -> str:
        if v is None or v == 'NULL':
            return 'NULL'
        try:
            return f'{float(v):.1f}°'
        except (ValueError, TypeError):
            return str(v)

    @staticmethod
    def _fmt_score(v) -> str:
        if v is None or v == 'NULL':
            return 'NULL'
        return str(v)