"""
UI 樣式集中管理模組。

以常數色票與樣式產生函式統一定義 PyQt 樣式字串，
供首頁、即時視窗、上傳分析、歷史清單與結果頁共用。
"""

# ============================================================================
# COLOR PALETTE
# ============================================================================
# Dark theme colors
DARK_BG = "#2c3e50"
DARK_BG_ALT = "#34495e"
DARK_BG_RGBA = "rgba(44, 62, 80, 0.95)"
DARK_BG_ALT_RGBA = "rgba(52, 73, 94, 0.95)"
VERY_DARK = "#1a1a1a"

# Light colors
LIGHT_TEXT = "#ecf0f1"
LIGHT_GRAY = "#95a5a6"
DARK_GRAY = "#7f8c8d"
LIGHT_GRAY_INACTIVE = "#bdc3c7"

# Primary accent
PRIMARY_ACCENT = "#3498db"
DARK_ACCENT = "#34495e"

# Button colors
BTN_GREEN = "#27ae60"
BTN_GREEN_DARK = "#229954"
BTN_GREEN_LIGHT = "#2ecc71"
BTN_RED = "#e74c3c"
BTN_RED_DARK = "#c0392b"
BTN_RED_LIGHT = "#ec7063"
BTN_ORANGE = "#f39c12"
BTN_ORANGE_DARK = "#e67e22"
BTN_ORANGE_LIGHT = "#f4a742"
BTN_TEAL = "#16a085"
BTN_TEAL_DARK = "#138d75"
BTN_TEAL_LIGHT = "#1abc9c"
BTN_PURPLE = "#8e44ad"
BTN_PURPLE_DARK = "#6c3483"
BTN_PURPLE_LIGHT = "#9b59b6"

# ============================================================================
# STYLE GENERATORS (Macro functions)
# ============================================================================
def _button_style(normal_light, normal_dark, hover_light, hover_dark,
                  pressed, width=None, height=None, font_size=16):
    """Generate button stylesheet with gradient, hover, and pressed states"""
    # 只在有指定寬度時設定 min-width 和 max-width
    width_style = f"min-width: {width}px; max-width: {width}px;" if width else ""
    pad_v = 12 if font_size == 16 else 8
    pad_h = 16 if font_size == 16 else 12
    grad_n = f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {normal_light}, stop:1 {normal_dark})"
    grad_h = f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {hover_light}, stop:1 {hover_dark})"
    return f"""
            QPushButton {{
                background: {grad_n};
                color: white;
                font-size: {font_size}px;
                font-weight: bold;
                padding: {pad_v}px {pad_h}px;
                border: none;
                border-radius: 8px;
                {width_style}
            }}
            QPushButton:hover {{
                background: {grad_h};
            }}
            QPushButton:pressed {{
                background: {pressed};
            }}
            QPushButton:disabled {{
                background: {DARK_GRAY};
                color: {LIGHT_GRAY_INACTIVE};
            }}
        """

def _group_box_style(title_content=""):
    """Generate group box stylesheet for panels"""
    return f"""
            QGroupBox {{
                font-size: 16px;
                font-weight: bold;
                color: {PRIMARY_ACCENT};
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {DARK_BG_RGBA}, stop:1 {DARK_BG_ALT_RGBA});
                border: 2px solid {PRIMARY_ACCENT};
                border-radius: 12px;
                margin-top: 15px;
                padding: 20px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 5px 15px;
                background-color: {DARK_BG};
                border-radius: 6px;
            }}
            {title_content}
        """

def _messagebox_style(font_size=11, label_min_width=""):
    """Generate message box stylesheet (QMessageBox and dialogs)"""
    min_width = f"min-width: {label_min_width};" if label_min_width else ""
    return f"""
            QMessageBox {{
                background-color: white;
            }}
            QLabel {{
                color: black;
                font-size: {font_size}px;
                {min_width}
            }}
            QPushButton {{
                color: black;
                background-color: #e0e0e0;
                border: 1px solid #999;
                padding: 5px 15px;
                min-width: 60px;
            }}
            QPushButton:hover {{
                background-color: #d0d0d0;
            }}
        """

def _dialog_style():
    """Generate dialog stylesheet with dark theme and combo boxes"""
    return f"""
            QDialog {{
                background-color: {DARK_BG};
                color: {LIGHT_TEXT};
            }}
            QLabel {{
                color: {LIGHT_TEXT};
                font-size: 13px;
            }}
            QComboBox {{
                background-color: {DARK_BG_ALT};
                color: {LIGHT_TEXT};
                border: 1px solid {PRIMARY_ACCENT};
                border-radius: 4px;
                padding: 5px;
                font-size: 12px;
                min-width: 100px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: {DARK_BG_ALT};
                color: {LIGHT_TEXT};
                selection-background-color: {PRIMARY_ACCENT};
            }}
            QRadioButton {{
                color: {LIGHT_TEXT};
                font-size: 14px;
                spacing: 10px;
            }}
            QRadioButton::indicator {{
                width: 20px;
                height: 20px;
                border-radius: 10px;
                border: 2px solid {PRIMARY_ACCENT};
                background-color: {DARK_BG_ALT};
            }}
            QRadioButton::indicator:checked {{
                background-color: {PRIMARY_ACCENT};
                border: 2px solid {PRIMARY_ACCENT};
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 #ffffff, stop:0.5 {PRIMARY_ACCENT}, stop:1 {PRIMARY_ACCENT});
            }}
            QRadioButton::indicator:hover {{
                border: 2px solid #5dade2;
            }}
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {PRIMARY_ACCENT}, stop:1 #2980b9);
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                min-width: 80px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5dade2, stop:1 {PRIMARY_ACCENT});
            }}
        """

# ============================================================================
# GENERATED STYLES
# ============================================================================
MAIN_WINDOW_STYLE = f"""
            QMainWindow {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {DARK_BG}, stop:1 {DARK_BG_ALT});
            }}
            QWidget {{
                color: {LIGHT_TEXT};
                font-family: "Microsoft JhengHei", "微軟正黑體", Arial;
            }}
            QLabel {{
                color: {LIGHT_TEXT};
            }}
            QMessageBox {{
                background-color: #f8fafc;
            }}
            QMessageBox QLabel {{
                color: #0f172a;
                background: transparent;
            }}
            QMessageBox QPushButton {{
                color: #0f172a;
                background-color: #e2e8f0;
                border: 1px solid #94a3b8;
                border-radius: 6px;
                padding: 6px 14px;
                min-width: 72px;
            }}
            QMessageBox QPushButton:hover {{
                background-color: #cbd5e1;
            }}
        """

# Button styles using macro (移除固定寬度以適應不同解析度)
START_BUTTON_STYLE = _button_style(BTN_GREEN_LIGHT, BTN_GREEN, BTN_GREEN_LIGHT, BTN_GREEN, BTN_GREEN_DARK)
STOP_BUTTON_STYLE = _button_style(BTN_RED, BTN_RED_DARK, BTN_RED_LIGHT, BTN_RED, BTN_RED_DARK)
PAUSE_BUTTON_STYLE = _button_style(BTN_ORANGE, BTN_ORANGE_DARK, BTN_ORANGE_LIGHT, BTN_ORANGE, BTN_ORANGE_DARK)
SAVE_BUTTON_STYLE = _button_style(BTN_TEAL, BTN_TEAL_DARK, BTN_TEAL_LIGHT, BTN_TEAL, BTN_TEAL_DARK)
CONFIG_BUTTON_STYLE = _button_style(BTN_PURPLE, BTN_PURPLE_DARK, BTN_PURPLE_LIGHT, BTN_PURPLE, BTN_PURPLE_DARK, 
                                     width=50, font_size=24)

# Message box styles using macro
MESSAGEBOX_STANDARD_STYLE = _messagebox_style(font_size=12)
ERROR_MESSAGEBOX_STYLE = _messagebox_style(font_size=11)
SUCCESS_MESSAGEBOX_STYLE = _messagebox_style(font_size=11)
MESSAGEBOX_WIDE_STYLE = _messagebox_style(font_size=12, label_min_width="200px")
MESSAGEBOX_EXTRA_WIDE_STYLE = _messagebox_style(font_size=11, label_min_width="400px")

# Dialog style using macro
RULA_CONFIG_DIALOG_STYLE = _dialog_style()

VIDEO_LABEL_STYLE = f"""
            border: 3px solid {PRIMARY_ACCENT};
            border-radius: 10px;
            background-color: {VERY_DARK};
            font-size: 16px;
            color: {LIGHT_GRAY};
        """

# Panel styles using macro
_PANEL_LABEL = f"""
            QLabel {{
                color: {LIGHT_TEXT};
                background: transparent;
            }}
        """

_PANEL_TEXTEDIT = f"""
            QTextEdit {{
                background-color: rgba(26, 26, 26, 0.8);
                color: {LIGHT_TEXT};
                border: 1px solid {DARK_ACCENT};
                border-radius: 5px;
                font-family: "Courier New", monospace;
                font-size: 11px;
                padding: 5px;
            }}
        """

SCORE_PANEL_STYLE = _group_box_style(_PANEL_LABEL)
COORDINATES_PANEL_STYLE = _group_box_style(_PANEL_LABEL + _PANEL_TEXTEDIT)

FPS_LABEL_STYLE = f"""
            font-size: 16px;
            font-weight: bold;
            padding: 12px 20px;
            background-color: rgba(52, 73, 94, 0.7);
            border-radius: 8px;
            color: {PRIMARY_ACCENT};
        """

# Record button style (Red when recording，移除固定寬度)
RECORD_BUTTON_STYLE = _button_style(
    normal_light=BTN_RED_LIGHT,
    normal_dark=BTN_RED_DARK,
    hover_light=BTN_RED_LIGHT,
    hover_dark=BTN_RED,
    pressed=BTN_RED_DARK,
    font_size=16
)

# Record button style (Inactive/Ready to record，移除固定寬度)
RECORD_BUTTON_READY_STYLE = _button_style(
    normal_light="#7f8c8d",
    normal_dark="#5d6d7e",
    hover_light="#95a5a6",
    hover_dark="#7f8c8d",
    pressed="#5d6d7e",
    font_size=16
)

# ============================================================================
# LAUNCHER / UPLOAD / HISTORY WINDOW STYLES
# ============================================================================

# Launcher window main background (dark slate → dark emerald)
LAUNCHER_BG_STYLE = """
    QMainWindow {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #0f172a, stop:1 #064e3b);
    }
    QWidget {
        color: #ecf0f1;
        font-family: "Microsoft JhengHei", "微軟正黑體", Arial;
    }
    QMessageBox {
        background-color: #f8fafc;
    }
    QMessageBox QLabel {
        color: #0f172a;
        background: transparent;
    }
    QMessageBox QPushButton {
        color: #0f172a;
        background-color: #e2e8f0;
        border: 1px solid #94a3b8;
        border-radius: 6px;
        padding: 6px 14px;
        min-width: 72px;
    }
    QMessageBox QPushButton:hover {
        background-color: #cbd5e1;
    }
"""

LAUNCHER_CENTRAL_STYLE = """
    QWidget#launcher_bg {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #0f172a, stop:1 #064e3b);
    }
"""

# Header card (white card on top)
HEADER_CARD_STYLE = """
    QFrame#header_card {
        background-color: rgba(255, 255, 255, 0.97);
        border-radius: 16px;
    }
    QLabel {
        color: #0f172a;
        background: transparent;
    }
"""

# Mode card (white card for each launch option)
MODE_CARD_STYLE = """
    QFrame#mode_card {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
    }
    QLabel {
        background: transparent;
    }
"""

# Emerald launch button (即時分析)
LAUNCH_BTN_EMERALD = _button_style(
    "#10b981", "#059669", "#34d399", "#10b981", "#047857", font_size=15
)

# Blue launch button (上傳影片)
LAUNCH_BTN_BLUE = _button_style(
    "#3b82f6", "#2563eb", "#60a5fa", "#3b82f6", "#1e40af", font_size=15
)

# Upload/history window main background
UPLOAD_BG_STYLE = """
    QMainWindow {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #0f172a, stop:1 #064e3b);
    }
    QWidget {
        color: #ecf0f1;
        font-family: "Microsoft JhengHei", "微軟正黑體", Arial;
    }
    QMessageBox {
        background-color: #f8fafc;
    }
    QMessageBox QLabel {
        color: #0f172a;
        background: transparent;
    }
    QMessageBox QPushButton {
        color: #0f172a;
        background-color: #e2e8f0;
        border: 1px solid #94a3b8;
        border-radius: 6px;
        padding: 6px 14px;
        min-width: 72px;
    }
    QMessageBox QPushButton:hover {
        background-color: #cbd5e1;
    }
"""

CONTENT_CARD_STYLE = """
    QFrame#content_card {
        background-color: rgba(255, 255, 255, 0.97);
        border-radius: 16px;
    }
    QLabel {
        color: #0f172a;
        background: transparent;
    }
    QLineEdit, QDateEdit, QSpinBox, QComboBox {
        background-color: #f8fafc;
        color: #0f172a;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 13px;
    }
    QLineEdit:focus, QDateEdit:focus, QSpinBox:focus, QComboBox:focus {
        border: 1px solid #10b981;
    }
    QGroupBox {
        color: #0f172a;
        font-size: 14px;
        font-weight: bold;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-top: 12px;
        padding: 10px;
        background: transparent;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 2px 8px;
        color: #059669;
    }
    /* Improve visibility of checkboxes (larger indicator, stronger contrast) */
    QCheckBox {
        color: #0f172a;
        font-size: 14px;
        spacing: 8px;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border: 2px solid #94a3b8;
        border-radius: 4px;
        background: transparent;
    }
    /* Explicit unchecked state: no image */
    QCheckBox::indicator:unchecked {
        image: none;
        background-color: transparent;
        border: 2px solid #94a3b8;
    }
    QCheckBox::indicator:checked {
        image: none;
        background-color: #10b981;
        border: 2px solid #10b981;
    }
    QCheckBox::indicator:hover {
        border: 2px solid #059669;
    }
    QTableWidget {
        background-color: white;
        color: #0f172a;
        border: none;
        gridline-color: #f1f5f9;
        font-size: 13px;
    }
    QTableWidget::item:selected {
        background-color: #ecfdf5;
        color: #0f172a;
    }
    QHeaderView::section {
        background-color: #f8fafc;
        color: #0f172a;
        font-weight: bold;
        font-size: 13px;
        border: none;
        border-bottom: 2px solid #e2e8f0;
        padding: 8px;
    }
    QProgressBar {
        background-color: #f1f5f9;
        border-radius: 6px;
        border: none;
        height: 12px;
        text-align: center;
        color: #0f172a;
        font-size: 11px;
    }
    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #10b981, stop:1 #059669);
        border-radius: 6px;
    }
"""

# Back / action buttons inside upload/history
BACK_BTN_STYLE = _button_style(
    "#64748b", "#475569", "#94a3b8", "#64748b", "#334155", font_size=13
)
EMERALD_BTN_STYLE = _button_style(
    "#10b981", "#059669", "#34d399", "#10b981", "#047857", font_size=13
)
BLUE_BTN_STYLE = _button_style(
    "#3b82f6", "#2563eb", "#60a5fa", "#3b82f6", "#1e40af", font_size=13
)
RED_BTN_STYLE = _button_style(
    BTN_RED, BTN_RED_DARK, BTN_RED_LIGHT, BTN_RED, BTN_RED_DARK, font_size=13
)
