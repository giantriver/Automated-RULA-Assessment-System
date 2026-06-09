"""
分析歷史清單視窗。

負責載入本機歷史 JSON，顯示每筆分析摘要，並提供：
- 查看完整結果
- 匯出 CSV
- 單筆 / 批量刪除
"""

import os
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QMessageBox, QFileDialog, QAbstractItemView
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from ..core.video_file_processor import load_history, export_csv
from .styles import (
    UPLOAD_BG_STYLE, CONTENT_CARD_STYLE, HEADER_CARD_STYLE,
    BACK_BTN_STYLE, EMERALD_BTN_STYLE, BLUE_BTN_STYLE, RED_BTN_STYLE,
)
from .language import language_manager, t


# ──────────────────────────────────────────────────────────────────────────────
_SCORE_COLORS = {
    1: ('#d1fae5', '#065f46'),
    2: ('#d1fae5', '#065f46'),
    3: ('#fef3c7', '#92400e'),
    4: ('#fef3c7', '#92400e'),
    5: ('#fee2e2', '#991b1b'),
    6: ('#fee2e2', '#991b1b'),
    7: ('#fca5a5', '#7c2d12'),
}


def _score_badge(score):
    return _SCORE_COLORS.get(score, ('#f1f5f9', '#64748b'))


# ──────────────────────────────────────────────────────────────────────────────
class HistoryWindow(QMainWindow):
    """
    本機分析紀錄清單

    Signals:
        back_requested: 使用者按「回到主頁」
        view_requested(dict): 使用者要查看某筆完整結果
    """

    back_requested = pyqtSignal()
    view_requested = pyqtSignal(dict)

    # ── Column indices ────────────────────────────────────────────────────────
    _COL_CHECK  = 0
    _COL_DATE   = 1
    _COL_ASSR   = 2
    _COL_ORG    = 3
    _COL_TASK   = 4
    _COL_FNAME  = 5
    _COL_CREAT  = 6
    _COL_MAX    = 7
    _COL_AVG    = 8
    _COL_BACK   = 9
    _COL_MODE   = 10
    _COL_SPEED  = 11
    _COL_ACT    = 12
    _NUM_COLS   = 13

    def __init__(self):
        super().__init__()
        language_manager.add_observer(self.on_language_changed)
        self.setMinimumSize(1100, 620)
        self.resize(1360, 740)
        self.setStyleSheet(UPLOAD_BG_STYLE)

        self._history = []
        self._init_ui()
        self._retranslate_ui()
        self._load_data()

    # ── Build UI ──────────────────────────────────────────────────────────────
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        outer = QVBoxLayout(central)
        outer.setSpacing(20)
        outer.setContentsMargins(40, 32, 40, 32)

        outer.addWidget(self._build_header())
        outer.addWidget(self._build_table_card())

    # ── Header ────────────────────────────────────────────────────────────────
    def _build_header(self) -> QFrame:
        card = QFrame()
        card.setObjectName('header_card')
        card.setStyleSheet(HEADER_CARD_STYLE)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        row = QHBoxLayout(card)
        row.setContentsMargins(24, 16, 24, 16)
        row.setSpacing(14)

        icon = QLabel('📋')
        icon.setFixedSize(48, 48)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #10b981, stop:1 #059669);
                border-radius: 12px;
                font-size: 22px;
            }
        """)
        row.addWidget(icon)

        text_col = QVBoxLayout()
        self._header_title = QLabel()
        self._header_title.setFont(QFont('Microsoft JhengHei', 16, QFont.Weight.Bold))
        self._header_title.setStyleSheet('color: #0f172a;')
        text_col.addWidget(self._header_title)

        self._header_sub = QLabel()
        self._header_sub.setStyleSheet('color: #64748b; font-size: 13px;')
        text_col.addWidget(self._header_sub)

        row.addLayout(text_col)
        row.addStretch()

        self._batch_del_btn = QPushButton()
        self._batch_del_btn.setStyleSheet(RED_BTN_STYLE)
        self._batch_del_btn.setMinimumWidth(120)
        self._batch_del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._batch_del_btn.setEnabled(False)
        self._batch_del_btn.clicked.connect(self._on_batch_delete)
        row.addWidget(self._batch_del_btn)

        self._refresh_btn = QPushButton()
        self._refresh_btn.setStyleSheet(BLUE_BTN_STYLE)
        self._refresh_btn.setFixedWidth(110)
        self._refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_btn.clicked.connect(self._load_data)
        row.addWidget(self._refresh_btn)

        self._back_btn = QPushButton()
        self._back_btn.setStyleSheet(BACK_BTN_STYLE)
        self._back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._back_btn.clicked.connect(self.back_requested.emit)
        row.addWidget(self._back_btn)

        return card

    # ── Table card ────────────────────────────────────────────────────────────
    def _build_table_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName('content_card')
        card.setStyleSheet(CONTENT_CARD_STYLE)

        col = QVBoxLayout(card)
        col.setContentsMargins(20, 20, 20, 20)
        col.setSpacing(12)

        self._empty_lbl = QLabel()
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_lbl.setStyleSheet(
            'color: #94a3b8; font-size: 15px; padding: 40px;'
        )
        self._empty_lbl.setVisible(False)
        col.addWidget(self._empty_lbl)

        self._table = QTableWidget(0, self._NUM_COLS)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setSortingEnabled(False)
        self._table.setStyleSheet(
            'QTableWidget { alternate-background-color: #f8fafc; }'
        )

        hdr = self._table.horizontalHeader()
        hdr.setSectionsClickable(True)
        hdr.sectionClicked.connect(self._on_header_section_clicked)
        hdr.setStretchLastSection(False)

        # ☑ column: fixed narrow; filename: stretch to fill; actions: fixed wide
        for i in range(self._NUM_COLS):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(self._COL_CHECK, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(self._COL_FNAME, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(self._COL_ACT,   QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(self._COL_CHECK, 36)
        self._table.setColumnWidth(self._COL_ACT,   268)

        self._table.itemChanged.connect(self._on_item_changed)
        col.addWidget(self._table)
        return card

    # ── Language ──────────────────────────────────────────────────────────────
    def on_language_changed(self, _lang_code):
        self._retranslate_ui()
        self._refresh_table()

    def _retranslate_ui(self):
        self.setWindowTitle(t('history_window_title'))
        self._header_title.setText(t('history_header_title'))
        self._header_sub.setText(t('history_header_sub'))
        self._back_btn.setText(t('history_back_btn'))
        self._refresh_btn.setText(t('history_refresh_btn'))
        self._empty_lbl.setText(t('history_empty'))
        self._update_table_headers()
        self._update_batch_delete_btn()

    def _update_table_headers(self):
        n_rows = self._table.rowCount()
        n_sel  = self._selected_count()
        self._table.setHorizontalHeaderLabels([
            '',   # col 0 overridden below with native checkbox item
            t('history_col_date'),
            t('history_col_assessor'),
            t('history_col_org'),
            t('history_col_task'),
            t('history_col_filename'),
            t('history_col_created'),
            t('history_col_max'),
            t('history_col_avg'),
            t('history_col_backend'),
            t('history_col_analysis_mode'),
            t('history_col_speed_anomaly'),
            t('history_col_actions'),
        ])
        # Use Qt native checkbox rendering in header (avoids font-dependent ☐/☑)
        check_hdr = QTableWidgetItem()
        if n_rows > 0 and n_sel == n_rows:
            check_hdr.setCheckState(Qt.CheckState.Checked)
        elif n_sel > 0:
            check_hdr.setCheckState(Qt.CheckState.PartiallyChecked)
        else:
            check_hdr.setCheckState(Qt.CheckState.Unchecked)
        self._table.setHorizontalHeaderItem(self._COL_CHECK, check_hdr)

    # ── Load / render data ────────────────────────────────────────────────────
    def _load_data(self):
        self._history = load_history()
        self._refresh_table()

    def _refresh_table(self):
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        self._table.blockSignals(False)

        if not self._history:
            self._empty_lbl.setVisible(True)
            self._table.setVisible(False)
            self._update_batch_delete_btn()
            self._update_table_headers()
            return

        self._empty_lbl.setVisible(False)
        self._table.setVisible(True)

        self._table.blockSignals(True)
        for row_idx, rec in enumerate(self._history):
            self._table.insertRow(row_idx)
            self._fill_row(row_idx, rec)
        self._table.blockSignals(False)

        self._update_batch_delete_btn()
        self._update_table_headers()

    def _fill_row(self, row_idx: int, rec: dict):
        meta  = rec.get('meta', {})
        stats = rec.get('stats', {})
        center = Qt.AlignmentFlag.AlignCenter
        left   = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

        def _cell(text, align=left):
            item = QTableWidgetItem(str(text))
            item.setTextAlignment(align)
            return item

        # Col 0: checkbox
        check_item = QTableWidgetItem()
        check_item.setCheckState(Qt.CheckState.Unchecked)
        check_item.setTextAlignment(center)
        self._table.setItem(row_idx, self._COL_CHECK, check_item)

        self._table.setItem(row_idx, self._COL_DATE,  _cell(meta.get('survey_date', '—')))
        self._table.setItem(row_idx, self._COL_ASSR,  _cell(meta.get('assessor', '—')))
        self._table.setItem(row_idx, self._COL_ORG,   _cell(meta.get('organization', '—')))
        self._table.setItem(row_idx, self._COL_TASK,  _cell(meta.get('task_name', '—')))
        self._table.setItem(row_idx, self._COL_FNAME, _cell(rec.get('original_filename', '—')))

        raw_ts = rec.get('created_at', '')
        try:
            ts_str = datetime.fromisoformat(raw_ts).strftime('%Y/%m/%d %H:%M')
        except Exception:
            ts_str = raw_ts[:16] if raw_ts else '—'
        self._table.setItem(row_idx, self._COL_CREAT, _cell(ts_str))

        # Max score with colour
        max_s = stats.get('max_score')
        max_item = _cell(str(max_s) if max_s is not None else '—', center)
        if max_s is not None:
            bg, fg = _score_badge(int(max_s))
            max_item.setBackground(QColor(bg))
            max_item.setForeground(QColor(fg))
        self._table.setItem(row_idx, self._COL_MAX, max_item)

        avg_s = stats.get('avg_score')
        self._table.setItem(
            row_idx, self._COL_AVG,
            _cell(f'{avg_s:.1f}' if avg_s is not None else '—', center)
        )

        backend = str(rec.get('backend_mode', '—')).upper()
        self._table.setItem(row_idx, self._COL_BACK, _cell(backend, center))

        # Analysis mode with colour
        mode = str(rec.get('analysis_mode', '—')).upper()
        mode_item = _cell(mode, center)
        mode_item.setForeground(QColor('#7c3aed' if mode == '3D' else '#0369a1'))
        self._table.setItem(row_idx, self._COL_MODE, mode_item)

        speed_text = t('common_on') if rec.get('speed_anomaly_enabled', True) else t('common_off')
        self._table.setItem(row_idx, self._COL_SPEED, _cell(speed_text, center))

        # Action buttons
        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(6, 4, 6, 4)
        btn_layout.setSpacing(6)

        for label, style, slot in (
            (t('history_view_btn'),        BLUE_BTN_STYLE,    lambda _, r=rec: self._on_view(r)),
            (t('history_export_btn_text'), EMERALD_BTN_STYLE, lambda _, r=rec: self._on_export(r)),
            (t('history_delete_btn'),      RED_BTN_STYLE,     lambda _, r=rec: self._on_delete(r)),
        ):
            btn = QPushButton(label)
            btn.setMinimumWidth(72)
            btn.setFixedHeight(30)
            btn.setStyleSheet(style)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(slot)
            btn_layout.addWidget(btn)

        self._table.setCellWidget(row_idx, self._COL_ACT, btn_widget)
        self._table.setRowHeight(row_idx, 46)

    # ── Checkbox helpers ──────────────────────────────────────────────────────
    def _selected_count(self) -> int:
        return sum(
            1 for r in range(self._table.rowCount())
            if (item := self._table.item(r, self._COL_CHECK)) is not None
            and item.checkState() == Qt.CheckState.Checked
        )

    def _on_header_section_clicked(self, logical_idx: int):
        if logical_idx != self._COL_CHECK:
            return
        n = self._table.rowCount()
        if n == 0:
            return
        all_checked = self._selected_count() == n
        new_state = Qt.CheckState.Unchecked if all_checked else Qt.CheckState.Checked
        self._table.blockSignals(True)
        for r in range(n):
            item = self._table.item(r, self._COL_CHECK)
            if item:
                item.setCheckState(new_state)
        self._table.blockSignals(False)
        self._update_batch_delete_btn()
        self._update_table_headers()

    def _on_item_changed(self, item: QTableWidgetItem):
        if item.column() != self._COL_CHECK:
            return
        self._update_batch_delete_btn()
        self._update_table_headers()

    def _update_batch_delete_btn(self):
        n = self._selected_count()
        label = t('history_batch_delete_btn')
        self._batch_del_btn.setText(f'{label} ({n})' if n > 0 else label)
        self._batch_del_btn.setEnabled(n > 0)

    # ── Actions ───────────────────────────────────────────────────────────────
    def _on_view(self, rec: dict):
        self.view_requested.emit(rec)

    def _on_export(self, rec: dict):
        default_name = (
            f"rula_{rec.get('meta', {}).get('task_name', 'analysis')}"
            f"_{rec.get('meta', {}).get('survey_date', '')}.csv"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, t('history_export_dialog_title'), default_name,
            'CSV Files (*.csv);;All Files (*)'
        )
        if path:
            try:
                export_csv(rec, path)
                QMessageBox.information(
                    self, t('history_export_success_title'),
                    t('history_export_success_msg').format(path)
                )
            except Exception as e:
                QMessageBox.critical(self, t('history_export_fail_title'), str(e))

    def _on_delete(self, rec: dict):
        reply = QMessageBox.question(
            self, t('history_delete_confirm_title'),
            t('history_delete_confirm_msg').format(rec.get('original_filename', '')),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            json_path = rec.get('_json_path', '')
            if json_path and os.path.exists(json_path):
                try:
                    os.remove(json_path)
                except Exception as e:
                    QMessageBox.warning(self, t('history_delete_fail_title'), str(e))
                    return
            self._load_data()

    def _on_batch_delete(self):
        selected_rows = [
            r for r in range(self._table.rowCount())
            if (item := self._table.item(r, self._COL_CHECK)) is not None
            and item.checkState() == Qt.CheckState.Checked
        ]
        if not selected_rows:
            return

        selected_recs = [self._history[r] for r in selected_rows]
        reply = QMessageBox.warning(
            self,
            t('history_batch_delete_confirm_title'),
            t('history_batch_delete_confirm_msg').format(count=len(selected_recs)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        failed = []
        for rec in selected_recs:
            json_path = rec.get('_json_path', '')
            if json_path and os.path.exists(json_path):
                try:
                    os.remove(json_path)
                except Exception as e:
                    failed.append(f'{os.path.basename(json_path)}: {e}')

        if failed:
            QMessageBox.warning(self, t('history_delete_fail_title'), '\n'.join(failed))
        self._load_data()
