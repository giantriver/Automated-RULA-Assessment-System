"""
RULA 評估系統 - 主程式
啟動首頁，讓使用者選擇「即時分析」或「上傳影片分析」
"""

import sys
import os
import argparse
# 強迫 Python 把 main.py 所在的那個資料夾當作搜尋的第一順位
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from .ui.launcher_window import LauncherWindow
from .ui.realtime_window import MainWindow
from .ui.upload_window   import UploadWindow
from .ui.history_window  import HistoryWindow
from .ui.result_window   import ResultWindow
from .core import config


class AppController:
    """
    中央控制器：管理各視窗的切換。

    視窗流程：
        LauncherWindow
          ├─ 即時分析   → MainWindow（既有即時評估）
          ├─ 上傳影片   → UploadWindow（填表 + 送出分析）
          │                └─ analysis_done → ResultWindow（獨立結果頁）
          └─ 分析紀錄   → HistoryWindow（本機歷史清單）
                           └─ view_requested → ResultWindow（獨立結果頁）
    """

    def __init__(self):
        self._launcher : LauncherWindow | None = None # type annotation
        self._realtime : MainWindow     | None = None
        self._upload   : UploadWindow   | None = None
        self._history  : HistoryWindow  | None = None
        # result windows are non-modal; keep a list so they aren't GC'd
        self._result_windows: list[ResultWindow] = []

    # ── Boot ──────────────────────────────────────────────────────────────────
    def start(self):
        self._show_launcher()

    # ── Launcher ──────────────────────────────────────────────────────────────
    def _show_launcher(self):
        if self._launcher is None:
            self._launcher = LauncherWindow()
            self._launcher.realtime_requested.connect(self._show_realtime)
            self._launcher.upload_requested.connect(self._show_upload)
            self._launcher.history_requested.connect(self._show_history)
        self._launcher.showMaximized()

    # ── Real-time analysis ───────────────────────────────────────────────────
    def _show_realtime(self):
        self._launcher.hide()
        if self._realtime is None:
            self._realtime = MainWindow()
            self._realtime.back_requested.connect(self._on_realtime_back)
            self._realtime.destroyed.connect(self._on_realtime_closed)
        self._realtime.showMaximized()
        self._realtime.activateWindow()

    def _on_realtime_back(self):
        if self._realtime:
            self._realtime.close()
            self._realtime = None
        self._show_launcher()

    def _on_realtime_closed(self):
        self._realtime = None
        self._show_launcher()

    # ── Upload / offline analysis ────────────────────────────────────────────
    def _show_upload(self):
        self._launcher.hide()
        if self._upload is None:
            self._upload = UploadWindow()
            self._upload.back_requested.connect(self._on_upload_back)
            self._upload.analysis_done.connect(self._open_result_window)
            self._upload.destroyed.connect(lambda: setattr(self, '_upload', None))
        self._upload.showMaximized()
        self._upload.activateWindow()

    def _on_upload_back(self):
        if self._upload:
            self._upload.close()
            self._upload = None
        self._show_launcher()

    # ── Result window (shared by upload + history) ────────────────────────────
    def _open_result_window(self, results: dict):
        win = ResultWindow(results)
        win.back_requested.connect(lambda w=win: self._close_result_window(w))
        self._result_windows.append(win)
        win.showMaximized()
        win.activateWindow()

    def _close_result_window(self, win: ResultWindow):
        try:
            self._result_windows.remove(win)
        except ValueError:
            pass

    # ── History ───────────────────────────────────────────────────────────────
    def _show_history(self):
        self._launcher.hide()
        if self._history is None:
            self._history = HistoryWindow()
            self._history.back_requested.connect(self._on_history_back)
            self._history.view_requested.connect(self._open_result_window)
            self._history.destroyed.connect(lambda: setattr(self, '_history', None))
        self._history.showMaximized()
        self._history.activateWindow()

    def _on_history_back(self):
        if self._history:
            self._history.close()
            self._history = None
        self._show_launcher()


# ──────────────────────────────────────────────────────────────────────────────
def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(description='RULA 評估系統') # python main.py -h
    parser.add_argument(
        '-d', '--display-mode',
        type=str, choices=['RULA', 'COORDINATES'], default=config.DISPLAY_MODE,
        help='即時模式顯示方式: RULA | COORDINATES'
    )
    parser.add_argument(
        '-p', '--pose-backend',
        type=str, choices=['MEDIAPIPE', 'RTMW2D', 'RTMW3D'],
        default=(config.POSE_BACKEND if config.POSE_BACKEND in ('RTMW2D', 'RTMW3D') else 'MEDIAPIPE'),
        help='即時模式姿勢辨識後端'
    )
    parser.add_argument(
        '--realtime', action='store_true',
        help='直接進入即時分析（略過啟動選擇頁）'
    )
    args = parser.parse_args() # 把你在 terminal 打的參數讀進來並轉成 Python 變數

    config.DISPLAY_MODE = args.display_mode
    config.POSE_BACKEND = args.pose_backend

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    if args.realtime:
        window = MainWindow()
        window.showMaximized()
    else:
        controller = AppController()
        controller.start()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
