"""
UI 共用元件模組。

提供即時視窗與結果輸出會用到的可重用元件與工具，包含：
- ScorePanel：RULA 分數與角度顯示面板
- CoordinatesPanel：關鍵點座標顯示面板
- FrameRenderer：影像顯示縮放工具
- SnapshotManager：快照與文字紀錄儲存
- ChartGenerator：分數統計圖產生
"""

from PyQt6.QtWidgets import QVBoxLayout, QLabel, QGroupBox, QGridLayout, QTextEdit
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
from datetime import datetime
import cv2
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非 GUI 後端
import matplotlib.pyplot as plt

from .styles import SCORE_PANEL_STYLE, COORDINATES_PANEL_STYLE
from .language import language_manager, t


class ScorePanel(QGroupBox):
    """RULA Score Display Panel"""
    
    def __init__(self, title="RULA Score", parent=None):
        """
        創建分數顯示面板

        Args:
            title: 面板標題
            parent: 父級 widget
        """
        super().__init__(title, parent)
        self.setStyleSheet(SCORE_PANEL_STYLE)
        
        # 注冊語言觀察者
        self.lang = language_manager
        self.lang.add_observer(self.on_language_changed)
        
        self.layout = QGridLayout()
        self.layout.setSpacing(8)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        # 角度標籤 - 使用語言系統
        row = 0
        labels_data = [
            ("upper_arm", "label_upper_arm_angle"),
            ("lower_arm", "label_lower_arm_angle"),
            ("wrist", "label_wrist_angle"),
            ("neck", "label_neck_angle"),
            ("trunk", "label_trunk_angle"),
        ]
        
        self.angle_labels = {}
        self.part_score_labels = {}
        self.angle_text_labels = {}  # 保存文本標籤以便更新語言
        self.score_text_labels = {}  # 保存分數文本標籤
        
        for key, text_key in labels_data:
            # 角度標籤
            label = QLabel(t(text_key))
            label.setStyleSheet("font-size: 13px; color: #ffffff;")
            value = QLabel("--")
            value.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff;")
            self.layout.addWidget(label, row, 0)
            self.layout.addWidget(value, row, 1)
            self.angle_labels[key] = value
            self.angle_text_labels[key] = (label, text_key)  # 保存標籤和翻譯鍵
            
            # 部位分數標籤
            score_label = QLabel(t('label_score'))
            score_label.setStyleSheet("font-size: 12px; color: #95a5a6;")
            score_value = QLabel("--")
            score_value.setStyleSheet("font-size: 13px; font-weight: bold; color: #f39c12;")
            self.layout.addWidget(score_label, row, 2)
            self.layout.addWidget(score_value, row, 3)
            self.part_score_labels[key] = score_value
            self.score_text_labels[key] = score_label  # 保存標籤
            
            row += 1
        
        # 分隔線
        separator = QLabel()
        separator.setStyleSheet("""
            border: none;
            border-top: 2px solid rgba(52, 152, 219, 0.3);
            margin: 10px 0;
        """)
        separator.setMaximumHeight(10)
        self.layout.addWidget(separator, row, 0, 1, 2)
        row += 1
        
        # RULA 分數 - 使用語言系統
        score_data = [
            ("table_a", "label_table_a_score"),
            ("table_b", "label_table_b_score"),
            ("table_c", "label_table_c_score"),
        ]
        
        self.score_labels = {}
        self.table_text_labels = {}  # 保存 Table 標籤
        for key, text_key in score_data:
            label = QLabel(t(text_key))
            label.setStyleSheet("font-size: 13px; color: #3498db; font-weight: bold;")
            value = QLabel("--")
            value.setStyleSheet("font-size: 14px; font-weight: bold; color: #ecf0f1;")
            self.layout.addWidget(label, row, 0)
            self.layout.addWidget(value, row, 1)
            self.score_labels[key] = value
            self.table_text_labels[key] = (label, text_key)  # 保存標籤和翻譯鍵
            row += 1
        
        self.setLayout(self.layout)
    
    def on_language_changed(self, lang_code):
        """語言改變時更新標籤文本"""
        # 更新角度標籤
        for key, (label, text_key) in self.angle_text_labels.items():
            label.setText(t(text_key))
        
        # 更新分數標籤
        for label in self.score_text_labels.values():
            label.setText(t('label_score'))
        
        # 更新 Table 標籤
        for key, (label, text_key) in self.table_text_labels.items():
            label.setText(t(text_key))
    
    def update_score_panel(self, rula_data):
        """
        更新分數面板
        
        Args:
            rula_data: RULA 計算結果字典
        """
        if not rula_data:
            return
            
        # 更新角度
        angle_keys = {
            'upper_arm': 'upper_arm_angle',
            'lower_arm': 'lower_arm_angle',
            'wrist': 'wrist_angle',
            'neck': 'neck_angle',
            'trunk': 'trunk_angle',
        }
        
        # 部位分數對應鍵
        score_keys = {
            'upper_arm': 'upper_arm_score',
            'lower_arm': 'lower_arm_score',
            'wrist': 'wrist_score',
            'neck': 'neck_score',
            'trunk': 'trunk_score',
        }
        
        for key, data_key in angle_keys.items():
            # 更新角度
            value = rula_data.get(data_key, 'NULL')
            if value != 'NULL':
                self.angle_labels[key].setText(f"{value}°")
            else:
                self.angle_labels[key].setText("--")
            
            # 更新部位分數
            score_value = rula_data.get(score_keys[key], '--')
            self.part_score_labels[key].setText(str(score_value))
        
        # 更新分數
        table_a = rula_data.get('wrist_and_arm_score', '--')
        table_b = rula_data.get('neck_trunk_leg_score', '--')
        table_c = rula_data.get('score', '--')
        
        self.score_labels['table_a'].setText(str(table_a))
        self.score_labels['table_b'].setText(str(table_b))
        self.score_labels['table_c'].setText(str(table_c))
    
    def reset_panel(self):
        """重置面板所有值"""
        for key in self.angle_labels:
            self.angle_labels[key].setText("--")
            self.part_score_labels[key].setText("--")
        
        for key in self.score_labels:
            self.score_labels[key].setText("--")
    
    def get_score(self):
        """取得當前面板的總分"""
        try:
            score_text = self.score_labels['table_c'].text()
            return score_text if score_text != '--' else 'N/A'
        except:
            return 'N/A'


class CoordinatesPanel(QGroupBox):
    """Keypoint Coordinates Display Panel"""
    
    def __init__(self, title="Keypoint Coordinates", parent=None):
        """
        創建坐標顯示面板
        
        Args:
            title: 面板標題
            parent: 父級 widget
        """
        super().__init__(title, parent)
        self.setStyleSheet(COORDINATES_PANEL_STYLE)
        
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        # 使用 QTextEdit 顯示坐標信息
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setMinimumHeight(600)
        self.text_edit.setText("等待骨架數據...")
        
        self.layout.addWidget(self.text_edit)
        self.setLayout(self.layout)
    
    def update_coordinates(self, landmarks):
        """
        更新坐標顯示面板 - 只顯示用於 RULA 角度計算的關鍵點
        
        Args:
            landmarks: 骨架關鍵點列表 [[x, y, z, visibility], ...]
        """
        # 只顯示用於 RULA 計算的關鍵點
        key_points = {
            11: "Left Shoulder",
            12: "Right Shoulder",
            13: "Left Elbow",
            14: "Right Elbow",
            15: "Left Wrist",
            16: "Right Wrist",
            23: "Left Hip",
            24: "Right Hip",
        }
        
        # 構建顯示文本
        text_lines = []
        
        for idx, name in key_points.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                x, y, z, vis = lm[0], lm[1], lm[2], lm[3]
                text_lines.append(f"【{idx:2d}】 {name:20s}")
                text_lines.append(f"      X: {x:7.4f}  Y: {y:7.4f}  Z: {z:7.4f}")
                text_lines.append(f"      Visibility: {vis:.4f}")
                text_lines.append("")
        
        display_text = "\n".join(text_lines)
        
        # 更新 QTextEdit
        self.text_edit.setPlainText(display_text)
    
    def reset_panel(self):
        """重置面板"""
        self.text_edit.setPlainText("等待骨架數據...")
    
    def get_text(self):
        """取得當前顯示的坐標文本"""
        return self.text_edit.toPlainText()


class FrameRenderer:
    """Handle frame display and rendering"""
    
    @staticmethod
    def display_frame(label, frame):
        """
        顯示影像幀到 QLabel
        
        Args:
            label: QLabel widget 用於顯示圖像
            frame: RGB 格式的影像 (numpy array)
        """
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # 縮放以適應標籤大小，同時保持寬高比
        scaled_pixmap = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)
    
    @staticmethod
    def draw_scores_on_frame(frame, left_score, right_score):
        """
        在影像上繪製分數資訊
        
        Args:
            frame: RGB 格式的影像 (numpy array，會被修改)
            left_score: 左側分數
            right_score: 右側分數
        """
        # 設置文字參數
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        line_height = 40
        y_start = 50
        
        # 計算背景區域大小
        bg_width = 550
        bg_height = 180
        
        # 創建半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (15 + bg_width, 15 + bg_height), (44, 62, 80), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # 標題
        cv2.putText(frame, "RULA Evaluation Results", (30, y_start), 
                   font, 1.1, (52, 152, 219), thickness + 1)
        
        # 繪製左側分數
        y = y_start + line_height + 5
        cv2.putText(frame, f"Left Side - Score: {left_score}", 
                   (30, y), font, font_scale, (46, 204, 113), thickness)
        
        # 繪製右側分數
        y += line_height
        cv2.putText(frame, f"Right Side - Score: {right_score}", 
                   (30, y), font, font_scale, (46, 204, 113), thickness)
        
        # 繪製時間戳
        y += line_height - 5
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Time: {timestamp}", (30, y), 
                   font, 0.7, (189, 195, 199), 2)


class SnapshotManager:
    """Handle snapshot saving functionality"""
    
    SNAPSHOT_DIR = "rula_snapshots"
    RECORDING_DIR = "rula_records"
    
    @staticmethod
    def ensure_directory_exists(directory=None):
        """確保保存目錄存在
        
        Args:
            directory: 指定目錄，若為 None 則使用 SNAPSHOT_DIR
        """
        target_dir = directory if directory else SnapshotManager.SNAPSHOT_DIR
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
    
    @staticmethod
    def save_rula_snapshot(frame, left_panel, right_panel, parent_window=None):
        """
        保存 RULA 評估的快照（圖片+文本）
        
        Args:
            frame: RGB 格式的影像 (numpy array)
            left_panel: 左側 ScorePanel
            right_panel: 右側 ScorePanel
            parent_window: 父級窗口（用於消息框）
        
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            SnapshotManager.ensure_directory_exists()
            
            # 生成文件名（使用時間戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(SnapshotManager.SNAPSHOT_DIR, f"rula_{timestamp}.png")
            txt_path = os.path.join(SnapshotManager.SNAPSHOT_DIR, f"rula_{timestamp}.txt")
            
            # 複製當前影像用於保存
            frame_to_save = frame.copy()
            
            # 在影像上繪製分數資訊
            left_score = left_panel.get_score()
            right_score = right_panel.get_score()
            FrameRenderer.draw_scores_on_frame(frame_to_save, left_score, right_score)
            
            # 保存影像（OpenCV 使用 BGR 格式）
            cv2.imwrite(image_path, cv2.cvtColor(frame_to_save, cv2.COLOR_RGB2BGR))
            
            # 保存文本資訊
            SnapshotManager._save_rula_scores_to_text(txt_path, left_panel, right_panel)
            
            info_text = f"圖片: {image_path}\n文本: {txt_path}"
            return True, info_text
            
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def save_coordinates_snapshot(frame, landmarks, parent_window=None):
        """
        保存坐標顯示的快照（圖片+坐標文本）
        
        Args:
            frame: RGB 格式的影像 (numpy array)
            landmarks: 骨架關鍵點列表
            parent_window: 父級窗口（用於消息框）
        
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            SnapshotManager.ensure_directory_exists()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(SnapshotManager.SNAPSHOT_DIR, f"coordinates_{timestamp}.png")
            txt_path = os.path.join(SnapshotManager.SNAPSHOT_DIR, f"coordinates_{timestamp}.txt")
            
            # 保存影像
            cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # 保存坐標文本
            SnapshotManager._save_coordinates_to_text(txt_path, landmarks)
            
            info_text = f"圖片: {image_path}\n文本: {txt_path}"
            return True, info_text
            
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def _save_rula_scores_to_text(filepath, left_panel, right_panel):
        """
        保存分數到文本文件
        
        Args:
            filepath: 文件路徑
            left_panel: 左側 ScorePanel
            right_panel: 右側 ScorePanel
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("RULA 即時評估結果\n")
            f.write("=" * 50 + "\n")
            f.write(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            
            # 保存左側數據
            f.write("左側身體評估:\n")
            f.write("-" * 50 + "\n")
            SnapshotManager._write_panel_scores(f, left_panel)
            f.write("\n")
            
            # 保存右側數據
            f.write("右側身體評估:\n")
            f.write("-" * 50 + "\n")
            SnapshotManager._write_panel_scores(f, right_panel)
    
    @staticmethod
    def _write_panel_scores(file, panel):
        """
        將面板分數寫入文件
        
        Args:
            file: 文件對象
            panel: ScorePanel
        """
        # 寫入角度
        angle_names = {
            'upper_arm': '上臂角度',
            'lower_arm': '前臂角度',
            'wrist': '手腕角度',
            'neck': '頸部角度',
            'trunk': '軀幹角度',
        }
        
        for key, name in angle_names.items():
            angle = panel.angle_labels[key].text()
            score = panel.part_score_labels[key].text()
            file.write(f"  {name}: {angle} (分數: {score})\n")
        
        file.write("\n")
        
        # 寫入總分
        table_a = panel.score_labels['table_a'].text()
        table_b = panel.score_labels['table_b'].text()
        table_c = panel.score_labels['table_c'].text()
        
        file.write(f"  Table A 分數: {table_a}\n")
        file.write(f"  Table B 分數: {table_b}\n")
        file.write(f"  Table C 分數 (總分): {table_c}\n")
    
    @staticmethod
    def _save_coordinates_to_text(filepath, landmarks):
        """
        保存坐標到文本文件
        
        Args:
            filepath: 文件路徑
            landmarks: 骨架關鍵點列表
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("關鍵點坐標數據\n")
            f.write("=" * 70 + "\n")
            f.write(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            
            # 只顯示用於 RULA 計算的關鍵點
            key_points = {
                0: "Nose",
                11: "Left Shoulder",
                12: "Right Shoulder",
                13: "Left Elbow",
                14: "Right Elbow",
                15: "Left Wrist",
                16: "Right Wrist",
                23: "Left Hip",
                24: "Right Hip",
            }
            
            for idx, name in key_points.items():
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    x, y, z, vis = lm[0], lm[1], lm[2], lm[3]
                    f.write(f"【{idx:2d}】 {name:20s}\n")
                    f.write(f"      X: {x:7.4f}  Y: {y:7.4f}  Z: {z:7.4f}\n")
                    f.write(f"      Visibility: {vis:.4f}\n\n")


class ChartGenerator:
    """Handle RULA chart generation"""
    
    @staticmethod
    def generate_rula_charts(rula_records, recording_start_time, filepath):
        """
        生成 RULA 分數圖表（折線圖+圓餅圖）
        
        Args:
            rula_records: RULA 記錄列表
            recording_start_time: 錄影開始時間
            filepath: 輸出圖表文件路徑
        """
        try:
            if not rula_records:
                return
            
            # 提取數據，使用 NaN 表示無效數據
            timestamps = [r['timestamp'] for r in rula_records]
            left_scores = []
            right_scores = []
            
            for r in rula_records:
                # 左側分數處理 - 無效數據使用 NaN
                left_score = r['left'].get('score', 0)
                if left_score in ['--', 'NULL', None, '']:
                    left_scores.append(np.nan)
                else:
                    try:
                        left_scores.append(int(left_score))
                    except (ValueError, TypeError):
                        left_scores.append(np.nan)
                
                # 右側分數處理 - 無效數據使用 NaN
                right_score = r['right'].get('score', 0)
                if right_score in ['--', 'NULL', None, '']:
                    right_scores.append(np.nan)
                else:
                    try:
                        right_scores.append(int(right_score))
                    except (ValueError, TypeError):
                        right_scores.append(np.nan)
            
            # 設置中文字體（嘗試使用系統字體）
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 創建圖表 (2行2列)
            fig = plt.figure(figsize=(16, 10))
            
            # === 1. 左側折線圖 ===
            ax1 = plt.subplot(2, 2, 1)
            ax1.plot(timestamps, left_scores, 'b-', linewidth=2, label=t('chart_legend_left'), marker='o', markersize=4)
            ax1.set_xlabel(t('chart_time'), fontsize=12)
            ax1.set_ylabel(t('chart_score_label'), fontsize=12)
            ax1.set_title(t('chart_left_line'), fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=10)
            # 設置 Y 軸範圍，使用 nanmax 忽略 NaN 值
            max_left = np.nanmax(left_scores) if not np.all(np.isnan(left_scores)) else 0
            max_right = np.nanmax(right_scores) if not np.all(np.isnan(right_scores)) else 0
            max_score = max(max_left, max_right)
            ax1.set_ylim(0, max_score + 1 if max_score > 0 else 8)
            
            # === 2. 右側折線圖 ===
            ax2 = plt.subplot(2, 2, 2)
            ax2.plot(timestamps, right_scores, 'r-', linewidth=2, label=t('chart_legend_right'), marker='s', markersize=4)
            ax2.set_xlabel(t('chart_time'), fontsize=12)
            ax2.set_ylabel(t('chart_score_label'), fontsize=12)
            ax2.set_title(t('chart_right_line'), fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            # 設置 Y 軸範圍，與左側一致
            ax2.set_ylim(0, max_score + 1 if max_score > 0 else 8)
            
            # === 3. 左側圓餅圖 ===
            ax3 = plt.subplot(2, 2, 3)
            left_counts = {}
            for score in left_scores:
                # 只統計有效的非 NaN 分數
                if not np.isnan(score) and score > 0:
                    left_counts[score] = left_counts.get(score, 0) + 1
            
            if left_counts:
                labels = [t('chart_score_prefix').format(int(score)) for score in sorted(left_counts.keys())]
                sizes = [left_counts[score] for score in sorted(left_counts.keys())]
                colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(sizes)))
                
                wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                                     startangle=90, textprops={'fontsize': 10})
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                ax3.set_title(t('chart_left_pie'), fontsize=14, fontweight='bold')
            else:
                ax3.text(0.5, 0.5, t('chart_no_data'), ha='center', va='center', fontsize=14)
                ax3.set_title(t('chart_left_pie'), fontsize=14, fontweight='bold')
            
            # === 4. 右側圓餅圖 ===
            ax4 = plt.subplot(2, 2, 4)
            right_counts = {}
            for score in right_scores:
                # 只統計有效的非 NaN 分數
                if not np.isnan(score) and score > 0:
                    right_counts[score] = right_counts.get(score, 0) + 1
            
            if right_counts:
                labels = [t('chart_score_prefix').format(int(score)) for score in sorted(right_counts.keys())]
                sizes = [right_counts[score] for score in sorted(right_counts.keys())]
                colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(sizes)))
                
                wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                                     startangle=90, textprops={'fontsize': 10})
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                ax4.set_title(t('chart_right_pie'), fontsize=14, fontweight='bold')
            else:
                ax4.text(0.5, 0.5, t('chart_no_data'), ha='center', va='center', fontsize=14)
                ax4.set_title(t('chart_right_pie'), fontsize=14, fontweight='bold')
            
            # 添加總標題
            duration = timestamps[-1] if timestamps else 0
            fig.suptitle(t('chart_title').format(duration, len(rula_records)),
                        fontsize=16, fontweight='bold', y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"圖表已保存: {filepath}")
            
        except Exception as e:
            print(f"生成圖表失敗: {e}")
            import traceback
            traceback.print_exc()


