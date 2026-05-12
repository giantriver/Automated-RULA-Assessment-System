"""
多語系文字管理模組。

提供集中式語言狀態與翻譯字典查詢，透過觀察者機制通知各視窗更新文字。
目前支援英文（en）與繁體中文（zh_TW）。
"""

class LanguageManager:
    """语言管理器"""
    
    # 支持的语言
    LANGUAGES = {
        'en': 'English',
        'zh_TW': '繁體中文'
    }
    
    # 默认语言
    DEFAULT_LANGUAGE = 'en'
    
    def __init__(self):
        self.current_language = self.DEFAULT_LANGUAGE
        self._observers = []
    
    def set_language(self, lang_code):
        """设置当前语言"""
        if lang_code in self.LANGUAGES:
            self.current_language = lang_code
            self._notify_observers()
    
    def get_language(self):
        """获取当前语言代码"""
        return self.current_language
    
    def add_observer(self, callback):
        """添加语言变更观察者"""
        self._observers.append(callback)
    
    def _notify_observers(self):
        """通知所有观察者语言已变更"""
        for callback in self._observers:
            callback(self.current_language)
    
    def get_text(self, key):
        """获取指定键的翻译文本"""
        return TRANSLATIONS.get(key, {}).get(self.current_language, key)
    
    def __call__(self, key):
        """快捷方式获取文本"""
        return self.get_text(key)


# 全局语言管理器实例
language_manager = LanguageManager()


# 翻译字典
TRANSLATIONS = {
    # 窗口标题
    'window_title': {
        'en': 'RULA Real-time Assessment System',
        'zh_TW': 'RULA 即時評估系統'
    },
    'window_title_with_source': {
        'en': 'RULA Real-time Assessment System - Pose Backend: {}',
        'zh_TW': 'RULA 即時評估系統 - 姿勢辨識後端：{}'
    },
    
    # 按钮文本
    'btn_start': {
        'en': 'Start',
        'zh_TW': '開始'
    },
    'btn_stop': {
        'en': 'Stop',
        'zh_TW': '停止'
    },
    'btn_pause': {
        'en': 'Pause',
        'zh_TW': '暫停'
    },
    'btn_resume': {
        'en': 'Resume',
        'zh_TW': '繼續'
    },
    'btn_snapshot': {
        'en': '📷 Snapshot',
        'zh_TW': '📷 快照'
    },
    'btn_record': {
        'en': '⏺ Record',
        'zh_TW': '⏺ 錄影'
    },
    'btn_stop_record': {
        'en': '⏹ Stop',
        'zh_TW': '⏹ 停止'
    },
    'btn_language': {
        'en': '🌐',
        'zh_TW': '🌐'
    },
    
    # 工具提示
    'tooltip_snapshot': {
        'en': 'Save current frame and scores',
        'zh_TW': '保存當前畫面和分數'
    },
    'tooltip_record': {
        'en': 'Start/Stop recording',
        'zh_TW': '開始/停止錄影'
    },
    'tooltip_config': {
        'en': 'RULA parameter settings',
        'zh_TW': 'RULA 參數設定'
    },
    'tooltip_language': {
        'en': 'Switch language / 切換語言',
        'zh_TW': 'Switch language / 切換語言'
    },
    
    # 状态文本
    'status_waiting': {
        'en': 'Waiting to start...',
        'zh_TW': '等待開始...'
    },
    'status_stopped': {
        'en': 'Stopped',
        'zh_TW': '已停止'
    },
    'status_error': {
        'en': 'Error: {}',
        'zh_TW': '錯誤: {}'
    },
    'fps_label': {
        'en': 'FPS: {}',
        'zh_TW': 'FPS: {}'
    },
    'rula_freq_label': {
        'en': 'RULA: {} Hz',
        'zh_TW': 'RULA: {} Hz'
    },
    'tooltip_fps': {
        'en': 'Frames Per Second - Camera capture frame rate\nIndicates how smoothly the video displays',
        'zh_TW': 'FPS (每秒幀數) - 攝影機擷取畫面的速度\n數值越高表示畫面越流暢'
    },
    'tooltip_rula_freq': {
        'en': 'Posture Assessment Frequency\nShows how many times per second your posture is evaluated\nHigher values mean more frequent assessments and more detailed data',
        'zh_TW': '姿勢評估頻率\n顯示系統每秒評估您姿勢的次數\n數值越高代表評估越頻繁、數據越詳細'
    },
    
    # 消息框标题
    'msg_warning': {
        'en': 'Warning',
        'zh_TW': '警告'
    },
    'msg_error': {
        'en': 'Error',
        'zh_TW': '錯誤'
    },
    'msg_success': {
        'en': 'Success',
        'zh_TW': '成功'
    },
    'msg_snapshot_success': {
        'en': 'Snapshot Saved',
        'zh_TW': '保存成功'
    },
    'msg_record_complete': {
        'en': 'Recording Complete',
        'zh_TW': '錄影完成'
    },
    
    # 消息框内容
    'msg_no_frame_snapshot': {
        'en': 'No frame available to save',
        'zh_TW': '沒有可保存的畫面'
    },
    'msg_no_frame_record': {
        'en': 'No frame available to record',
        'zh_TW': '沒有可錄製的畫面'
    },
    'msg_snapshot_saved': {
        'en': 'Files saved successfully!',
        'zh_TW': '文件已成功保存！'
    },
    'msg_snapshot_failed': {
        'en': 'Failed to save snapshot',
        'zh_TW': '保存失敗'
    },
    'msg_generic_error': {
        'en': 'An error occurred',
        'zh_TW': '發生錯誤'
    },
    'msg_record_stop_error': {
        'en': 'Error stopping recording:\n{}',
        'zh_TW': '停止錄影時發生錯誤:\n{}'
    },
    
    # 录影完成信息
    'record_complete_info': {
        'en': 'Recording complete!\n\nVideo: {}\nDuration: {:.1f} sec\nFrames: {}\n\nRULA Records: {}\nChart: {}',
        'zh_TW': '錄影已完成！\n\n影片: {}\n時長: {:.1f} 秒\n幀數: {}\n\nRULA 記錄: {}\n圖表: {}'
    },
    
    # 面板标题
    'panel_left_rula': {
        'en': 'Left RULA Score',
        'zh_TW': '左側 RULA 分數'
    },
    'panel_right_rula': {
        'en': 'Right RULA Score',
        'zh_TW': '右側 RULA 分數'
    },
    'panel_coordinates': {
        'en': 'Joint Coordinates',
        'zh_TW': '關節座標'
    },
    
    # RULA 参数名称
    'upper_arm': {
        'en': 'Upper Arm',
        'zh_TW': '上臂'
    },
    'lower_arm': {
        'en': 'Lower Arm',
        'zh_TW': '前臂'
    },
    'wrist': {
        'en': 'Wrist',
        'zh_TW': '手腕'
    },
    'neck': {
        'en': 'Neck',
        'zh_TW': '頸部'
    },
    'trunk': {
        'en': 'Trunk',
        'zh_TW': '軀幹'
    },
    'angle': {
        'en': 'Angle',
        'zh_TW': '角度'
    },
    'score': {
        'en': 'Score',
        'zh_TW': '分數'
    },
    
    # RULA 角度标签（用于面板显示）
    'label_upper_arm_angle': {
        'en': 'Upper Arm Angle:',
        'zh_TW': '上臂角度:'
    },
    'label_lower_arm_angle': {
        'en': 'Lower Arm Angle:',
        'zh_TW': '前臂角度:'
    },
    'label_wrist_angle': {
        'en': 'Wrist Angle:',
        'zh_TW': '手腕角度:'
    },
    'label_neck_angle': {
        'en': 'Neck Angle:',
        'zh_TW': '頸部角度:'
    },
    'label_trunk_angle': {
        'en': 'Trunk Angle:',
        'zh_TW': '軀幹角度:'
    },
    'label_score': {
        'en': 'Score:',
        'zh_TW': '分數:'
    },
    'label_table_a_score': {
        'en': 'Table A Score:',
        'zh_TW': 'Table A 分數:'
    },
    'label_table_b_score': {
        'en': 'Table B Score:',
        'zh_TW': 'Table B 分數:'
    },
    'label_table_c_score': {
        'en': 'Table C Score:',
        'zh_TW': 'Table C 分數:'
    },
    
    'table_a': {
        'en': 'Table A',
        'zh_TW': 'Table A'
    },
    'table_b': {
        'en': 'Table B',
        'zh_TW': 'Table B'
    },
    'table_c': {
        'en': 'Table C',
        'zh_TW': 'Table C'
    },
    'final_score': {
        'en': 'Final Score',
        'zh_TW': '總分'
    },
    
    # 配置对话框
    'config_title': {
        'en': 'RULA Parameter Settings',
        'zh_TW': 'RULA 預設參數設定'
    },
    'config_subtitle': {
        'en': 'Adjust RULA fixed parameters:',
        'zh_TW': '調整 RULA 固定參數：'
    },
    'config_pose_backend': {
        'en': 'Pose Backend:',
        'zh_TW': '姿勢辨識後端：'
    },
    'config_pose_backend_desc': {
        'en': 'Switch real-time pose estimation backend.',
        'zh_TW': '切換即時姿勢辨識後端。'
    },
    'config_option_backend_mediapipe': {
        'en': 'MediaPipe',
        'zh_TW': 'MediaPipe'
    },
    'config_option_backend_rtmw3d': {
        'en': 'RTM Pose (RTMW3D)',
        'zh_TW': 'RTM Pose（RTMW3D）'
    },
    'config_wrist_twist': {
        'en': 'Wrist Twist:',
        'zh_TW': '手腕扭轉 (wrist_twist):'
    },
    'config_wrist_twist_desc': {
        'en': '1=Neutral, 2=Twisted',
        'zh_TW': '1=中立位置, 2=扭轉'
    },
    'config_legs': {
        'en': 'Leg Posture:',
        'zh_TW': '腿部姿勢 (legs):'
    },
    'config_legs_desc': {
        'en': '1=Legs and feet supported, 2=Not supported',
        'zh_TW': '1=腿和腳得到支撐, 2=沒有支撐'
    },
    'config_muscle_use_a': {
        'en': 'Muscle Use - Table A:',
        'zh_TW': '肌肉使用 - Table A (muscle_use_a):'
    },
    'config_muscle_use_a_desc': {
        'en': '0=None, 1=Static >1min or ≥4 times/min',
        'zh_TW': '0=無, 1=靜態>1分鐘 或 ≥4次/分鐘'
    },
    'config_muscle_use_b': {
        'en': 'Muscle Use - Table B:',
        'zh_TW': '肌肉使用 - Table B (muscle_use_b):'
    },
    'config_muscle_use_b_desc': {
        'en': '0=None, 1=Static >1min or ≥4 times/min',
        'zh_TW': '0=無, 1=靜態>1分鐘 或 ≥4次/分鐘'
    },
    'config_force_load_a': {
        'en': 'Force/Load - Arm:',
        'zh_TW': '負荷力量-手臂 (force_load_a):'
    },
    'config_force_load_a_desc': {
        'en': '0=<2kg, 1=2-10kg, 2=>10kg',
        'zh_TW': '0=<2kg, 1=2-10kg, 2=>10kg'
    },
    'config_force_load_b': {
        'en': 'Force/Load - Body:',
        'zh_TW': '負荷力量-身體 (force_load_b):'
    },
    'config_force_load_b_desc': {
        'en': '0=<2kg, 1=2-10kg, 2=>10kg',
        'zh_TW': '0=<2kg, 1=2-10kg, 2=>10kg'
    },
    'config_option_wrist_1': {
        'en': '1 - Neutral position',
        'zh_TW': '1 - 中立位置'
    },
    'config_option_wrist_2': {
        'en': '2 - Twisted',
        'zh_TW': '2 - 扭轉'
    },
    'config_option_legs_1': {
        'en': '1 - Legs and feet are supported',
        'zh_TW': '1 – 腿和腳得到支撐）'
    },
    'config_option_legs_2': {
        'en': '2 - Legs and feet are not supported ',
        'zh_TW': '2 – 腿和腳沒有支撐'
    },
    'config_option_muscle_0': {
        'en': '0 - No sustained or repetitive muscle activity',
        'zh_TW': '0 – 無持續或重複性動作'
    },
    'config_option_muscle_1': {
        'en': '1 - Static posture >1 min or repetitive movement ≥4 times/min',
        'zh_TW': '1 – 姿勢維持超過 1 分鐘，或動作每分鐘重複 ≥ 4 次'
    },
    'config_option_force_0': {
        'en': '0 - <2kg',
        'zh_TW': '0 - <2kg'
    },
    'config_option_force_1': {
        'en': '1 - 2-10kg',
        'zh_TW': '1 - 2-10kg'
    },
    'config_option_force_2': {
        'en': '2 - >10kg',
        'zh_TW': '2 - >10kg'
    },
    'config_muscle_use': {
        'en': 'Muscle Use Score',
        'zh_TW': '肌肉使用分數'
    },
    'config_force_load': {
        'en': 'Force/Load Score',
        'zh_TW': '力量/負荷分數'
    },
    'config_save': {
        'en': 'Save',
        'zh_TW': '保存'
    },
    'config_cancel': {
        'en': 'Close',
        'zh_TW': '關閉'
    },
    
    # 语言选择对话框
    'lang_dialog_title': {
        'en': 'Language Selection',
        'zh_TW': '語言選擇'
    },
    'lang_dialog_subtitle': {
        'en': 'Select your preferred language:',
        'zh_TW': '選擇您偏好的語言：'
    },
    'lang_english': {
        'en': 'English',
        'zh_TW': 'English'
    },
    'lang_chinese': {
        'en': '繁體中文 (Traditional Chinese)',
        'zh_TW': '繁體中文 (Traditional Chinese)'
    },
    'lang_confirm': {
        'en': 'Confirm',
        'zh_TW': '確認'
    },
    'lang_cancel': {
        'en': 'Cancel',
        'zh_TW': '取消'
    },
    
    # 倒计时
    'countdown': {
        'en': '{}',
        'zh_TW': '{}'
    },
    
    # 图表和报告 
    'chart_left_line': {
        'en': 'Left RULA Score Changes (Gaps = Invalid Data)',
        'zh_TW': '左側 RULA 分數變化 (斷點=無效數據)'
    },
    'chart_right_line': {
        'en': 'Right RULA Score Changes (Gaps = Invalid Data)',
        'zh_TW': '右側 RULA 分數變化 (斷點=無效數據)'
    },
    'chart_left_pie': {
        'en': 'Left RULA Score Distribution',
        'zh_TW': '左側 RULA 分數分布'
    },
    'chart_right_pie': {
        'en': 'Right RULA Score Distribution',
        'zh_TW': '右側 RULA 分數分布'
    },
    'chart_time': {
        'en': 'Time (sec)',
        'zh_TW': '時間 (秒)'
    },
    'chart_score_label': {
        'en': 'RULA Score',
        'zh_TW': 'RULA 分數'
    },
    'chart_score_prefix': {
        'en': 'Score {}',
        'zh_TW': '分數 {}'
    },
    'chart_no_data': {
        'en': 'No Valid Data',
        'zh_TW': '無有效數據'
    },
    'chart_title': {
        'en': 'RULA Assessment Report - Duration: {:.1f}s | Records: {}',
        'zh_TW': 'RULA 評估報告 - 總時長: {:.1f}秒 | 記錄數: {}'
    },
    'chart_legend_left': {
        'en': 'Left RULA',
        'zh_TW': '左側 RULA'
    },
    'chart_legend_right': {
        'en': 'Right RULA',
        'zh_TW': '右側 RULA'
    },
    
    # 文本记录
    'record_title': {
        'en': 'RULA Recording Score Report',
        'zh_TW': 'RULA 錄影分數記錄'
    },
    'record_time': {
        'en': 'Recording Time:',
        'zh_TW': '錄影時間:'
    },
    'record_duration': {
        'en': 'Total Duration:',
        'zh_TW': '總時長:'
    },
    'record_frames': {
        'en': 'Total Frames:',
        'zh_TW': '總幀數:'
    },
    'record_count': {
        'en': 'Record Count:',
        'zh_TW': '記錄數量:'
    },
    'record_rula_calc_setting': {
        'en': 'RULA Calculation Setting',
        'zh_TW': 'RULA 計算設定'
    },
    'record_calc_frequency': {
        'en': 'Calculate once every {} frames (approx. {:.1f} Hz)',
        'zh_TW': '每 {} 幀計算一次 (約 {:.1f} Hz)'
    },
    'record_rula_parameters': {
        'en': 'RULA Fixed Parameters',
        'zh_TW': 'RULA 固定參數'
    },
    'record_wrist_twist': {
        'en': 'Wrist Twist:',
        'zh_TW': '手腕扭轉:'
    },
    'record_legs': {
        'en': 'Legs:',
        'zh_TW': '腿部姿勢:'
    },
    'record_muscle_use_a': {
        'en': 'Muscle Use - Table A:',
        'zh_TW': '肌肉使用 - Table A:'
    },
    'record_muscle_use_b': {
        'en': 'Muscle Use - Table B:',
        'zh_TW': '肌肉使用 - Table B:'
    },
    'record_force_load_a': {
        'en': 'Force/Load - Table A:',
        'zh_TW': '負荷力量 - Table A:'
    },
    'record_force_load_b': {
        'en': 'Force/Load - Table B:',
        'zh_TW': '負荷力量 - Table B:'
    },
    'record_time_prefix': {
        'en': 'Time:',
        'zh_TW': '時間:'
    },
    'record_frame_prefix': {
        'en': 'Frame:',
        'zh_TW': '幀:'
    },
    'record_left_side': {
        'en': '【Left Side】',
        'zh_TW': '【左側】'
    },
    'record_right_side': {
        'en': '【Right Side】',
        'zh_TW': '【右側】'
    },
    'record_statistics': {
        'en': 'Statistics',
        'zh_TW': '統計資訊'
    },
    'record_left_scores': {
        'en': 'Left RULA Scores:',
        'zh_TW': '左側 RULA 分數:'
    },
    'record_right_scores': {
        'en': 'Right RULA Scores:',
        'zh_TW': '右側 RULA 分數:'
    },
    'record_average': {
        'en': 'Average:',
        'zh_TW': '平均:'
    },
    'record_min': {
        'en': 'Minimum:',
        'zh_TW': '最小:'
    },
    'record_max': {
        'en': 'Maximum:',
        'zh_TW': '最大:'
    },
    'record_no_data': {
        'en': 'No Data',
        'zh_TW': '無數據'
    },
    'record_upper_arm': {
        'en': 'Upper Arm Angle:',
        'zh_TW': '上臂角度:'
    },
    'record_lower_arm': {
        'en': 'Lower Arm Angle:',
        'zh_TW': '前臂角度:'
    },
    'record_wrist': {
        'en': 'Wrist Angle:',
        'zh_TW': '手腕角度:'
    },
    'record_neck': {
        'en': 'Neck Angle:',
        'zh_TW': '頸部角度:'
    },
    'record_trunk': {
        'en': 'Trunk Angle:',
        'zh_TW': '軀幹角度:'
    },
    'record_score': {
        'en': 'Score:',
        'zh_TW': '分數:'
    },
    'record_seconds': {
        'en': 'seconds',
        'zh_TW': '秒'
    },
    
    # 源类型
    'source_mediapipe': {
        'en': 'MediaPipe',
        'zh_TW': 'MediaPipe'
    },
    'source_rtmw3d_webcam': {
        'en': 'RTMW3D',
        'zh_TW': 'RTMW3D'
    },
    'source_video': {
        'en': 'Video File',
        'zh_TW': '影片檔案'
    },

    # ── Launcher window ──────────────────────────────────────────────────────
    'launcher_window_title': {
        'en': 'RULA Posture Assessment System',
        'zh_TW': 'RULA 姿勢評估系統'
    },
    'launcher_title': {
        'en': 'RULA Posture Assessment System',
        'zh_TW': 'RULA 姿勢評估系統'
    },
    'launcher_subtitle': {
        'en': 'Select an analysis mode to begin',
        'zh_TW': '選擇分析模式以開始評估'
    },
    'launcher_realtime_title': {
        'en': 'Real-time Analysis',
        'zh_TW': '即時分析'
    },
    'launcher_realtime_desc': {
        'en': 'Open camera for real-time\nRULA posture assessment and scoring',
        'zh_TW': '開啟攝影機，即時進行\nRULA 姿勢評估與評分'
    },
    'launcher_realtime_btn': {
        'en': 'Start Real-time Analysis',
        'zh_TW': '開始即時分析'
    },
    'launcher_upload_title': {
        'en': 'Upload Video Analysis',
        'zh_TW': '上傳影片分析'
    },
    'launcher_upload_desc': {
        'en': 'Select a video file for frame-by-frame\nRULA assessment and report export',
        'zh_TW': '選擇影片檔案，逐幀進行\nRULA 姿勢評估與報告匯出'
    },
    'launcher_upload_btn': {
        'en': 'Analyze Video',
        'zh_TW': '選擇影片分析'
    },
    'launcher_history_btn': {
        'en': 'View Records',
        'zh_TW': '查看分析紀錄'
    },

    # ── Upload window ────────────────────────────────────────────────────────
    'upload_window_title': {
        'en': 'Upload Video Analysis — RULA System',
        'zh_TW': '上傳影片分析 — RULA 評估系統'
    },
    'upload_header_title': {
        'en': 'Upload Video Analysis',
        'zh_TW': '上傳影片分析'
    },
    'upload_header_sub': {
        'en': 'Select a video and fill in survey info for frame-by-frame RULA analysis',
        'zh_TW': '選擇影片檔案並填寫調查資訊，系統將進行逐幀 RULA 分析'
    },
    'upload_back_btn': {
        'en': '← Back',
        'zh_TW': '← 回到主頁'
    },
    'upload_file_group': {
        'en': 'Video File',
        'zh_TW': '影片檔案'
    },
    'upload_no_file': {
        'en': 'No file selected',
        'zh_TW': '尚未選擇檔案'
    },
    'upload_choose_btn': {
        'en': 'Choose Video…',
        'zh_TW': '選擇影片…'
    },
    'upload_choose_dialog_title': {
        'en': 'Select Video File',
        'zh_TW': '選擇影片檔案'
    },
    'upload_meta_group': {
        'en': 'Survey Info',
        'zh_TW': '調查資訊'
    },
    'upload_date_label': {
        'en': 'Survey Date',
        'zh_TW': '調查日期'
    },
    'upload_assessor_label': {
        'en': 'Assessor',
        'zh_TW': '評估員'
    },
    'upload_assessor_placeholder': {
        'en': 'Assessor name',
        'zh_TW': '評估員姓名'
    },
    'upload_org_label': {
        'en': 'Organization',
        'zh_TW': '單位'
    },
    'upload_org_placeholder': {
        'en': 'Organization name',
        'zh_TW': '單位名稱'
    },
    'upload_task_label': {
        'en': 'Task Name',
        'zh_TW': '作業名稱'
    },
    'upload_task_placeholder': {
        'en': 'Task name',
        'zh_TW': '作業名稱'
    },
    'upload_settings_group': {
        'en': 'Analysis Settings',
        'zh_TW': '分析設定'
    },
    'upload_interval_label': {
        'en': 'Sample Interval (frames)',
        'zh_TW': '取樣間隔（幀）'
    },
    'upload_interval_tooltip': {
        'en': 'Sample every N frames (larger = faster, lower accuracy)',
        'zh_TW': '每隔幾幀取樣一次（數字越大越快，精度越低）'
    },
    'upload_backend_label': {
        'en': 'Detection Backend',
        'zh_TW': '偵測後端'
    },
    'upload_backend_rtmw3d': {
        'en': 'RTMW3D (High Accuracy)',
        'zh_TW': 'RTMW3D（高精度）'
    },
    'upload_backend_mediapipe': {
        'en': 'MediaPipe (Fast)',
        'zh_TW': 'MediaPipe（快速）'
    },
    'upload_wrist_twist_label': {
        'en': 'Wrist Twist',
        'zh_TW': '手腕扭轉'
    },
    'upload_wrist_twist_option_neutral': {
        'en': 'Neutral',
        'zh_TW': '無扭轉'
    },
    'upload_wrist_twist_option_twisted': {
        'en': 'Twisted',
        'zh_TW': '有扭轉'
    },
    'upload_legs_label': {
        'en': 'Legs',
        'zh_TW': '腿部姿勢'
    },
    'upload_legs_option_supported': {
        'en': 'Supported / stable',
        'zh_TW': '支撐穩定'
    },
    'upload_legs_option_unstable': {
        'en': 'Unstable / awkward',
        'zh_TW': '不穩定或不自然'
    },
    'upload_muscle_use_a_label': {
        'en': 'Muscle Use A',
        'zh_TW': '肌肉使用 A'
    },
    'upload_muscle_use_b_label': {
        'en': 'Muscle Use B',
        'zh_TW': '肌肉使用 B'
    },
    'upload_muscle_use_option_no': {
        'en': 'No',
        'zh_TW': '否'
    },
    'upload_muscle_use_option_yes': {
        'en': 'Yes',
        'zh_TW': '是'
    },
    'upload_force_load_a_label': {
        'en': 'Force/Load A',
        'zh_TW': '負荷 A'
    },
    'upload_force_load_b_label': {
        'en': 'Force/Load B',
        'zh_TW': '負荷 B'
    },
    'upload_load_option_0': {
        'en': '< 4.4 lb (intermittent)',
        'zh_TW': '載重 < 4.4 磅（間歇）'
    },
    'upload_load_option_1': {
        'en': '4.4-22 lb (intermittent)',
        'zh_TW': '載重 4.4-22 磅（間歇）'
    },
    'upload_load_option_2': {
        'en': '4.4-22 lb (static/repetitive)',
        'zh_TW': '載重 4.4-22 磅（靜態或重複）'
    },
    'upload_load_option_3': {
        'en': '> 22 lb or repetitive/impact',
        'zh_TW': '載重 > 22 磅 或 重複/衝擊'
    },
    'upload_analyze_btn': {
        'en': '▶  Start Analysis',
        'zh_TW': '▶  開始分析'
    },
    'upload_status_ready': {
        'en': 'Ready to analyze…',
        'zh_TW': '準備分析…'
    },
    'upload_status_init': {
        'en': 'Initializing…',
        'zh_TW': '初始化中…'
    },
    'upload_cancel_btn': {
        'en': 'Cancel',
        'zh_TW': '取消'
    },
    'upload_no_file_title': {
        'en': 'No File Selected',
        'zh_TW': '未選擇檔案'
    },
    'upload_no_file_msg': {
        'en': 'Please select a video file first.',
        'zh_TW': '請先選擇要分析的影片檔案。'
    },
    'upload_error_title': {
        'en': 'Analysis Error',
        'zh_TW': '分析錯誤'
    },
    'upload_error_msg': {
        'en': 'Error during analysis:\n{}',
        'zh_TW': '分析過程發生錯誤：\n{}'
    },

    # ── History window ───────────────────────────────────────────────────────
    'history_window_title': {
        'en': 'Analysis Records — RULA System',
        'zh_TW': '分析紀錄查詢 — RULA 評估系統'
    },
    'history_header_title': {
        'en': 'Analysis Records',
        'zh_TW': '分析紀錄查詢'
    },
    'history_header_sub': {
        'en': 'View locally stored video analysis records',
        'zh_TW': '查看本機儲存的影片分析記錄'
    },
    'history_back_btn': {
        'en': '← Back',
        'zh_TW': '← 回到主頁'
    },
    'history_refresh_btn': {
        'en': '↻  Refresh',
        'zh_TW': '↻  重新整理'
    },
    'history_empty': {
        'en': 'No records yet\nPlease analyze a video first',
        'zh_TW': '尚無分析記錄\n請先進行影片分析'
    },
    'history_col_date':     {'en': 'Survey Date',  'zh_TW': '調查日期'},
    'history_col_assessor': {'en': 'Assessor',      'zh_TW': '評估員'},
    'history_col_org':      {'en': 'Organization',  'zh_TW': '單位'},
    'history_col_task':     {'en': 'Task Name',     'zh_TW': '作業名稱'},
    'history_col_filename': {'en': 'Filename',      'zh_TW': '影片檔名'},
    'history_col_created':  {'en': 'Upload Time',   'zh_TW': '上傳時間'},
    'history_col_max':      {'en': 'Max Score',     'zh_TW': '最高分'},
    'history_col_avg':      {'en': 'Avg Score',     'zh_TW': '平均分'},
    'history_col_backend':  {'en': 'Model',         'zh_TW': '辨識模型'},
    'history_col_actions':  {'en': 'Actions',       'zh_TW': '操作'},
    'history_view_btn': {
        'en': 'View',
        'zh_TW': '查看'
    },
    'history_export_btn_text': {
        'en': 'Export',
        'zh_TW': '匯出'
    },
    'history_delete_btn': {
        'en': 'Delete',
        'zh_TW': '刪除'
    },
    'history_export_dialog_title': {
        'en': 'Save CSV Report',
        'zh_TW': '儲存 CSV 報告'
    },
    'history_export_success_title': {
        'en': 'Export Successful',
        'zh_TW': '匯出成功'
    },
    'history_export_success_msg': {
        'en': 'CSV saved to:\n{}',
        'zh_TW': 'CSV 已儲存至：\n{}'
    },
    'history_export_fail_title': {
        'en': 'Export Failed',
        'zh_TW': '匯出失敗'
    },
    'history_delete_confirm_title': {
        'en': 'Confirm Delete',
        'zh_TW': '確認刪除'
    },
    'history_delete_confirm_msg': {
        'en': 'Delete this record?\n{}',
        'zh_TW': '確定要刪除此筆記錄嗎？\n{}'
    },
    'history_delete_fail_title': {
        'en': 'Delete Failed',
        'zh_TW': '刪除失敗'
    },

    # ── Result window ────────────────────────────────────────────────────────
    'result_window_title_suffix': {
        'en': '— RULA Analysis',
        'zh_TW': '— RULA 分析結果'
    },
    'result_header_title_prefix': {
        'en': 'RULA Analysis  —  ',
        'zh_TW': 'RULA 分析結果  —  '
    },
    'result_header_sub_local': {
        'en': 'Local Analysis',
        'zh_TW': '本機分析'
    },
    'result_export_btn': {
        'en': '⬇  Export CSV',
        'zh_TW': '⬇  匯出 CSV'
    },
    'result_close_btn': {
        'en': '✕  Close',
        'zh_TW': '✕  關閉'
    },
    'result_no_video_text': {
        'en': 'No video preview',
        'zh_TW': '無影片預覽'
    },
    'result_skel_checkbox': {
        'en': 'Show Skeleton',
        'zh_TW': '顯示骨架'
    },
    'result_ctrl_title': {
        'en': 'Playback Controls',
        'zh_TW': '採樣播放控制'
    },
    'result_prev_btn': {
        'en': '⏮  Prev',
        'zh_TW': '⏮  上一幀'
    },
    'result_play_btn': {
        'en': '▶  Play',
        'zh_TW': '▶  播放'
    },
    'result_pause_btn': {
        'en': '⏸  Pause',
        'zh_TW': '⏸  暫停'
    },
    'result_next_btn': {
        'en': 'Next  ⏭',
        'zh_TW': '下一幀  ⏭'
    },
    'result_frame_counter': {
        'en': 'Frame {} / {}',
        'zh_TW': '幀 {} / {}'
    },
    'result_time_label': {
        'en': 'Time: {:.2f} s',
        'zh_TW': '時間：{:.2f} s'
    },
    'result_no_video_note': {
        'en': '(Original video unavailable, charts only)',
        'zh_TW': '（原始影片不可用，僅顯示圖表）'
    },
    'result_stat_title':    {'en': 'Summary',        'zh_TW': '分析總覽'},
    'result_stat_total':    {'en': 'Total Frames',   'zh_TW': '總幀數'},
    'result_stat_valid':    {'en': 'Valid Frames',   'zh_TW': '有效幀'},
    'result_stat_invalid':  {'en': 'Invalid Frames', 'zh_TW': '無效幀'},
    'result_stat_duration': {'en': 'Duration',       'zh_TW': '影片長度'},
    'result_stat_analysis_duration': {'en': 'Analysis Time', 'zh_TW': '分析時長'},
    'result_stat_max':      {'en': 'Max Score',      'zh_TW': '最高分'},
    'result_stat_avg':      {'en': 'Avg Score',      'zh_TW': '平均分'},
    'result_stat_anom':     {'en': 'Anomaly Frames', 'zh_TW': '異常幀數'},
    'result_chart_title': {
        'en': 'RULA Score Trend  (click to jump)',
        'zh_TW': 'RULA 分數趨勢  （點擊跳幀）'
    },
    'result_chart_x': {
        'en': 'Time (sec)',
        'zh_TW': '時間（秒）'
    },
    'result_chart_y': {
        'en': 'RULA Score',
        'zh_TW': 'RULA 分數'
    },
    'result_bar_title': {
        'en': 'Score Distribution',
        'zh_TW': '分數分佈'
    },
    'result_bar_x': {
        'en': 'RULA Score',
        'zh_TW': 'RULA 分數'
    },
    'result_bar_y': {
        'en': 'Frames',
        'zh_TW': '幀數'
    },
    'result_tab_trend': {
        'en': 'Score Trend',
        'zh_TW': '分數趨勢'
    },
    'result_tab_bar': {
        'en': 'Bar Distribution',
        'zh_TW': '長條分布'
    },
    'result_tab_pie': {
        'en': 'Pie Distribution',
        'zh_TW': '圓餅分布'
    },
    'result_metrics_btn': {
        'en': '📐 Angles & Scores',
        'zh_TW': '📐 角度與分數'
    },
    'result_metrics_title': {
        'en': 'Frame Angles & Table Scores',
        'zh_TW': '當前幀角度與 Table 分數'
    },
    'result_metrics_left_angles': {
        'en': 'Left Side Angles',
        'zh_TW': '左側角度'
    },
    'result_metrics_right_angles': {
        'en': 'Right Side Angles',
        'zh_TW': '右側角度'
    },
    'result_metrics_left_scores': {
        'en': 'Left Table Scores',
        'zh_TW': '左側 Table 分數'
    },
    'result_metrics_right_scores': {
        'en': 'Right Table Scores',
        'zh_TW': '右側 Table 分數'
    },
    'result_export_dialog_title': {
        'en': 'Save CSV Report',
        'zh_TW': '儲存 CSV 報告'
    },
    'result_export_success_title': {
        'en': 'Export Successful',
        'zh_TW': '匯出成功'
    },
    'result_export_success_msg': {
        'en': 'CSV saved to:\n{}',
        'zh_TW': 'CSV 已儲存至：\n{}'
    },
    'result_export_fail_title': {
        'en': 'Export Failed',
        'zh_TW': '匯出失敗'
    },
}


def get_text(key):
    """获取翻译文本的快捷函数"""
    return language_manager.get_text(key)


def t(key):
    """更简短的翻译函数别名"""
    return language_manager.get_text(key)
