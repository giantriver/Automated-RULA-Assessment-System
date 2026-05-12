# MediaPipe 版本：關節異常偵測實作計畫

## 一、可行性評估

**結論：可行。**

### 現況盤點

| 項目 | 目前狀態 |
|------|----------|
| MediaPipe visibility 欄位 | ✅ 每個 landmark 都有，存在 `landmarks_arr[i][3]` |
| MediaPipe world 3D 座標 | ✅ 已使用，存在 `landmarks_arr[i][0:3]` |
| 基礎異常偵測 | ✅ 已有 `_compute_occlusion_mask`（固定門檻版，**待改名為 `_compute_anomaly_mask`**） |
| `joint_anomaly` 存入 record | ✅ 每幀都會存（目前鍵名為 `joint_occlusion`，**待改名**） |
| 自適應速度門檻（Pass 1） | ✅ 已實作 `_run_pass1()`，MAD-based 自適應門檻，各關節群組獨立 |
| stable `body_scale_ref` | ✅ Pass 1 預掃描計算肩寬中位數，Pass 2 優先使用 |
| 「上一個可靠幀」正確追蹤 | ✅ `_anomaly_prev_reliable`（已改名），Pass 1 完成後重置使用 |
| 異常結果影響 RULA 角度計算 | ✅ `rula_calculator.py` 已加入 `_joints_reliable()`，異常關節 → 角度回傳 NULL |
| 異常種類分類 | ✅ `joint_anomaly_detail` 含 `reason`（3 種）、`visibility`、`speed_ratio` |
| 骨架圖異常關節 X 標記 | ✅ `result_window.py` 的 `_show_frame()` 已實作：讀取 `joint_anomaly`，對 False 的關節畫橘紅色 X（白邊增加對比） |

### 方法可行性

- MediaPipe 提供 33 個 world landmark，每個都有 visibility，完全符合 plan.md 的資料假設。
- `VideoFileProcessor` 已是兩輪可擴充的離線處理器，適合加入 Pass 1 預掃描。
- `rula_score_side()` 已有按 index 查 confidence 的邏輯（`check_confidence`），改成同時查 occlusion mask 不需大幅重構。
- 本計畫只針對**離線影片分析（`VideoFileProcessor`）**。

---

## 二、修改計畫概覽

> **第一版目標（已完成）：找出異常關節點，並讓異常關節導致角度回傳 NULL（與低信心一致），不修改 RULA 評分表格邏輯。**

```
✅ Phase 0：全專案命名統一（occlusion → anomaly）
✅ Phase 1：video_file_processor.py — 加入 Pass 1，自適應門檻
✅ Phase 2：video_file_processor.py — 強化 Pass 2，整合至主處理迴圈
             ↳ _compute_anomaly_mask 新增第三回傳值 detail（含 reason/visibility/speed_ratio）

── 以上完成後，result_window.py 的骨架圖 X 標記即可自動受益 ──

✅ Phase 3：rula_calculator.py — 異常關節導致 angle_calc 回傳 NULL（_joints_reliable helper）
✅ Phase 4：result_window + dialogs.py — 新增異常幀統計 stat card；關節彈窗顯示 4 欄診斷資訊
❌ Phase 5（未來）：修改 RULA 評分表格邏輯（異常 → 分數懲罰或標注）
```

---

## 三、詳細修改步驟

### Phase 0｜命名統一（occlusion → anomaly）

「遮擋」一詞只涵蓋 visibility 低的情況，無法描述速度異常跳動（模型誤判）這類問題。統一改為「異常」，語意更精準。

#### 0-1　`video_file_processor.py` 改名清單

| 舊名稱 | 新名稱 |
|--------|--------|
| `_compute_occlusion_mask` | `_compute_anomaly_mask` |
| `_occ_prev_reliable` | `_anomaly_prev_reliable` |
| `_occ_dt` | `_anomaly_dt` |
| `_occ_vis_th` / `_OCC_VIS_TH` | `_ANOM_VIS_TH` |
| `_OCC_VIS_MID_TH` | `_ANOM_VIS_MID_TH` |
| `_OCC_SPEED_TH_LOW` | `_ANOM_SPEED_TH_LOW` |
| `_OCC_SPEED_TH_HIGH` | `_ANOM_SPEED_TH_HIGH` |
| record 欄位 `joint_occlusion` | `joint_anomaly` |

#### 0-2　`result_window.py` 改名清單

| 舊內容 | 新內容 |
|--------|--------|
| `# ── Occlusion overlay (MediaPipe only)` | `# ── Joint anomaly overlay (MediaPipe only)` |
| `joint_occlusion = rec.get('joint_occlusion')` | `joint_anomaly = rec.get('joint_anomaly')` |
| `[OCC DEBUG]` print | `[ANOM DEBUG]` print（或後續直接刪除） |
| `for i, reliable in enumerate(joint_occlusion)` | `for i, reliable in enumerate(joint_anomaly)` |
| `if (joint_occlusion and ...` | `if (joint_anomaly and ...` |

---

### Phase 1｜`video_file_processor.py` — Pass 1 預掃描

#### 1-1　定義關節群組常數

在檔案頂部（目前 `_OCC_*` 常數附近）新增：

```python
# MediaPipe 33 點關節群組（用於建立各群組速度分布）
_JOINT_GROUPS = {
    "trunk": [11, 12, 23, 24],          # 左右肩、左右髖
    "head":  [0, 7, 8],                 # 鼻子、左右耳
    "arm":   [13, 14],                  # 左右肘
    "hand":  [15, 16, 17, 18, 19, 20],  # 左右腕、手指點
    "leg":   [25, 26, 27, 28],          # 左右膝、左右踝
}

# 每個關節屬於哪個群組（反查表）
_JOINT_TO_GROUP: dict[int, str] = {
    jidx: grp
    for grp, idxs in _JOINT_GROUPS.items()
    for jidx in idxs
}
```

#### 1-2　新增 `_run_pass1()` 函式

在 `_compute_occlusion_mask` 下方新增：

```python
def _run_pass1(
    video_path: str,
    detector: PoseDetector,
    frame_interval: int,
    fps: float,
    vis_high: float = 0.80,
) -> tuple[float, dict[str, tuple[float, float]]]:
    """
    Pass 1：預掃描影片，計算 body_scale_ref 與各關節群組自適應速度門檻。

    Returns:
        body_scale_ref  : 穩定肩寬中位數（> 0 才有效）
        group_thresholds: {group_name: (th_low, th_high)}
    """
    import statistics

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, {}

    shoulder_widths: list[float] = []
    # 每個群組各收集一份速度樣本 list
    group_speeds: dict[str, list[float]] = {g: [] for g in _JOINT_GROUPS}

    prev_reliable = [None] * 33  # [x, y, z] or None
    prev_frame_idx = -1
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected = detector.process_frame(rgb)
            if detected:
                arr = detector.get_landmarks_array()
                if arr and len(arr) == 33:
                    # 收集肩寬樣本
                    L_SHO, R_SHO = arr[11], arr[12]
                    if L_SHO[3] >= vis_high and R_SHO[3] >= vis_high:
                        sw = (
                            (L_SHO[0]-R_SHO[0])**2 +
                            (L_SHO[1]-R_SHO[1])**2 +
                            (L_SHO[2]-R_SHO[2])**2
                        ) ** 0.5
                        if sw > 1e-6:
                            shoulder_widths.append(sw)

                    # 收集速度樣本（只用高 visibility 點）
                    if prev_frame_idx >= 0:
                        dt = (frame_idx - prev_frame_idx) / fps
                        for i, lm in enumerate(arr):
                            grp = _JOINT_TO_GROUP.get(i)
                            if grp is None:
                                continue
                            if float(lm[3]) < vis_high:
                                continue
                            prev = prev_reliable[i]
                            if prev is None:
                                continue
                            jump = (
                                (lm[0]-prev[0])**2 +
                                (lm[1]-prev[1])**2 +
                                (lm[2]-prev[2])**2
                            ) ** 0.5
                            # 先用暫時肩寬做正規化（Pass 1 結束後再用穩定值，影響不大）
                            tmp_scale = shoulder_widths[-1] if shoulder_widths else 0.1
                            speed_ratio = (jump / dt) / tmp_scale if dt > 1e-9 else 0.0
                            group_speeds[grp].append(speed_ratio)

                    # 更新 prev_reliable（pass 1 全用高 visibility 的點）
                    for i, lm in enumerate(arr):
                        if float(lm[3]) >= vis_high:
                            prev_reliable[i] = [lm[0], lm[1], lm[2]]
                    prev_frame_idx = frame_idx

        frame_idx += 1

    cap.release()

    # 計算 body_scale_ref
    body_scale_ref = statistics.median(shoulder_widths) if shoulder_widths else 0.0

    # 計算各群組自適應門檻（MAD-based robust statistics）
    group_thresholds: dict[str, tuple[float, float]] = {}
    for grp, speeds in group_speeds.items():
        if len(speeds) < 10:
            # 樣本不足，退回百分位數；若仍不足則用固定值
            if len(speeds) >= 5:
                speeds_sorted = sorted(speeds)
                n = len(speeds_sorted)
                th_low  = speeds_sorted[int(n * 0.95)]
                th_high = speeds_sorted[int(n * 0.99)]
            else:
                th_low, th_high = _ANOM_SPEED_TH_LOW, _ANOM_SPEED_TH_HIGH
        else:
            med = statistics.median(speeds)
            abs_devs = [abs(v - med) for v in speeds]
            mad = statistics.median(abs_devs)
            robust_std = 1.4826 * mad
            th_low  = med + 3 * robust_std
            th_high = med + 5 * robust_std
            # 避免 MAD=0 造成門檻過低
            if robust_std < 1e-6:
                speeds_sorted = sorted(speeds)
                n = len(speeds_sorted)
                th_low  = speeds_sorted[int(n * 0.95)]
                th_high = speeds_sorted[int(n * 0.99)]

        # 至少要大於固定下限，避免正常動作被誤判
        th_low  = max(th_low,  _ANOM_SPEED_TH_LOW)
        th_high = max(th_high, _ANOM_SPEED_TH_HIGH)
        group_thresholds[grp] = (th_low, th_high)

    return body_scale_ref, group_thresholds
```

---

### Phase 2｜`video_file_processor.py` — 強化 `_compute_occlusion_mask` 與主迴圈

#### 2-1　修改 `_compute_anomaly_mask` 簽名，接受自適應門檻，並回傳 detail

```python
def _compute_anomaly_mask(
    landmarks_arr,
    prev_reliable,
    body_scale,
    dt,
    group_thresholds: dict | None = None,   # 新增
) -> tuple[list[bool], list, list]:  # 新增第三回傳值
```

函式內部把原本的固定常數改成從 `group_thresholds` 查詢，並收集每個關節的異常原因：

```python
    for i, lm in enumerate(landmarks_arr):
        x, y, z, vis = float(lm[0]), float(lm[1]), float(lm[2]), float(lm[3])
        grp = _JOINT_TO_GROUP.get(i, 'trunk')
        if group_thresholds:
            th_low, th_high = group_thresholds.get(grp, (_ANOM_SPEED_TH_LOW, _ANOM_SPEED_TH_HIGH))
        else:
            th_low, th_high = _ANOM_SPEED_TH_LOW, _ANOM_SPEED_TH_HIGH
        reliable = True
        reason = None
        speed_ratio_val = None

        if vis < _ANOM_VIS_TH:
            reliable = False
            reason = 'low_visibility'
        else:
            prev = prev_reliable[i]
            if prev is not None and dt > 1e-9 and body_scale > 1e-6:
                jump = ((x-prev[0])**2 + (y-prev[1])**2 + (z-prev[2])**2) ** 0.5
                speed_ratio_val = (jump / dt) / body_scale
                if speed_ratio_val > th_high:
                    reliable = False
                    reason = 'speed_jump'
                elif vis < _ANOM_VIS_MID_TH and speed_ratio_val > th_low:
                    reliable = False
                    reason = 'low_vis_speed_jump'

        mask.append(reliable)
        if reliable:
            new_prev[i] = [x, y, z]
            detail.append(None)
        else:
            detail.append({
                'reason': reason,
                'visibility': vis,
                'speed_ratio': speed_ratio_val,
            })

    return mask, new_prev, detail
```

**三種 reason 的意義：**

| reason | 觸發條件 |
|--------|----------|
| `low_visibility` | `visibility < 0.50` |
| `speed_jump` | 速度比 > `th_high`（無視 visibility）|
| `low_vis_speed_jump` | `0.50 ≤ vis < 0.80` 且速度比 > `th_low` |

#### 2-2　在 `_process()` 主迴圈加入 Pass 1 呼叫

在 `detector = PoseDetector(...)` 初始化之後、`while not self._cancelled` 之前：

```python
# Pass 1：建立自適應速度門檻（MediaPipe only）
_body_scale_ref = 0.0
_group_thresholds: dict = {}
if self.backend_mode == 'MEDIAPIPE':
    self.progress_updated.emit(4, 'Pass 1：建立速度分布...')
    _body_scale_ref, _group_thresholds = _run_pass1(
        self.video_path, detector, self.frame_interval, fps
    )
    # 重置影片讀取位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
```

#### 2-3　Pass 2 使用穩定 body_scale_ref，異常偵測在角度計算之前

原本每幀即時算 `body_scale`，改成：

```python
if self.backend_mode == 'MEDIAPIPE' and landmarks_arr and len(landmarks_arr) == 33:
    # 優先用 Pass 1 算出的穩定肩寬；若 Pass 1 失敗才退回當幀即時值
    if _body_scale_ref > 1e-6:
        body_scale = _body_scale_ref
    else:
        L_SHO, R_SHO = landmarks_arr[11], landmarks_arr[12]
        body_scale = (
            (L_SHO[0]-R_SHO[0])**2 +
            (L_SHO[1]-R_SHO[1])**2 +
            (L_SHO[2]-R_SHO[2])**2
        ) ** 0.5
    # ⚠️ 異常偵測必須在 angle_calc 之前，才能把 joint_anomaly 傳入
    joint_anomaly, _anomaly_prev_reliable, joint_anomaly_detail = _compute_anomaly_mask(
        landmarks_arr, _anomaly_prev_reliable, body_scale, _anomaly_dt,
        group_thresholds=_group_thresholds,
    )

# 角度計算（傳入 joint_anomaly 讓異常關節回傳 NULL）
angle_data = angle_calc(pose, ..., joint_anomaly=joint_anomaly)

# 存入 record
record['joint_anomaly']        = joint_anomaly
record['joint_anomaly_detail'] = joint_anomaly_detail  # list[None|dict]
```

---

---

### Phase 3｜`rula_calculator.py` — 異常關節導致角度回傳 NULL ✅

#### 3-1　新增 `_joints_reliable()` helper

```python
def _joints_reliable(joint_anomaly, indices: list) -> bool:
    """Returns False if any of the required joint indices is marked anomalous."""
    if joint_anomaly is None:
        return True
    for idx in indices:
        if idx < len(joint_anomaly) and not joint_anomaly[idx]:
            return False
    return True
```

#### 3-2　`rula_score_side` / `angle_calc` 新增 `joint_anomaly` 參數

```python
def rula_score_side(pose, side: str, previous_scores=None, joint_anomaly=None): ...
def angle_calc(pose, previous_left=None, previous_right=None, joint_anomaly=None): ...
```

#### 3-3　每個角度的條件加入 `_joints_reliable` 檢查

```python
# 原本
if check_confidence(pose, upper_arm_indices):
    angle = ...
else:
    angle_data['upper_arm_angle'] = 'NULL'

# 修改後（5 個角度均相同模式）
if check_confidence(pose, upper_arm_indices) and _joints_reliable(joint_anomaly, upper_arm_indices):
    angle = ...
else:
    angle_data['upper_arm_angle'] = 'NULL'
```

受影響的角度：upper_arm、lower_arm、wrist、neck、trunk。

> **Note：** 本版本讓異常關節導致角度 NULL（等同低信心），不修改 RULA 評分表格。修改評分邏輯留待後續 Phase 5。

---

### 補充：`result_window.py` X 標記現況

`result_window.py` 的 `_show_frame()` 中已有完整的 X 標記繪製邏輯。**Phase 0 改名後**，程式碼應如下（僅改變數名與註解）：

```python
# ── Joint anomaly overlay (MediaPipe only) ────────────────────
joint_anomaly = rec.get('joint_anomaly')
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
```

**結論：Phase 0 只改名稱與註解；只要 Phase 1 & 2 提升了 `joint_anomaly` 的品質，X 標記視覺化即自動受益。**

---

### Phase 4｜Result Window + Dialogs — 異常幀統計與關節彈窗診斷資訊 ✅

#### 4-1　`result_window.py` 新增 Anomaly Frames stat card

```python
anom_frames = sum(
    1 for r in self._records
    if r.get('joint_anomaly') and not all(r['joint_anomaly'])
)
anom_text = str(anom_frames) if anom_frames > 0 else '—'
# stat card 顏色：紫色 #7c3aed / #ede9fe
```

#### 4-2　`language.py` 新增 i18n key

```python
'result_stat_anom': {'en': 'Anomaly Frames', 'zh_TW': '異常幀數'},
```

#### 4-3　`dialogs.py` — `_show_joint_popup` 改為 4 欄診斷表格

原本只顯示關節名稱與 confidence，改為顯示：

| Joint | Confidence | Anomaly | Speed ratio |
|-------|-----------|---------|-------------|
| L Shoulder | 0.92 ✓ | — | — |
| L Elbow | 0.41 ✗ | low_vis | 1.2 |
| L Wrist | 0.78 ✓ | speed_jump | **4.7** |

- **Confidence**：visibility 值 + ✓/✗（紅/綠色）
- **Anomaly**：從 `joint_anomaly_detail[i]['reason']` 映射：
  - `'low_visibility'` → `'low_vis'`
  - `'speed_jump'` → `'speed_jump'`
  - `'low_vis_speed_jump'` → `'low_vis+spd'`
  - `None`（正常）→ `'—'`（綠色）
- **Speed ratio**：從 `joint_anomaly_detail[i]['speed_ratio']`，N/A 時顯示 `'—'`

資料來源：`self._rec.get('joint_anomaly_detail') or []`

---

## 四、改動檔案清單

| 檔案 | 改動類型 | 狀態 |
|------|---------|------|
| `core/video_file_processor.py` | Phase 0：常數/函式/欄位改名；Phase 1&2：新增 `_run_pass1()`、修改 `_compute_anomaly_mask`（加入 detail 回傳）、修改 `_process()`（異常偵測先於角度計算）| ✅ 完成 |
| `ui/result_window.py` | Phase 0：改註解與變數名；Phase 4：新增 Anomaly Frames stat card | ✅ 完成 |
| `core/rula_calculator.py` | Phase 3：新增 `_joints_reliable()`、修改 `rula_score_side()`、修改 `angle_calc()`（異常關節 → NULL） | ✅ 完成 |
| `ui/dialogs.py` | Phase 4：`_show_joint_popup` 改為 4 欄表格（Joint/Confidence/Anomaly/Speed ratio） | ✅ 完成 |
| `ui/language.py` | Phase 4：新增 `result_stat_anom` i18n key | ✅ 完成 |

---

## 五、測試建議

1. 在 `_process()` 完成後，加一段 debug print，印出各幀 `joint_anomaly` 中 False 的關節數量，確認異常偵測有正常觸發。
2. 人工製造異常片段（遺住鏡頭、或剤輯一段純手遺臉的影片），確認對應關節的 `joint_anomaly[i]` 確實為 False。
3. 確認 Pass 1 不影響整體分析時間超過 30%（Pass 1 跟 Pass 2 共讀兩次影片；若太慢可考慮只掃前 N 秒）。
4. 對比 `group_thresholds` 的計算結果，確認各群組門檻合理（hand 群組的門檻應高於 trunk 群組）。
5. 確認正常影片的 `joint_anomaly` **全部為 True**，不會誤報。

---

## 六、已知限制與後續建議

- **RTMPose 版本**：RTMPose 的信心分數語意與 MediaPipe visibility 不同（confidence ≠ visibility），需另外驗證門檻參數，不在第一版範圍內。
- **遮擋補值策略**：第一版採「沿用上一個可靠分數」，長時間遮擋（>N 幀）應改為標記 `unavailable`，可設定 `max_carry_frames` 參數。
