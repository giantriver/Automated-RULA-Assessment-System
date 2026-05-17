# 關節異常判定實現分析

## 概述

「關節異常判定」（Joint Anomaly Detection）是RULA分析流程中的一個關鍵環節，用於識別和排除因**置信度低、快速跳躍、遮擋**等原因導致的不可靠關節點，確保後續角度計算的準確性。

目前此功能**僅在 MediaPipe 後端中實現**，RTMW3D 後端暫不支援。

---

## 速度計算方法總整理（目前版本）

本專案的「速度」不是直接用像素速度，而是使用 **3D 關節位移速度經身體尺度正規化後的無因次速度比**（`speed_ratio`）。

### 核心公式

1. 關節 3D 位移（相鄰分析幀）

$$
jump = \sqrt{(x_t-x_{t-1})^2 + (y_t-y_{t-1})^2 + (z_t-z_{t-1})^2}
$$

2. 時間差

$$
dt = \frac{\Delta frame}{fps}
$$

3. 正規化速度比

$$
speed\_ratio = \frac{jump / dt}{body\_scale}
$$

說明：
- `jump / dt` 是世界座標下的瞬時速度。
- `body_scale` 使用肩寬作為人體尺度，讓不同身材可用同一門檻比較。
- `speed_ratio` 越大，代表相對於人體尺度的動作越「不合理地快」。

### Pass 1 與 Pass 2 的速度計算差異

#### Pass 1（建立分布與門檻）

- 目的：收集「正常高可信動作」的 raw speed（`jump / dt`）分布，掃描結束後再以 `body_scale_ref` 正規化成 `speed_ratio`，然後計算群組門檻。
- 僅納入高可見度點（`visibility >= 0.80`）。
- `dt` 來源：`(frame_idx - prev_frame_idx) / fps`。
- `body_scale_ref` 將在掃描完成後使用肩寬樣本的中位數計算。
- 每點計算出的 `raw_speed = jump / dt` 會依關節群組累積到 `group_raw_speeds[group]`，掃描結束後再做正規化。

#### Pass 2（逐幀異常判定）

- 目的：對每個關節即時判斷是否異常。
- **目前實作的 `dt` 是固定分析取樣間隔**：
    - `_anomaly_dt = frame_interval / fps`。
    - 但在實作上，Pass 2 會另外記錄上一個可靠點的 `frame_idx`，因此可以把 `prev_reliable` 與當前幀之間的**實際 gap** 算出來。
- `prev_reliable` 會同時保存上一個可靠點的位置與幀索引，例如：
    ```python
    {
        "pos": [x, y, z],
        "frame_idx": frame_idx
    }
    ```
- `body_scale` 來源：
    - 優先使用 Pass 1 的穩定肩寬中位數 `body_scale_ref`。
    - 若 Pass 1 無有效樣本，退回當前幀即時肩寬。
- `speed_ratio` 只在以下條件成立才計算：
    - 該點有 `prev_reliable`。
    - `dt > 1e-9`。
    - `body_scale > 1e-6`。

### 門檻如何套用到速度

- 每個關節先映射到群組（`trunk/head/arm/hand/leg`）。
-- 取該群組門檻 `(th_low, th_high)`：
    - 有 Pass 1 結果時：用群組自適應門檻（Pass 1 掃描時先以 raw speed 收集樣本，掃描結束後除以 `body_scale_ref` 得到 speed_ratio）。
    - 否則：用固定門檻 `3.0 / 10.0`。
- 判定規則：
    - `speed_ratio > th_high` → `speed_jump`（極端跳躍）。
    - `visibility < 0.80` 且 `speed_ratio > th_low` → `low_vis_speed_jump`。

### 為什麼這樣設計

- 同時考慮「空間位移」「時間尺度」「人體尺寸」，比單純位移更穩定。
- 先用 Pass 1 建立個體/影片特性，再於 Pass 2 判斷，可降低固定閾值誤判。
- 用 `prev_reliable` 而非前一幀，避免異常值連鎖污染速度估計。

**長時間間隔（gap）處理策略**

當目前關節位置與 `prev_reliable`（上一個被判定為可靠的同一關節）之間的時間間隔過長時，兩者之間可能已發生若干未被觀測到的正常動作。若此時仍以速度突變判定為異常，會有較高的誤判風險（例如遮擋後重新出現的正常位置被視為跳躍）。

因此，本研究在實作上採取保守策略：當時間間隔（從 `prev_reliable` 對應的幀到目前檢測幀）超過 1 秒時，**略過速度一致性檢查**（即不以 `speed_ratio` 判定該關節是否異常），僅以 `visibility` 或其他即時指標做初步判定。

這個規則應同時套用在 **Pass 1 與 Pass 2**：

- **Pass 1**：目標是建立「正常速度分布」。如果中間缺很多幀才重新出現，該段速度並不是穩定、連續觀測到的正常速度，直接納入樣本會污染 threshold，因此應跳過。
- **Pass 2**：目標是逐幀異常判定。若 gap 過長，代表目前觀測到的跳變不一定來自異常動作，而可能只是長時間未觀測後的重新出現，因此也應跳過速度檢查。
- **注意**：Pass 2 的 gap 判斷是透過 `frame_idx` 來計算實際經過時間，因此 `prev_reliable` 必須保存 `frame_idx`，不能只存座標。

建議的實作範例：

```python
# 假設 prev_reliable[i] 內同時保存了上一個可靠點的位置與幀索引，frame_idx 為當前幀索引，fps 已知
max_gap_seconds = 1.0  # 本文件建議使用 1 秒
gap_seconds = (frame_idx - prev_reliable[i]["frame_idx"]) / fps
if gap_seconds > max_gap_seconds:
    # 略過速度檢查，僅以 visibility 判定
    do_speed_check = False
else:
    do_speed_check = True

if do_speed_check and prev is not None:
    # 計算 jump / dt -> speed_ratio 並比較門檻
    ...
else:
    # 跳過速度檢查（避免誤判）
    ...
```

這個保守策略能有效降低遮擋重現（occlusion recovery）或長時間未觀測時的誤判；同時也能避免 Pass 1 的正常速度分布被不連續片段污染。若之後需要調整敏感度，可把 `max_gap_seconds` 做成可配置參數，但本文預設值建議先固定為 1 秒。

---

## 實現架構

### 整體流程圖

```
影片輸入
   ↓
┌─────────────────────────────────┐
│ Pass 1: 預掃描（MediaPipe only） │
├─────────────────────────────────┤
│ 1. 掃過全影片（抽樣幀）          │
│ 2. 計算肩寬中位數 (body_scale)  │
│ 3. 建立各關節群組速度分布        │
│ 4. 生成自適應速度門檻            │
└─────────────────────────────────┘
   ↓
┌──────────────────────────────────┐
│ Pass 2: 逐幀分析 (全後端)        │
├──────────────────────────────────┤
│ for each sampled frame:           │
│   ├─ MediaPipe: _compute_anomaly │
│   │    └─ 判定33個關節的可靠性   │
│   ├─ 角度計算 (angle_calc)       │
│   │    └─ 跳過異常關節            │
│   └─ 存儲 joint_anomaly 至記錄   │
└──────────────────────────────────┘
   ↓
UI 展示 / CSV 匯出
```

---

## 核心判定邏輯

### 1. Pass 1：自適應門檻建立 (`_run_pass1`)

#### 位置
`src/rula_realtime_app/core/video_file_processor.py` 行 104~199

#### 目的
在 MediaPipe 後端中，計算每個關節群組的**自適應速度門檻**，而非使用固定值。

#### 輸入參數
- `video_path`: 分析的影片
- `detector`: PoseDetector 實例
- `frame_interval`: 抽樣間隔（預設10幀）
- `fps`: 影片幀率
- `vis_high`: 高置信度門檻（預設0.80）

#### 流程

1. **掃描影片，收集樣本**
   ```python
   while 讀取幀:
       if 是抽樣幀:
           偵測姿勢，取得33點關節
           ├─ 收集肩寬（用於身體尺度）
           ├─ 若有上一個可靠幀，計算速度比
           └─ 將高置信度點加入各群組速度分布
   ```

2. **計算身體尺度參考 (body_scale_ref)**
   - 用肩寬（左肩~右肩距離）作為參考單位
   - 取所有有效肩寬樣本的中位數
   - 用途：正規化速度（使其與人體大小無關）

3. **計算群組速度門檻**
   
   首先將33個關節分為5個群組：
   ```python
   _JOINT_GROUPS = {
       'trunk': [11, 12, 23, 24],    # 左右肩、左右髖
       'head':  [0, 7, 8],            # 鼻子、左右耳
       'arm':   [13, 14],             # 左右肘
       'hand':  [15, 16, 17, 18, 19, 20],  # 腕、手指點
       'leg':   [25, 26, 27, 28],    # 膝、踝
   }
   ```

   對每個群組使用 **MAD（中位數絕對偏差）** 方法計算健壯統計：
   ```
   med = 群組速度中位數
   MAD = |v - med| 的中位數
   σ_robust = 1.4826 × MAD
   
   th_low  = med + 3 × σ_robust    (中度異常門檻)
   th_high = med + 5 × σ_robust    (極端異常門檻)
   ```

   若樣本不足，改用百分位法：
   ```
   th_low  = 95th percentile
   th_high = 99th percentile
   ```

#### 回傳值
```python
body_scale_ref: float           # 穩定肩寬中位數
group_thresholds: dict = {
    'trunk': (th_low, th_high),
    'head':  (th_low, th_high),
    'arm':   (th_low, th_high),
    'hand':  (th_low, th_high),
    'leg':   (th_low, th_high),
}
```

---

### 2. Pass 2：逐幀異常判定 (`_compute_anomaly_mask`)

#### 位置
`src/rula_realtime_app/core/video_file_processor.py` 行 41~103

#### 目的
在每個分析幀判定33個關節中哪些是可靠（True）、哪些異常（False）。

#### 輸入參數
```python
landmarks_arr:          # [33, 4] 陣列 (x, y, z, visibility)
prev_reliable:          # list[list|None]，上一個可靠幀的各點 3D 座標
body_scale:             # 身體參考尺度（肩寬）
dt:                     # 兩個分析幀的時間差（秒）
group_thresholds:       # dict，由 Pass 1 計算（若為 None 則用固定值）
```

#### 判定規則

對每個關節（i = 0...32），順序執行：

```
reliable = True，reason = None

1️⃣ 置信度檢查（visibility）
   ├─ if vis < 0.50:  直接不可靠
   │   ├─ reason = 'low_visibility'
   │   └─ reliable = False
   └─ else: 進入下一步

2️⃣ 速度跳躍檢查（需有上一個可靠幀）
   ├─ if prev[i] is None: 跳過速度檢查
   ├─ if dt 太小 or body_scale 太小: 跳過速度檢查
   │
   └─ 計算速度比（正規化跳躍距離）
      jump = √((x-prev_x)² + (y-prev_y)² + (z-prev_z)²)
      speed_ratio = (jump / dt) / body_scale
      
      ├─ if speed_ratio > th_high[關節群組]
      │   ├─ reason = 'speed_jump'（極端跳躍）
      │   └─ reliable = False
      │
      └─ elif vis < 0.80 AND speed_ratio > th_low[關節群組]
          ├─ reason = 'low_vis_speed_jump'（低置信+中速）
          └─ reliable = False

3️⃣ 更新追蹤狀態
   ├─ if reliable: prev[i] = [x, y, z]（更新此點）
   └─ else: prev[i] 保留舊值
```

#### 回傳值

```python
mask (list[bool]):
    # True = 可靠，False = 疑似異常
    [True, False, True, ..., False]  # length = 33

new_prev (list[list|None]):
    # 更新後的上一個可靠幀座標
    [None, [x,y,z], None, ..., [x,y,z]]

detail (list[None|dict]):
    # 診斷明細
    [
        None,                           # 可靠的點
        {
            'reason': 'low_visibility',
            'visibility': 0.4521,
            'speed_ratio': None
        },
        {
            'reason': 'speed_jump',
            'visibility': 0.8934,
            'speed_ratio': 15.42
        },
        ...
    ]
```

---

## 數據流

### 在主處理循環中的使用

#### 位置
`src/rula_realtime_app/core/video_file_processor.py` 行 315~380

```python
# Pass 2 主循環
for each sampled frame:
    detected = detector.process_frame(rgb)
    
    if detected and backend_mode == 'MEDIAPIPE':
        landmarks_arr = detector.get_landmarks_array()  # [33, 4]
        
        # ⚠️ 關鍵：異常判定必須在角度計算之前
        joint_anomaly, _anomaly_prev_reliable, joint_anomaly_detail = \
            _compute_anomaly_mask(
                landmarks_arr,
                _anomaly_prev_reliable,
                body_scale,
                _anomaly_dt,
                group_thresholds=_group_thresholds
            )
    
    # 將異常資訊傳給角度計算
    rula_left, rula_right = angle_calc(
        landmarks_arr,
        prev_left, prev_right,
        joint_anomaly=joint_anomaly  # ← 關鍵！
    )
    
    # 存儲到記錄
    record = {
        'joint_anomaly': joint_anomaly,           # list[bool] | None
        'joint_anomaly_detail': joint_anomaly_detail,  # list[dict|None] | None
        'frame': frame_idx,
        'timestamp': frame_idx / fps,
        'best_score': score_num,
        ...
    }
```

### 在角度計算中的使用

#### 位置
`src/rula_realtime_app/core/rula_calculator.py`

#### 核心邏輯

```python
def _joints_reliable(joint_anomaly, indices: list) -> bool:
    """檢查指定關節是否都可靠"""
    if joint_anomaly is None:
        return True  # 非 MediaPipe 後端，全部視為可靠
    
    for idx in indices:
        if idx < len(joint_anomaly) and not joint_anomaly[idx]:
            return False  # 任何關節異常 → 整個角度計算失敗
    return True
```

#### 角度計算時的處理

在 `rula_score_side()` 函數中：

```python
# 計算上臂角度
upper_arm_indices = [L_SHOULDER, L_ELBOW]  if side == 'Left' else [R_SHOULDER, R_ELBOW]

if check_confidence(pose, upper_arm_indices + [L_HIP, R_HIP]) and \
        _joints_reliable(joint_anomaly, upper_arm_indices + [L_HIP, R_HIP]):
    # 所有需要的關節都可靠
    計算角度...
    upper_arm_angle = theta  (數值)
else:
    # 任何關節異常 → 該角度無法計算
    upper_arm_angle = 'NULL'  (字串)
    # 根據配置，可選用上一幀結果或回退預設值
```

---

## 常數定義

### 固定門檻值

位置：`src/rula_realtime_app/core/video_file_processor.py` 行 19~23

```python
_ANOM_VIS_TH       = 0.50   # visibility 低於此值 → 直接不可靠
_ANOM_VIS_MID_TH   = 0.80   # visibility 中間帶（用於判定是否需檢查速度）
_ANOM_SPEED_TH_LOW  = 3.0   # 速度門檻（中）：低置信度時觸發
_ANOM_SPEED_TH_HIGH = 10.0  # 速度門檻（極端）：任何置信度都觸發
```

### 關節群組定義

```python
_JOINT_GROUPS: dict[str, list[int]] = {
    'trunk': [11, 12, 23, 24],
    'head':  [0, 7, 8],
    'arm':   [13, 14],
    'hand':  [15, 16, 17, 18, 19, 20],
    'leg':   [25, 26, 27, 28],
}

_JOINT_TO_GROUP: dict[int, str]  # 反查表，用於快速找到點所屬群組
```

---

## 可視化

### 在結果視窗中標記異常關節

#### 位置
`src/rula_realtime_app/ui/result_window.py` 行 888~905

#### 實現方式

```python
# 顯示幀時，檢查異常標記
joint_anomaly = rec.get('joint_anomaly')  # list[bool] | None

if joint_anomaly and backend == 'MEDIAPIPE':
    lms_2d = native.get('landmarks_2d')  # 正規化座標 [0~1]
    
    for i, reliable in enumerate(joint_anomaly):
        if not reliable and i < len(lms_2d):
            # 轉換為像素座標
            cx = int(lms_2d[i][0] * width)
            cy = int(lms_2d[i][1] * height)
            
            # 畫橘紅色 X 標記（外框白色增加對比）
            d = 8
            cv2.line(frame, (cx-d, cy-d), (cx+d, cy+d), (255,255,255), 4)  # 白邊
            cv2.line(frame, (cx-d, cy-d), (cx+d, cy+d), (255,80,0), 2)    # 橘紅
            cv2.line(frame, (cx+d, cy-d), (cx-d, cy+d), (255,255,255), 4)  # 白邊
            cv2.line(frame, (cx+d, cy-d), (cx-d, cy+d), (255,80,0), 2)    # 橘紅
```

視覺效果：異常關節被標記為帶白邊框的橘紅色 ❌

---

## 數據結構總結

### Record 中的異常相關欄位

```python
record = {
    'joint_anomaly': [
        True,   # 關節 0（鼻子）可靠
        False,  # 關節 1（左眼）異常
        True,   # 關節 2（右眼）可靠
        ...,
        False,  # 關節 32（右腳尖）異常
    ],
    
    'joint_anomaly_detail': [
        None,   # 關節 0：可靠，無診斷
        {
            'reason': 'low_visibility',
            'visibility': 0.3512,
            'speed_ratio': None
        },
        None,   # 關節 2：可靠
        ...,
        {
            'reason': 'speed_jump',
            'visibility': 0.7234,
            'speed_ratio': 12.54
        }
    ],
}
```

---

## 異常原因分類

| reason | 含義 | 觸發條件 | 備註 |
|--------|------|--------|------|
| `'low_visibility'` | 置信度過低 | `vis < 0.50` | 直接判定，無需檢查速度 |
| `'speed_jump'` | 極端速度跳躍 | `speed_ratio > th_high` | 任何置信度下都觸發 |
| `'low_vis_speed_jump'` | 低置信度 + 中速異常 | `vis < 0.80 AND speed_ratio > th_low` | 置信度不夠，速度又可疑 |

---

## 後端支援情況

| 項目 | MediaPipe | RTMW3D |
|------|-----------|---------|
| Pass 1（自適應門檻） | ✅ 已實現 | ❌ 不支援 |
| Pass 2（逐幀判定） | ✅ 已實現 | ❌ 不支援 |
| 角度計算時排除 | ✅ 已實現 | ✅ 已實現（但無異常標記） |
| UI 標記異常點 | ✅ 已實現 | ❌ 不支援 |

---

### 常見問題

**Q: 為什麼某些幀的角度顯示 'NULL'？**
A: 表示該角度所需的一個或多個關節被判定為異常，無法計算。

**Q: 可以禁用異常判定嗎？**
A: 目前異常判定是 MediaPipe 後端的必要流程，無法關閉。但可修改 `_ANOM_VIS_TH` 為極低值（如0.01）使幾乎所有點通過。

**Q: 為什麼 RTMW3D 沒有異常判定？**
A: RTMW3D 模型的輸出格式與 MediaPipe 不同，需另行實現。目前未做。

---

## 相關代碼參考

| 模組 | 位置 | 功能 |
|------|------|------|
| `_compute_anomaly_mask` | `video_file_processor.py:41~103` | 逐幀異常判定核心 |
| `_run_pass1` | `video_file_processor.py:104~199` | 自適應門檻計算 |
| `_joints_reliable` | `rula_calculator.py:12~18` | 檢查關節可靠性 |
| `rula_score_side` | `rula_calculator.py:53~320` | 角度計算（使用異常標記） |
| `_show_frame` | `result_window.py:860~945` | 結果視窗顯示（標記異常點） |

---

## 更新紀錄

- **2026-01-15**: 文檔初稿，涵蓋 Pass 1 與 Pass 2 流程
- **計劃中**: 實現 RTMW3D 異常判定
- **計劃中**: 支援自訂異常判定參數的 UI

---

## 計算範例（完整步驟）

下面用一組數值演示 Pass 1 與 Pass 2 中 `raw_speed` / `speed_ratio` 的計算，以及 `prev_reliable` 在判定中的作用。

情境說明（單一關節示例）：
- 關節索引：`15`（假設為左手腕）
- 兩個採樣時間點：上一個可靠幀位置 `p_{t-1}` 與當前採樣幀位置 `p_t`，座標為世界座標（m）。
- 取樣間隔：`frame_interval = 10`，影片 `fps = 30` → `dt = frame_interval / fps = 10 / 30 = 0.3333 s`。

1) Pass 1（收集 raw speed）

- 假設在 Pass 1 掃描時，上一個可靠幀的位置為：

```
p_{t-1} = [0.200, 0.300, 0.500]  # x,y,z (m)
```

- 當前檢測到的位置為：

```
p_t = [0.250, 0.330, 0.480]      # x,y,z (m)
```

- 計算位移（jump）：

$$
jump = \sqrt{(0.250-0.200)^2 + (0.330-0.300)^2 + (0.480-0.500)^2}
         = \sqrt{0.05^2 + 0.03^2 + (-0.02)^2}
         \approx 0.0616\ \text{m}
$$

- 計算 raw speed：

$$
raw\_speed = \frac{jump}{dt} = \frac{0.0616}{0.3333} \approx 0.1848\ \text{m/s}
$$

- 在 Pass 1 我們把 `raw_speed` 放入群組暫存：

```python
group_raw_speeds['hand'].append(0.1848)
```

（注意：此時還不除以 `body_scale_ref`，僅收集世界座標下的速度樣本）

2) Pass 1 掃描結束 → 計算 `body_scale_ref`

- 假設掃描期間收集到多筆肩寬樣本，最後計算中位數：

```
body_scale_ref = 0.400  # m
```

- 現在把 `group_raw_speeds['hand']` 轉為 `speed_ratio`：

$$
speed\_ratio = \frac{raw\_speed}{body\_scale\_ref} = \frac{0.1848}{0.4} \approx 0.462
$$

3) 比對門檻（假設群組門檻為固定示例）

- 假設最終計算出該群組的門檻為 `th_low = 3.0`、`th_high = 10.0`，則：

```
speed_ratio = 0.462 < th_low(3.0)  => 非異常
```

4) Pass 2（逐幀判定）與 `prev_reliable` 的用途

- 在 `_compute_anomaly_mask()` 中，`prev_reliable[i]` 儲存的是「上一個被判定為可靠的該關節 3D 座標」。
- 計算 `speed_ratio` 時使用的 `prev` 即為 `prev_reliable[i]`：

```python
prev = prev_reliable[i]
if prev is not None and dt > 1e-9 and body_scale > 1e-6:
        jump = math.sqrt((x - prev[0])**2 + (y - prev[1])**2 + (z - prev[2])**2)
        speed_ratio = (jump / dt) / body_scale
        # 比對 th_high / th_low 決定 reason
else:
        # 無 prev -> 跳過速度檢查（不以速度判定為異常）
```

- 因此 `prev_reliable` 能避免使用上一個本身已被標記為異常的幀位置來計算速度（避免污染）。若 `prev_reliable[i] is None`，速度檢查會被跳過，僅以 visibility 判定。示例：

    - 若當前 `visibility = 0.45`（< 0.5） → 直接 `low_visibility`。
    - 若 `visibility = 0.75` 且計算得到 `speed_ratio = 4.2`，且 `th_low = 3.0` → `low_vis_speed_jump`（因為 visibility < 0.80 且 speed_ratio > th_low）。

完整數值範例（觸發 low_vis_speed_jump）：

- 假設 `jump = 0.45 m`、`dt = 0.3333 s` → `raw_speed = 1.35 m/s`。
- `body_scale_ref = 0.3 m` → `speed_ratio = 1.35 / 0.3 = 4.5`。
- `visibility = 0.75 (< 0.80)`，`th_low = 3.0` → 判定 `low_vis_speed_jump`。

---

