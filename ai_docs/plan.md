第一版先做：

> **visibility-based filtering + adaptive normalized velocity outlier detection**

目標是：

> 判斷「某個關節點在某一幀是否可靠」，避免錯誤骨架直接進入 RULA 計分。

---

# 第一版方法總覽

整體流程建議做成 **Two-pass**：

```text
Pass 1：先掃過整支影片，建立各關節群組的正常速度分布
Pass 2：根據 visibility + 自適應速度門檻，判斷每個關節點是否可靠
```

這樣做的好處是：

```text
不用事先為每個關節手動設定 threshold
每支工作影片都可以根據自己的動作速度調整門檻
可抓出部分 confidence 高但位置突然跳動的錯誤關節
```

---

# 1. 輸入資料

每一幀需要有：

```python
frame_idx
time_sec
joint_name
x, y, z
visibility
```

如果你目前是用 MediaPipe world landmarks 或 RTMPose 3D skeleton，建議優先用 3D 座標算速度。

如果只有 2D 影像座標，也可以做，但一定要除以人體尺度，例如肩寬，避免不同距離、不同畫面大小造成影響。

---

# 2. 計算人體尺度 body_scale_ref

建議第一版使用：

```text
body_scale_ref = 肩寬
```

也就是：

```python
shoulder_width = distance(left_shoulder, right_shoulder)
```

但不要每一幀都直接用當下肩寬，因為肩膀也可能被誤判。

建議改成整支影片的穩定肩寬中位數：

```python
body_scale_ref = median(valid_shoulder_widths)
```

其中 `valid_shoulder_widths` 只使用：

```text
left_shoulder visibility >= 0.8
right_shoulder visibility >= 0.8
shoulder_width > 0
```

這樣會比較穩。

---

# 3. 計算 normalized joint velocity

對每個關節點計算：

```python
dt = (curr_frame_idx - prev_frame_idx) / fps
jump = distance(curr_joint, prev_joint)
speed_ratio = (jump / dt) / body_scale_ref
```

意思是：

```text
這個關節點每秒移動了多少個肩寬
```

例如：

```text
speed_ratio = 1.0
代表每秒移動約 1 個肩寬

speed_ratio = 5.0
代表每秒移動約 5 個肩寬，可能偏異常
```

---

# 4. Pass 1：建立影片內速度分布

第一輪先不要判斷 unreliable，而是先收集「比較可信」的速度樣本。

建議只收集：

```python
prev_visibility >= 0.8
curr_visibility >= 0.8
body_scale_ref 有效
dt > 0
jump 有效
```

不要把 visibility 低的點拿來建立速度分布，否則遮擋錯誤會污染 threshold。

---

## 關節分組

第一版不建議每個關節都單獨估 threshold，因為有些關節樣本數可能不夠。

建議用「關節群組」：

```python
JOINT_GROUPS = {
    "trunk": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "head": ["nose", "left_ear", "right_ear"],
    "arm": ["left_elbow", "right_elbow"],
    "hand": ["left_wrist", "right_wrist", "left_index", "right_index", "left_pinky", "right_pinky"]
}
```

大致邏輯是：

```text
trunk 通常速度較慢
head 速度中等
arm 速度中等偏快
hand 速度最快，也最容易被誤判
```

---

# 5. 用 robust statistics 自動估 threshold

對每個 joint group 的 `speed_ratio` 分布計算：

```python
median_v = median(speed_ratios)
mad_v = median(abs(speed_ratios - median_v))
robust_std = 1.4826 * mad_v
```

然後設定：

```python
SPEED_TH_LOW = median_v + 3 * robust_std
SPEED_TH_HIGH = median_v + 5 * robust_std
```

可以理解成：

```text
SPEED_TH_LOW：偏異常
SPEED_TH_HIGH：明顯異常
```

如果樣本太少或 `MAD = 0`，可以退回用百分位數：

```python
SPEED_TH_LOW = 95th percentile
SPEED_TH_HIGH = 99th percentile
```

---

# 6. Pass 2：正式判斷每個關節是否 reliable

第二輪才正式判斷。

判斷邏輯建議如下：

```python
if visibility < VIS_TH:
    unreliable = True
    reason = "low_visibility"

elif speed_ratio > SPEED_TH_HIGH[group]:
    unreliable = True
    reason = "extreme_velocity_anomaly"

elif visibility < VIS_MID_TH and speed_ratio > SPEED_TH_LOW[group]:
    unreliable = True
    reason = "medium_visibility_with_velocity_anomaly"

else:
    unreliable = False
    reason = "reliable"
```

參數第一版可以先用：

```python
VIS_TH = 0.5
VIS_MID_TH = 0.8
```

---

# 7. 速度要和「上一個可靠點」比較

這點很重要。

不要永遠和上一幀比較，因為上一幀本身可能已經錯了。

建議第二輪時，每個關節都記錄：

```python
prev_reliable_joint
prev_reliable_frame_idx
```

然後用目前點和上一個可靠點比較：

```python
dt = (curr_frame_idx - prev_reliable_frame_idx) / fps
jump = distance(curr_joint, prev_reliable_joint)
speed_ratio = (jump / dt) / body_scale_ref
```

如果目前點被判定可靠，才更新 `prev_reliable_joint`。

如果目前點被判定不可靠，就不要更新。

這樣可以避免錯誤點連續污染後面的判斷。

---

# 8. 建議輸出欄位

每個關節點可以輸出：

```text
frame_idx
time_sec
joint_name
x
y
z
visibility
speed_ratio
speed_th_low
speed_th_high
reliable
unreliable_reason
```

例如：

| frame | joint       | visibility | speed_ratio | reliable | reason                   |
| ----: | ----------- | ---------: | ----------: | -------- | ------------------------ |
|   120 | right_wrist |       0.93 |         8.2 | False    | extreme_velocity_anomaly |
|   121 | right_wrist |       0.42 |         1.1 | False    | low_visibility           |
|   122 | right_wrist |       0.88 |         1.5 | True     | reliable                 |

這樣之後 debug 會很清楚。

---

# 9. RULA 計分時的使用方式

關節 reliability 做完後，不要直接只看單一關節，要進一步判斷「角度是否可靠」。

例如 elbow angle 需要：

```text
shoulder
elbow
wrist
```

只要其中一個不可靠：

```python
elbow_angle_reliable = False
```

這一幀的 elbow angle 就不要直接拿來算 RULA。

可以這樣做：

```python
if not elbow_angle_reliable:
    elbow_score = previous_reliable_elbow_score
    reason = "use_previous_reliable_score"
else:
    elbow_score = calculate_elbow_score()
```

第一版可以先用：

```text
短時間不可靠：沿用上一個可靠角度或分數
長時間不可靠：標記為 unavailable
```

不要硬算，因為你的目標是避免錯誤骨架造成錯誤 RULA 分數。

---

# 10. 第一版完整流程

可以整理成這樣：

```text
Step 1：讀取影片骨架資料

Step 2：計算整支影片的 body_scale_ref，例如穩定肩寬中位數

Step 3：Pass 1
        收集 visibility >= 0.8 的關節速度
        依關節群組建立 speed_ratio 分布

Step 4：用 robust statistics 計算每個群組的
        SPEED_TH_LOW
        SPEED_TH_HIGH

Step 5：Pass 2
        對每一幀、每個關節進行 reliable / unreliable 判斷

Step 6：根據關節可靠性，判斷 RULA 所需角度是否可靠

Step 7：可靠角度才計分
        不可靠角度則補值、沿用上一幀，或標記 unavailable
```

---

# 11. 方法名稱建議

你可以把第一版命名為：

> **Adaptive Visibility-Velocity Joint Reliability Detection**

中文可以寫：

> **結合可見度與影片內自適應速度異常偵測之關節可靠性判定方法**

或更簡潔：

> **基於 visibility 與自適應 normalized velocity 的關節可靠性判定方法**

---

# 12. 論文寫法版本

可以寫成：

> 本研究首先利用姿勢估計模型所輸出的 visibility 進行初步關節可靠性判定。當關節點 visibility 低於 0.5 時，該關節點被視為不可靠。為了進一步處理 visibility 較高但位置可能出現錯誤跳動的情況，本研究計算相鄰分析幀之 normalized joint velocity。該速度以關節位移除以時間差後，再根據人體尺度參考值進行正規化，以降低受試者距離鏡頭或影像尺度差異造成的影響。
>
> 此外，本研究不採用固定速度門檻，而是根據每支工作影片中高 visibility 關節點的速度分布，使用 robust statistics 自動估計該影片內的速度異常門檻。當關節點之 normalized joint velocity 超過自適應門檻時，即使該點 visibility 較高，仍會被判定為不可靠。最後，若 RULA 角度計算所需之任一關節點被判定為不可靠，該角度將不直接用於 RULA 計分，以避免錯誤骨架造成錯誤的人因工程風險評估結果。
