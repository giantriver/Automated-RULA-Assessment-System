# 關節異常判定（Joint Anomaly Detection）

本文件說明目前專案中關節異常判定的實作邏輯、資料流與資料結構。

## 1. 目的與範圍

- 目的：過濾不可靠關節（低置信度、瞬間跳躍、遮擋重現），提升 RULA 角度計算穩定性。
- 目前僅 MediaPipe 後端有完整異常判定流程。
- RTMW3D 目前沒有對應的逐點異常判定與 UI 標記。

## 2. 速度指標定義

異常判定使用正規化後的 3D 速度比 `speed_ratio`，不是像素速度。

1) 關節位移

$$
jump = \sqrt{(x_t-x_{t-1})^2 + (y_t-y_{t-1})^2 + (z_t-z_{t-1})^2}
$$

2) 時間差

$$
dt = \frac{\Delta frame}{fps}
$$

3) 正規化速度比

$$
speed\_ratio = \frac{jump / dt}{body\_scale}
$$

說明：
- `jump / dt` 代表世界座標下瞬時速度。
- `body_scale` 使用肩寬，讓不同身形可用同一門檻比較。

## 3. 統一後的 prev_reliable 資料型態

Pass 1 與 Pass 2 已統一使用同一型態：

```python
prev_reliable: list[dict | None]
```

每個關節位置 `i` 的資料：

```python
prev_reliable[i] = {
    "pos": [x, y, z],
    "frame_idx": frame_idx,
}
```

若尚無可靠歷史點則為 `None`。

## 4. 全流程概觀

```text
影片輸入
  -> Pass 1（MediaPipe）：收集速度分布，建立群組門檻
  -> Pass 2（MediaPipe）：逐幀判定關節是否可靠
  -> angle_calc：只用可靠關節計算角度
  -> 記錄輸出（joint_anomaly / joint_anomaly_detail）
  -> UI 顯示與匯出
```

## 5. Pass 1：建立自適應速度門檻

函式：`_run_pass1`

重點：
- 抽樣掃描影片（`frame_interval`）。
- 只納入高可見度點（`visibility >= vis_high`）。
- 使用統一型態的 `prev_reliable[i]` 取出 `pos` 與 `frame_idx` 計算 `gap_seconds`。
- 若 `gap_seconds > 1.0` 秒，略過該筆速度樣本，避免污染分布。

速度樣本收集：
- 先收集 `raw_speed = jump / gap_seconds`。
- 掃描結束後，以 `body_scale_ref`（肩寬中位數）轉成 `speed_ratio`。

門檻計算：
- 樣本數 >= 20 時，用 MAD（robust 統計）推估單一門檻 `th_speed`。
- 樣本數 < 20 時，不建立門檻，後續會略過速度判定。

## 6. Pass 2：逐幀異常判定

函式：`_compute_anomaly_mask`

判定步驟（每個關節）：
1. `vis < 0.50` -> `low_visibility`。
2. 否則，若有 `prev_reliable[i]`，從其中讀出 `pos/frame_idx` 計算 `gap_seconds`。
3. 若該關節屬於群組且 `gap_seconds <= 1.0`，並且門檻存在與尺度有效，才計算 `speed_ratio`。
4. 若有 `th_speed`：
    - `speed_ratio > th_speed` -> `speed_jump`
5. 若本幀可靠，更新：

```python
new_prev[i] = {"pos": [x, y, z], "frame_idx": current_frame_idx}
```

## 7. 長 gap 保守策略

規則：`gap_seconds > 1.0` 秒時，略過速度一致性檢查。

原因：
- 長時間未觀測後重現的位移，不一定是異常跳躍。
- 這個策略可同時降低 Pass 1 分布污染與 Pass 2 誤判。

## 8. 常數與群組

主要常數：

```python
_ANOM_VIS_TH = 0.50
_ANOM_MAX_GAP_SECONDS = 1.0
```

關節群組：
- trunk: 11, 12, 23, 24
- head: 0, 7, 8
- arm: 13, 14
- hand: 15, 16, 17, 18, 19, 20

## 9. 輸出資料結構

每筆 frame record 會包含：

```python
{
    "joint_anomaly": list[bool] | None,
    "joint_anomaly_detail": list[dict | None] | None,
    "joint_group_thresholds": dict | None,
}
```

`joint_anomaly_detail` 的典型內容：

```python
{
    "reason": "low_visibility" | "speed_jump",
    "visibility": float,
    "speed_ratio": float | None,
    "speed_checked": bool,
    "th_speed": float | None,
}
```

補充說明：
- 若 `speed_checked` 為 False，代表該幀略過速度判定（例如門檻不存在或無可用前一點），UI 會顯示 `N/A`。
- `th_speed` 為該關節所屬群組的速度門檻，若無門檻則為 None。

## 10. 與角度計算的關係

- `angle_calc` 會檢查關節是否可靠。
- 任一必要關節異常，該角度輸出為 `NULL`（並依既有策略做回退）。

## 11. 常見問題

Q: 為什麼角度有時是 `NULL`？
A: 該角度所需關節至少一個被判定為異常或缺失。

Q: 可以暫時放寬異常判定嗎？
A: 可調整常數，例如降低可見度門檻或提高速度門檻，但建議搭配驗證資料。

Q: 為什麼 RTMW3D 沒有同樣標記？
A: 目前尚未為 RTMW3D 實作對應的逐點異常判定流程。

## 12. 相關程式位置

- `src/rula_realtime_app/core/video_file_processor.py`：Pass 1 與 Pass 2 主邏輯
- `src/rula_realtime_app/core/rula_calculator.py`：關節可靠性檢查與角度計算
- `src/rula_realtime_app/ui/result_window.py`：MediaPipe 異常點視覺標記

