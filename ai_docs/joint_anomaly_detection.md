# 關節異常判定（Joint Anomaly Detection）

本文件說明目前專案中關節異常判定的實作邏輯、資料流與資料結構。

## 1. 目的與範圍

- 目的：過濾不可靠關節（低置信度、瞬間跳躍、遮擋重現），提升 RULA 角度計算穩定性。
- 目前僅 MediaPipe 後端有完整異常判定流程。
- RTMW3D 目前沒有對應的逐點異常判定與 UI 標記。

## 2. 速度指標定義

異常判定使用正規化後的 3D 速度比 `speed_ratio`（單位為 1/s），讓不同身形和不同取樣間隔下可以用相同尺度比較。

1) 關節位移

$$
jump = \sqrt{(x_t-x_{t-1})^2 + (y_t-y_{t-1})^2 + (z_t-z_{t-1})^2}
$$

2) 取樣時間差（秒）

$$
dt = \frac{\Delta frame}{fps}
$$

3) 正規化速度比（程式內稱 `speed_ratio`，用於與閥值比較）

$$
speed\_ratio = \frac{jump / dt}{body\_scale}
$$

說明：
- `jump / dt` 代表世界座標下的瞬時速度（長度/秒）。
- `body_scale` 使用肩寬（world-space），把速度正規化成「相對於身體尺度的速度」，因此 `speed_ratio` 的單位為 1/s。

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
- 先收集 `raw_speed = jump / gap_seconds`（世界座標速度）。
- 掃描結束後，以 `body_scale_ref`（肩寬中位數）轉成 `speed_ratio`（即 `raw_speed / body_scale_ref`）。

門檻計算：
- 若樣本數 >= 20，使用中位數 + MAD（robust）估算 `adaptive_th = med + 5*robust_std`。
- 若樣本不足或 robust_std 太小，則視為無自適應門檻（Pass 2 會略過速度檢查）。

保護下限（`_ANOM_MIN_JUMP_RATIO`）：
- 為避免低動作影片造成的 threshold collapse（自適應門檻過低），程式會使用一個經驗性保護下限 `_ANOM_MIN_JUMP_RATIO`。
- 這個常數的語義是「允許的最小位移，佔身體尺度的比例（per sample interval）」；換言之，若

$$
_ANOM\_MIN\_JUMP\_RATIO = r
$$

那麼同一取樣間隔（sample interval）下允許的最大位移為

$$
jump_{max} = r \cdot body\_scale
$$

- 在程式中會把它轉成與 `speed_ratio` 同單位的阈值：

$$
min\_speed\_th = \frac{r}{sample\_dt},\quad sample\_dt = \frac{frame\_interval}{fps}
$$

- 最後取最大值：

$$
th\_speed = \begin{cases}
adaptive\_th &\text{若沒有 } min\_speed\_th\\
min\_speed\_th &\text{若沒有 } adaptive\_th\\
\max(adaptive\_th,\; min\_speed\_th) &\text{否則}
\end{cases}
$$

這樣的處理會保證：在同一個 `frame_interval` 下，`_ANOM_MIN_JUMP_RATIO` 直接對應「每次取樣允許的最大位移比例」，而 `min_speed_th` 則是把它轉成每秒的比較單位，以匹配 `speed_ratio`。

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

新增常數：

```python
_ANOM_MIN_JUMP_RATIO  # e.g. 0.15 ~ 0.30, 無因次比例
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

Q: `_ANOM_MIN_JUMP_RATIO` 的實際意義是什麼？
A: 它代表「在單一取樣間隔內允許的最大位移，佔身體尺度（肩寬）的比例 r」。程式會把它轉成每秒的 `min_speed_th = r / sample_dt`，再與自適應門檻比較。若你把 `r` 設為 `0.3`，代表在同一個 sample interval 下，允許的最大位移為 `0.3 * body_scale`。

Q: 使用不同 `frame_interval`（取樣間隔）是否需要不同的 `r`？
A: 不需要。由於程式把 `r` 除以 `sample_dt` 以得到和 `speed_ratio` 相同單位的 `min_speed_th`，你可以在不同的 `frame_interval` 下使用同一個 `_ANOM_MIN_JUMP_RATIO`（例如 0.3）。實務上若你改變取樣間隔後仍有過多誤判，先嘗試微調 `r`，或考慮加入額外的抑制（例如需要連續 N 幀超標才視為異常或在座標上做平滑）。

Q: 範例（快速換算）
A: 假設 `fps = 30`：

- `frame_interval = 10` → `sample_dt = 10/30 ≈ 0.333s`。
    - 若 `r = 0.3`，則 `min_speed_th = 0.3 / 0.333 ≈ 0.9 (1/s)`。
    - 在此情況下，允許的最大位移（單次取樣）為 `jump_max = 0.3 * body_scale`。

- `frame_interval = 5` → `sample_dt = 5/30 ≈ 0.167s`。
    - 若 `r = 0.3`，則 `min_speed_th = 0.3 / 0.167 ≈ 1.8 (1/s)`。
    - 同樣允許的最大位移為 `0.3 * body_scale`（每次取樣）。

範例數字（假設 `body_scale = 0.4`（world units，例如 0.4 m）：

- `r = 0.3`, `frame_interval = 10` → `jump_max = 0.3 * 0.4 = 0.12 (≈12 cm)`。

注意：`body_scale` 的單位取決於 pose 檢測輸出的 world-space（MediaPipe world coordinates）。這裡的 0.4 只是示意。

## 12. 相關程式位置

- `src/rula_realtime_app/core/video_file_processor.py`：Pass 1 與 Pass 2 主邏輯
- `src/rula_realtime_app/core/rula_calculator.py`：關節可靠性檢查與角度計算
- `src/rula_realtime_app/ui/result_window.py`：MediaPipe 異常點視覺標記

