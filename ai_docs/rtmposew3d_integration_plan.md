# RTMPOseW3D Integration Plan — Joint Anomaly Detection

目標：將現有僅作用於 MediaPipe 的關節異常判定（Pass 1 / Pass 2）擴充到支援 RTMPOseW3D（程式中稱 `RTMW3D`）。

概述結論（可行性）：可行，但需要釐清 RTMPOseW3D 的 3D 座標尺度、關鍵點對應和置信度欄位；並針對該模型的噪聲特性進行閥值重新校準。

需使用者確認（快速清單）
- RTMPOseW3D 的 keypoint layout（index → 關節名稱），是否有與 MediaPipe 33 點的一一對應表？
- RTMPOseW3D 提供的 3D 座標是 world-space（米）還是歸一化座標？單位是什麼？
- 每個 keypoint 的置信度/visibility 欄位名稱與數值範圍（0..1）是什麼？
- 是否有左右手/身體點的完整集合（手指、腕/肘/肩）？若缺少手部細節，需調整群組定義。
- 現有程式對 `RTMW3D` 的處理路徑（`video_file_processor.py`）是否允許拿到 native 3D keypoints 同步供 Pass1 使用？

技術差異與影響（要注意的地方）
- 座標尺度：MediaPipe 在程式中以 world-space 值且用肩寬作為 `body_scale`；若 RTMPOseW3D 輸出是歸一化或攝影機座標，必須轉換為可比較的尺度（或改採相對尺度，例如以肩寬像素距離）。
- keypoint 集合/索引：若 layout 不同，需要寫一個 mapper，將 RTMPOseW3D 的點映射到目前 `_JOINT_GROUPS` 使用的索引（或修改群組定義以使用新索引）。
- 置信度/visibility：目前流程以 `visibility >= _ANOM_VIS_TH` 篩選樣本；請確認 RTMPOseW3D 的置信度含義與分佈，否則要調整 `_ANOM_VIS_TH`。
- 噪聲特性：不同模型的抖動/抖動頻率差異會影響 MAD 與 adaptive_th 的數值分佈，需做 Pass 1 掃描後的校準（可能調整 `_ANOM_MIN_JUMP_RATIO` 或 adaptive multiplier，如 med + 5*robust_std）。
- 關節群組：現有的群組（trunk/head/arm/hand）可能需要調整以配合 RTMPOseW3D 支援的點集合。

實作步驟（高階）
1. 探查 RTMPOseW3D 輸出（需你提供或告知）
   - 確認 `keypoints_3d` 格式、索引、confidence 字段與座標單位。
2. 建立 mapping 層
   - 在程式新增 `RTMW3D_TO_MEDIAPIPE_MAP` 或相容的 mapper，將 RTMPOseW3D 點對照到 _JOINT_GROUPS 所需的索引。
3. 一般化 Pass 1/Pass 2
   - 把 `video_file_processor.py` 中只在 `if self.backend_mode == 'MEDIAPIPE'` 的判斷改為「若後端回傳可用 3D keypoints，則執行相同流程」。
   - 在 `_run_pass1` 與 `_compute_anomaly_mask` 加入 backend-agnostic 的資料抽取介面（例如 `get_landmarks_array(backend_normalized=True)` 或在呼叫前做 mapping/normalize）。
4. 座標尺度處理
   - 若 RTMPOseW3D 給的是非 metric 單位，選擇一個 body_scale 計算方式（例如肩寬的 pixel / raw units），並在 Pass1 將 raw speeds 正規化同一尺度。
5. 閥值校準
   - 用數支 RTMPOseW3D 影片做 Pass1 收集速度分佈，計算 adaptive_th 與 robust_std，並檢視是否需要提高 `_ANOM_MIN_JUMP_RATIO`。
6. UI / 記錄
   - 確保 records 裡的 `joint_group_thresholds` 及 `joint_anomaly_detail` 包含來源後端標記，並在結果視窗顯示時對應新索引或 mapper。
7. 驗證與回歸測試
   - 建三類影片（靜態電腦作業 / 搬運 / 擦拭），用 RTMPOseW3D + MediaPipe 分別跑，同步比對異常標記率與 RULA 結果差異。

需要你協助確認的具體項目（我需要你回覆這些以便我能寫出準確的變更）：
- 請提供或貼上 RTMPOseW3D 輸出的一個 sample frame JSON（包含 keypoints 與 confidence），或告訴我該模型的官方 layout/說明連結。
- 該模型 3D 座標的單位與參考（是否為相機座標、是否為公尺或歸一化）。
- 你期望 `RTMW3D` 在 UI 裡也顯示 `joint_group_thresholds` 嗎？（我建議顯示並標註來源 backend）

風險與緩解
- 風險：座標尺度不一致導致 threshold 無法直接共用 → 緩解：統一以「肩寬」做尺度標準，或在 Pass1 改用相對 pixel-scale。
- 風險：keypoint 缺失或數量不符 → 緩解：在 mapper 實作缺點降級（若某群組無足夠點則略過速度判定）。

估計工作量（粗略）
- 探查與 mapping：0.5–1 日（需 sample 輸出）
- 修改 `video_file_processor.py` 與 mapper：1–2 日
- 校準與測試（含三類影片）：1–2 日
- 總計：約 3–5 個工作天（依你提供 sample 與回覆速度而定）

下一步（我可以代勞）
- 若你同意，我會：
  1. 準備一個 `rtmw3d_mapper.py`（或在 `video_file_processor.py` 增加 mapping 函式），
  2. 在 `_run_pass1` 中加入 backend-agnostic 的 keypoint extraction，
  3. 跑一次 Pass1 的輸出紀錄，並給你 adaptive_th / min_speed_th 的建議初值。

請回覆你能提供的 sample 輸出（或授權我去抓取）以及你希望我先做哪個步驟。謝謝。
