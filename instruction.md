R# 📑 Full Project Blueprint: FRDA mFARS Prediction via OpenTSLM

## 1. Project Vision & Goal

The objective is to leverage a Time-Series Language Model (TSLM) approach—specifically **OpenTSLM** with a **Gemma-3-270m-it** backbone—to automate the assessment of **Friedreich’s Ataxia (FRDA)**. By processing raw IMU data from task-based recordings, the model should output a continuous prediction of the **mFARS score** (0–93), replacing the need for subjective clinical observation with objective, sensor-based metrics.

---

## 2. Clinical Domain & Target Labels

### A. The Disease: Friedreich’s Ataxia (FRDA)

FRDA is a progressive neurodegenerative condition. In the data, you will see signals characterized by:

* **Intention Tremor:** Shaking that increases as the participant nears a target (e.g., the mouth in the drinking task).
* **Dysmetria:** Over-shooting or under-shooting intended movements.
* **Postural Instability:** Increased "sway area" and irregular corrections during standing.

### B. The Ground Truth (Labels)

* **mFARS (Modified Friedreich Ataxia Rating Scale):** Our primary regression target.
* **0-20:** Mild impairment (ambulant).
* **20-60:** Moderate impairment (likely requires mobility aids).
* **60-93:** Severe impairment (non-ambulant/wheelchair dependent).


* **FDS (Functional Disability Scale):** A categorical secondary label.
* `FDS < 5`: Ambulant.
* `FDS >= 5`: Non-ambulant.



---

## 3. The AIM Sensor System (Data Acquisition)

Data is sourced from three specialized **AIM (Ataxia Instrumented Measurement)** devices. All use the **InvenSense ICM-20948** (6-axis IMU).

| Device | Task | Sampling Rate | Channels | Clinical Significance |
| --- | --- | --- | --- | --- |
| **AIM-C (Cup)** | Simulated Drinking | 100 Hz | 6-axis IMU + Force | Captures grip stability and upper-limb "smoothness." |
| **AIM-S (Spoon)** | Simulated Feeding | 100 Hz | 6-axis IMU | Captures coordination and multi-joint control. |
| **AIM-P (Pendant)** | 30s Quiet Stand | 50 Hz | 6-axis IMU | Captures truncal ataxia and balance strategy. |

---

## 4. Technical Implementation & Data Pipeline

### A. Directory Structure

* **Recordings:** `/home/ben/data/imu_biokin_data/s3_backup_30Sep2025/`
* **Model Weights:** `/home/ben/pretrained/gemma-3-270m-it`
* **Metadata:** A master dataframe (`resplit_train_with_adl.csv`) containing columns `file_name_01`, `file_name_02`, and `file_name_03`, which correspond to the JSON files for different tasks.

### B. The Unified 7-Channel Protocol

Because we are training **one model for all devices**, we must enforce a consistent input shape.

1. **Input Vector:** `[AccX, AccY, AccZ, GyrX, GyrY, GyrZ, Force]`
2. **Upsampling (AIM-P):** The 50Hz Pendant data must be interpolated (Linear or Cubic) to **100Hz** to ensure the "token duration" is consistent across all tasks.
3. **Force Padding:** * AIM-C: Real force values (0 to ~1023 or normalized).
* AIM-S/P: Must be a constant vector of **0s** for the 7th channel.



### C. Signal Pre-processing Logic

To ensure high-quality training, the following steps must be applied during data loading:

1. **Median Filtering:** A kernel size of 5 to remove high-frequency electronic noise without blurring the "ataxic" movement peaks.
2. **Gravity Alignment:** The sensors are pre-calibrated to 1g, but Z-score normalization should be applied per-channel to handle variation in sensor mounting and participant height/weight.
3. **Windowing:** * Target a **3000-sample window** (30 seconds at 100Hz).
* If a recording is longer, use a sliding window with 50% overlap.
* If shorter, zero-pad or reflect-pad to 3000 samples.



### D. OpenTSLM Integration

The agent must modify the OpenTSLM repository to:

1. **Patching:** Divide the 3000-sample sequence into patches (e.g., 50-100 samples per patch) to convert the continuous signal into tokens for Gemma-3.
2. **Regression Head:** Since Gemma-3 is an LLM, the `lm_head` (classification over vocabulary) should be replaced with a `RegressionHead` (MLP) consisting of:
* Linear(Hidden_Size, 512) -> ReLU -> Dropout(0.1) -> Linear(512, 1).


3. **Loss Function:** Use **Mean Squared Error (MSE)** or **Huber Loss** to handle potential outliers in the mFARS scores.

---

## 5. Summary for the Agent

"You are tasked with training a regression model using OpenTSLM. You must load JSON files from the specified backup directory, align them to a 7-channel 100Hz format using the provided `load_signals` logic, and fine-tune the Gemma-3 backbone to predict clinical mFARS scores. Focus on preserving the temporal dynamics of the IMU data while maximizing the window size to capture the full 30-second tasks."