# FRDA mFARS Prediction — Experiment Log

## Goal

Predict mFARS total score (0–93) from raw 6-axis IMU recordings collected by three AIM wearable devices (cup, spoon, pendant). The question driving these experiments is whether an LLM-based time-series model (OpenTSLM) can match or beat a handcrafted ridge regression baseline when trained on a small clinical dataset (~350 recordings from 85 participants).

---

## Data

**Metadata:** `/Users/bnguyen/projects/d/aims-exp/ml-aims/data/master_adults.csv`  
485 recordings across 127 adult participants. One row per JSON file. Key columns: `filename`, `participant_id`, `device_type` (cup/spoon/pendant), `mfars_total`, `mfars_upper_limb`, `mfars_lower_limb`, `mfars_upright_stability`, `staging`, `age`, `disease_duration`, `gaa_allele_1`, `gaa_allele_2`.

**Canonical split:** `/Users/bnguyen/projects/d/aims-exp/ml-aims/results/split_adults.csv`  
Participant-level split (no data leakage): 85 train / 17 val / 25 test. Translates to 352 / 47 / 86 recordings respectively across all three devices.

**Raw recordings (on VM ben):** `/home/ben/data/imu_biokin_data/s3_backup_30Sep2025/`  
JSON files, each containing a `record` array of IMU rows.

**Target:** `mfars_total` (the full 0–93 composite score).

---

## Signal Preprocessing

All signals go through the same pipeline regardless of model:

1. Parse JSON `record` field (handles four formats: list-of-lists 8-col, 9-col, JSON-encoded string, dict-of-dicts).
2. Extract 7 channels: `[AccX, AccY, AccZ, GyrX, GyrY, GyrZ, Force]`. Force is zeroed for spoon and pendant.
3. AIM-P (pendant, 50 Hz) is linearly upsampled to 100 Hz.
4. Median filter per channel (kernel size 5) to remove electronic noise.
5. Z-score normalisation per channel (for LLM runs only; ridge uses raw values).
6. Windowing: 3000-sample windows (30 s at 100 Hz), 1500-sample (50%) stride. Short recordings are reflect/zero-padded to exactly one window.

---

## Models

### Ridge baseline (RidgeWindowRegressor + HGBR blend)

Extracts 14 handcrafted statistical features per channel per window (mean, std, median, p10, p25, p75, p90, IQR, abs-mean, RMS, mean-abs-diff, std-diff, skewness, kurtosis) plus a test-ID indicator (which device), giving a 99-dimensional feature vector. Trains two ridge models: one at window level and one at file level (mean-pooled windows). Blends their predictions. Optionally adds a HistGradientBoostingRegressor trained on richer per-file aggregated features (mean/std/min/max/median across windows), and blends that in too. Hyperparameter search is over alpha ∈ {0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000}.

### OpenTSLM (frozen LLM backbone + encoder + regression head)

A pretrained LLM backbone is kept frozen throughout. The time-series is patched into tokens by a TransformerCNNEncoder (patch_size samples → one token embedding). An MLPProjector maps encoder output into the LLM's hidden space. The LLM runs a forward pass and its last hidden layer is mean-pooled (masked) to produce a single vector, which feeds into a regression head: `Linear(hidden_size, 512) → ReLU → Dropout(0.1) → Linear(512, 1)`. Only the encoder, projector, and regression head are trained. Loss is Huber (smooth L1, β=1). Three backbone sizes were tested:

- **Gemma-3-270M** (`/home/ben/pretrained/gemma-3-270m-it`), hidden size 640
- **Qwen2.5-0.5B** (`/home/ben/pretrained/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit`), hidden size 896
- **Llama-3.2-1B** (`/home/ben/pretrained/Llama-3.2-1B`), hidden size 2048

All LLM runs: 60 epochs max, early stopping patience 15, weight decay 1e-2, grad clip 1.0, linear warmup 3%, target normalisation on.

---

## Evaluation

Two levels are reported for every run:

- **Window-level:** each 30-second window is treated as an independent prediction.
- **File-level:** all windows for a recording are averaged into one prediction, then scored against the file's true mFARS. This is the primary metric because clinically one score per recording is what matters.

Metrics: R², MAE, RMSE, Pearson r. The experiment log records val (file-level R², MAE) and test (file-level R², MAE, RMSE, Pearson r).

**Important:** val metrics were used for all model selection and hyperparameter decisions. Test was only evaluated after training finished and is strictly held out.

---

## Infrastructure

All experiments ran on VM **ben** (RTX 4080, 16 GB VRAM, CUDA 13, PyTorch 2.6, transformers 4.53, conda env `llm`).

The experiment runner `run_frda_experiments.sh` handles the full train→test→log cycle. It generates a single shared split manifest once, reuses it across all runs, skips completed runs on restart, aborts if CUDA is unavailable before starting any LLM run, and appends each completed run to `results/experiments_log.jsonl`. Results can be viewed with:

```bash
ssh ben 'cd /home/ben/projects/OpenTSLM && python3 show_results.py'
```

---

## Results

All metrics are file-level (per recording, not per window).

| Run | val R² | test R² | test Pearson r |
|-----|--------|---------|----------------|
| ridge_alpha_grid_default | 0.4684 | **0.5428** | **0.7804** |
| ridge_no_hgbr | 0.2471 | 0.2569 | 0.5349 |
| ridge_no_test_id | 0.4683 | 0.5428 | 0.7805 |
| opentslm_gemma270m_lr2e4 | 0.1676 | 0.3232 | 0.7367 |
| opentslm_gemma270m_lr1e3 | 0.3200 | **0.5440** | 0.7724 |
| opentslm_gemma270m_patch100 | 0.2463 | 0.3378 | 0.6326 |
| opentslm_qwen05b_lr2e4 | 0.3586 | 0.5087 | 0.7697 |
| opentslm_qwen05b_lr1e3 | −0.0052 | −0.0424 | −0.1282 |
| opentslm_llama1b_lr2e4 | 0.3738 | 0.3946 | 0.6340 |
| opentslm_llama1b_lr1e3 | 0.2019 | 0.1819 | 0.5840 |

---

## What the Results Tell Us

**Ridge with HGBR blend is the most reliable model.** It achieves test R² = 0.54 and Pearson r = 0.78, and its val performance (0.47) tracks the test performance well — meaning it generalises predictably. The HGBR blend is critical: removing it drops test R² to 0.26. Including the test-ID feature (which device recorded the signal) does not change performance materially (ridge_no_test_id is nearly identical), which is a mild surprise given how different cup/spoon/pendant recordings are.

**Gemma-3-270M at lr=1e-3 matches ridge on test** (0.5440 vs 0.5428), but its val score is lower (0.32 vs 0.47). This means if you used val to pick the best model, you would not choose it — you'd pick ridge. The result is encouraging but not yet actionable for model selection purposes.

**Qwen2.5-0.5B at lr=2e-4 is a reasonable second LLM option** (test R² = 0.51, Pearson 0.77). However lr=1e-3 completely collapses it (negative R²), making it sensitive to learning rate.

**Llama-3.2-1B underperforms despite having the largest backbone.** Test R² of 0.39 at lr=2e-4, and collapse at lr=1e-3. A larger hidden dimension does not help when the dataset is this small and the backbone is frozen — the encoder and projector have more parameters to fit against fewer signals.

**Larger patches hurt.** Gemma with patch_size=100 (30 patches per window) does worse than patch_size=50 (60 patches), suggesting the temporal resolution from finer patches is important.

**The val→test gap is the central problem.** For LLM runs the val R² is systematically lower than test R², which is unusual. This likely reflects that the 17-participant val set has a different mFARS distribution than the 25-participant test set (val has only 6 pendant recordings vs 14 in test), causing noisy val estimates. With only 47 val recordings the signal-to-noise ratio for hyperparameter selection is poor.

---

## What to Try Next

1. **Cross-validate on train+val instead of val-only selection.** With 102 train+val participants, a 5-fold CV would give much more stable model selection than a 17-participant holdout. The test set stays locked.

2. **Stratify by device in the val set.** Currently val has only 6 pendant recordings. Ensure the split preserves device proportions.

3. **Try a linear probe on frozen LLM embeddings.** Rather than training the encoder from scratch, feed preprocessed windows directly through the frozen LLM (as patch tokens) and fit a ridge on the pooled embeddings. This isolates whether the LLM's representation is useful at all without worrying about encoder training.

4. **Clinical context fusion (TTCA-style).** Freeze the LLM, use it to encode a text string of clinical covariates (age, staging, GAA repeat lengths, device type), then cross-attend the clinical embedding into the patch token sequence before the regression head. The hypothesis is that mFARS is partly predicted by static patient characteristics the sensor alone can't see.

5. **Device-specific models.** Train and evaluate separately on cup, spoon, and pendant rather than pooling all devices. The mFARS subscores map naturally: cup+spoon → upper limb, pendant → upright stability.
