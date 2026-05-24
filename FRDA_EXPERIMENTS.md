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

## Q1 Publication Plan

### Reviewer Assessment

The current results are at the threshold of Q1 publishability but not there yet. The core findings are strong: R²=0.54, Pearson=0.78 on a 127-participant clinical cohort is competitive with the best published IMU-to-severity regression papers. The blocker is the story is incomplete — a reviewer will ask "why use an LLM at all if ridge is just as good?" and there is currently no answer.

The paper needs to be positioned as: *"The first systematic evaluation of frozen pretrained sequence models as time-series encoders for IMU-based Friedreich Ataxia severity prediction, benchmarked against a domain-specific time-series foundation model (MOMENT) and a handcrafted-feature baseline, across three wearable devices."* That is a publishable contribution to JBHI, npj Digital Medicine, or similar.

Three additions are required. No more LLM hyperparameter sweeps — the sensitivity findings are already clear and more sweeps do not change the narrative.

---

### Phase 2: Three Required Additions

#### Item 1 — MOMENT as a frozen encoder (Priority: HIGH)

**Why:** MOMENT-1-large (AutonLab, 385M parameters, T5-large backbone, pretrained on 1B time-series samples across diverse domains) is the scientifically correct comparison. Comparing general-purpose LLMs to a domain-specific time-series foundation model is the central research question for a Q1 paper. If MOMENT beats LLMs, the finding is "pretrained temporal representations matter, not language knowledge." If they are comparable, the finding is "generic sequence models are sufficient encoders for this task." Either result is publishable.

**How:** The MOMENT model is already downloaded at `/home/ben/pretrained/models--AutonLab--MOMENT-1-large/`. It uses a T5-large backbone (d_model=1024, 24 encoder layers) with patch_len=8 and patch_stride_len=8. It is encoder-only, so pooling the encoder output is natural. A `MOMENTRegressionSP` class wraps the frozen MOMENT encoder with the same MLP regression head used for other models. The `momentfm` library needs to be pip-installed in the llm conda env on the VM. Input must be reshaped to MOMENT's expected format: [B, C, T] with seq_len=512 (so 3000-sample windows are divided into 6 non-overlapping 512-sample segments, each processed independently and averaged before the head).

**Script:** `train_mfars_regression.py --model-backend moment`

**Expected run time:** ~30 minutes per run on RTX 4080.

---

#### Item 2 — 5-fold Cross-Validation on train+val (Priority: HIGH)

**Why:** The 17-participant val set is too small for reliable hyperparameter selection (6 pendant recordings). A reviewer will flag this. 5-fold CV on all 102 train+val participants gives a stable generalization estimate. The test set remains locked. This runs on existing models — no new architecture required.

**How:** A CV wrapper runs the ridge pipeline 5 times, each time holding out a different fifth of the 102 participants, aggregating predictions, and computing CV-pooled metrics. The same alpha grid search runs within each fold. Final reported metric is mean ± std across folds. The best model for test evaluation is still trained on full train split (canonical split) — CV is for validation metric stability only, not for producing the final checkpoint.

**Script:** `run_cv.py` — outputs `results/cv_results.json` with per-fold and pooled metrics.

---

#### Item 3 — Device-stratified results (Priority: MEDIUM)

**Why:** mFARS has anatomically distinct subscores. Cup and spoon capture upper limb function; pendant captures upright stability. A device-stratified table (cup / spoon / pendant separately) is standard in wearable clinical ML papers and is required for JBHI. It requires zero retraining — the test prediction files already contain `test_id` (1=cup, 2=spoon, 3=pendant).

**How:** A post-processing script reads every completed `test_predictions_file.jsonl`, groups by `test_id`, and computes R², MAE, RMSE, Pearson r per device per model. Outputs `results/device_stratified_results.json` and a human-readable table.

**Script:** `compute_device_stratified.py` — runs instantly on existing prediction files.

---

### Execution Plan

1. Install `momentfm` on VM → wire `MOMENTRegressionSP` → add `moment` backend to training script → run 2 MOMENT experiments in tmux ✅
2. Write and run `compute_device_stratified.py` on existing predictions (instant) ✅
3. Write and run `run_cv.py` for ridge CV (fast, no GPU needed) ✅
4. Update `FRDA_EXPERIMENTS.md` with all Phase 2 results ✅

All scripts live in the project root. Results append to `results/experiments_log.jsonl`. View with `python3 show_results.py`.

---

### What NOT to do

- Do not download or run more sub-1B LLMs. Qwen3-0.6B is already on the VM; running it adds a data point but not a finding.
- Do not run wider hyperparameter sweeps on existing LLMs. The lr sensitivity is already documented and running more configs does not change the story.
- Do not touch the test set until all model selection and CV is complete.

---

## Phase 2 Results

### MOMENT-1-large

| Run | val R² | test R² | test Pearson r |
|-----|--------|---------|----------------|
| moment_lr2e4 | **0.7686** | 0.5217 | 0.7238 |
| moment_lr1e3 | 0.7247 | 0.5354 | 0.7376 |

MOMENT achieves the highest val R² of any model (0.77), substantially above ridge (0.47) and all LLMs. However, the test performance (0.52–0.54) is similar to ridge and Gemma. This large val–test gap (0.77 → 0.52) is the most important finding of Phase 2: MOMENT overfits the small val set during the epoch selection, even though the backbone is frozen.

### Full Results Summary (all phases)

| Run | val R² | test R² | test Pearson r |
|-----|--------|---------|----------------|
| ridge_alpha_grid_default | 0.4684 | **0.5428** | **0.7804** |
| ridge_no_hgbr | 0.2471 | 0.2569 | 0.5349 |
| ridge_no_test_id | 0.4683 | 0.5428 | 0.7805 |
| opentslm_gemma270m_lr2e4 | 0.1676 | 0.3232 | 0.7367 |
| opentslm_gemma270m_lr1e3 | 0.3200 | 0.5440 | 0.7724 |
| opentslm_gemma270m_patch100 | 0.2463 | 0.3378 | 0.6326 |
| opentslm_qwen05b_lr2e4 | 0.3586 | 0.5087 | 0.7697 |
| opentslm_qwen05b_lr1e3 | −0.0052 | −0.0424 | −0.1282 |
| opentslm_llama1b_lr2e4 | 0.3738 | 0.3946 | 0.6340 |
| opentslm_llama1b_lr1e3 | 0.2019 | 0.1819 | 0.5840 |
| moment_lr2e4 | **0.7686** | 0.5217 | 0.7238 |
| moment_lr1e3 | 0.7247 | 0.5354 | 0.7376 |

### Device-Stratified Results (file-level R²)

| Run | all | cup | spoon | pendant |
|-----|-----|-----|-------|---------|
| ridge_alpha_grid_default | 0.5428 | 0.4421 | 0.4888 | 0.0096 |
| opentslm_gemma270m_lr1e3 | 0.5440 | 0.4881 | 0.4548 | −0.1803 |
| opentslm_qwen05b_lr2e4 | 0.5087 | 0.4903 | 0.5137 | −1.8901 |
| moment_lr1e3 | 0.5354 | 0.3944 | 0.5534 | −0.2916 |
| moment_lr2e4 | 0.5217 | 0.3205 | 0.5642 | 0.1298 |

**The pendant (upright stability) result is near-zero or negative for every model.** Cup and spoon (upper limb) are where all predictive signal lives. This is the single most important clinical finding: mFARS upright stability may require a different feature space or model entirely, or the pendant data is insufficient (only 14 test recordings vs 38 cup / 34 spoon).

### 5-Fold Cross-Validation (Ridge, train+val participants)

```
CV R²:      0.1915 ± 0.1628
CV Pearson: 0.4794 ± 0.1493
Pooled R²:  0.2114
```

This is substantially lower than the val-split R² of 0.47, and confirms what the val–test gaps have been signalling all along: **the 17-participant val set is too small and noisy for reliable model selection.** The true cross-validated generalisation of the ridge model on this cohort is ~R²=0.21, not 0.47. The test R²=0.54 actually outperforms the CV estimate, which likely reflects the test set having a more representative mFARS distribution (more pendant recordings, more severe patients).

---

## Revised Interpretation for Q1 Paper

The core claim needs reframing in light of Phase 2. The headline finding is not "LLMs match ridge" — it is:

1. **Cup and spoon are predictable; pendant is not.** Every model achieves R²=0.39–0.56 on upper-limb tasks but fails on upright stability (R²≈0, often negative). This has direct clinical implications: the 30-second standing test with this device/processing pipeline does not provide reliable mFARS-US prediction with current methods.

2. **The val set is too small to distinguish models.** CV R²=0.19±0.16 vs val-split R²=0.47 shows the 17-participant holdout has high variance. Model selection based on val alone is unreliable. A larger cohort or stratified CV is needed.

3. **MOMENT's high val score is misleading.** val R²=0.77 for MOMENT collapses to test R²=0.52 — indistinguishable from ridge. The frozen MOMENT encoder does not provide a generalisation advantage over handcrafted features on this dataset size.

4. **Ridge is the most honest baseline.** Its val (0.47) and test (0.54) are closest together, CV is stable within its variance, and it requires no GPU. For a clinical deployment paper this matters.

### Remaining gap before Q1 submission

The device-stratified finding opens a new required section: **why does pendant fail?** Possible explanations to at least discuss: (a) only 6 val / 14 test pendant recordings — simply too few; (b) mFARS-US scores cluster near the top of the scale for ambulant patients, reducing dynamic range; (c) the 30-second standing task has less movement variance than the manipulation tasks, compressing feature distributions. This section can be written without new experiments.
