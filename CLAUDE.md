# CLAUDE.md — OpenTSLM FRDA mFARS Prediction

## Project goal

Fine-tune a time-series LLM to predict **mFARS** (Modified Friedreich Ataxia Rating Scale, 0–93) from raw 6-axis IMU sensor recordings (accelerometer + gyroscope) captured by three wearable AIM devices:

| Device | Task | Sampling rate | Channels |
|--------|------|--------------|----------|
| AIM-C (cup) | Simulated drinking, dominant hand, 5 trials | 100 Hz | 6-axis IMU + force (7 total) |
| AIM-S (spoon) | Simulated feeding, dominant hand, 5 trials | 100 Hz | 6-axis IMU (force=0 padded) |
| AIM-P (pendant) | 30-second quiet standing | 50 Hz → upsampled to 100 Hz | 6-axis IMU (force=0 padded) |

mFARS breakdown: **Total = Upper Limb (UL) + Lower Limb (LL) + Upright Stability (US)**. Cup and spoon capture UL; pendant captures US. This matters when choosing prediction targets.

---

## Canonical data paths

| File | Description |
|------|-------------|
| `/Users/bnguyen/projects/d/aims-exp/ml-aims/data/master_adults.csv` | 485 recordings × 43 columns. One row per JSON file. Key columns: `filename`, `participant_id`, `device_type` (cup/spoon/pendant), `mfars_total`, `mfars_upper_limb`, `mfars_lower_limb`, `mfars_upright_stability`, `staging`, `age`, `disease_duration`, `gaa_allele_1`, `gaa_allele_2`, `dominant_hand`, `sex` |
| `/Users/bnguyen/projects/d/aims-exp/ml-aims/results/split_adults.csv` | 127 participants × 2 columns: `participant_id`, `split` (train/val/test). Counts: train=85, val=17, test=25 |
| `/home/ben/data/imu_biokin_data/s3_backup_30Sep2025/` | Raw JSON recordings (on VM `ben`, ~892 MB, not committed) |
| `/home/ben/pretrained/gemma-3-270m-it` | Pretrained Gemma-3 270M weights (on VM `ben`) |

**Split rule:** Train and val only for all model development and hyperparameter selection. Test set is strictly held out — never touch it until final evaluation.

Recordings per device per split (from master_adults + split_adults):

| Split | Cup | Spoon | Pendant | Total |
|-------|-----|-------|---------|-------|
| Train | 143 | 132 | 77 | 352 |
| Val   | 22  | 19   | 6  | 47  |
| Test  | 38  | 34   | 14 | 86  |

---

## Current code structure

```
src/opentslm/
├── time_series_datasets/frda/
│   ├── frda_loader.py          Signal loading, preprocessing, windowing, split manifest creation
│   └── FRDAMFARSDataset.py     PyTorch Dataset wrapping frda_loader for OpenTSLM-SP format
├── model/
│   ├── llm/
│   │   └── OpenTSLMRegressionSP.py   Gemma-3 backbone + MLP regression head
│   └── regression/
│       └── ridge_window.py     Handcrafted-feature ridge regression baseline
train_mfars_regression.py       Training entry point (--model-backend ridge_window | opentslm)
test_mfars_regression.py        Evaluation on test set
predict_mfars_from_json.py      Single-file inference
```

### Key issue with the current data pipeline

The training scripts currently use `create_split_manifest()` from `frda_loader.py`, which **generates its own patient split** from a metadata CSV + JSON root path. This is **not** using `split_adults.csv`. Any new work must be wired to respect the canonical split at `/Users/bnguyen/projects/d/aims-exp/ml-aims/results/split_adults.csv` and load recordings from `master_adults.csv`.

The `FRDAMFARSDataset` reads from a JSON manifest produced by `create_split_manifest()`. The correct approach is to either:
1. Write a loader that reads directly from `master_adults.csv` + `split_adults.csv`, or
2. Pre-generate a manifest from those two files and pass it via `--split-manifest`

---

## Signal preprocessing (implemented in frda_loader.py)

1. Parse JSON `record` field (handles 4 formats: list-of-lists 8-col, 9-col, JSON-encoded string, dict-of-dicts)
2. Extract 7 channels: `[AccX, AccY, AccZ, GyrX, GyrY, GyrZ, Force]` — Force=0 for spoon/pendant
3. AIM-P (pendant): upsample 50 Hz → 100 Hz via linear interpolation
4. Median filter per channel (kernel=5) to remove electronic noise
5. Z-score normalize per channel
6. Window into 3000-sample (30 s) windows with 1500-sample (50%) stride; zero/reflect-pad if shorter

---

## Model architecture (OpenTSLMRegressionSP)

- **Backbone:** Gemma-3-270M-it (frozen or partially fine-tuned)
- **Encoder:** `TransformerCNNEncoder` patches the [T=3000, C=7] window into patch tokens (patch_size=50 → 60 patches)
- **Regression head:** `Linear(hidden_size, 512) → ReLU → Dropout(0.1) → Linear(512, 1)`
- **Text prompt:** Per-channel mean/std string prepended as context, e.g. `"Accelerometer X-axis, mean=0.12, std=0.34:"`

---

## Current results summary

Best performing approach: **ridge regression on handcrafted window features** (not the LLM backbone).

| Run | Model | Window R² | File R² | Pearson r (file) |
|-----|-------|-----------|---------|-----------------|
| ridge_ensemble_run5_alpha1 | Ridge+HGBR blend | 0.60 | 0.40 | 0.77 |
| full_gpu_run | Gemma-3 end-to-end | -0.82 | -1.91 | — |

The LLM backbone currently hurts performance. The ridge baseline on handcrafted features (mean, std, percentiles, first-difference) outperforms full fine-tuning by a large margin, likely because ~350 training windows is too small to fine-tune 270M parameters.

---

## Open questions / investigation priorities

1. **Is the data pipeline wired correctly?** Verify `FRDAMFARSDataset` + `train_mfars_regression.py` actually use `master_adults.csv` + `split_adults.csv` (or a manifest derived from them), not a re-generated split.

2. **Why does the LLM backbone fail?** Options: (a) full fine-tuning collapses pretrained weights with this little data, (b) text prompts with mean/std carry no useful clinical signal, (c) tokenisation mismatch — LLM tokenizer sees numbers digit-by-digit.

3. **Smaller backbone candidates** (if Gemma-3-270M is too large for this dataset size):
   - `Qwen2.5-0.5B` (~500M but very well pretrained on tabular/numeric text)
   - `SmolLM2-135M` (HuggingFaceTB/SmolLM2-135M — 135M, fits in small GPU memory)
   - Freeze backbone entirely, train only the patch encoder + regression head (effectively a transformer encoder approach, not generative LLM)

4. **TTCA-style cross-modal fusion:** Freeze LLM, use it to encode a clinical context string (age, staging, GAA alleles, device type), then cross-attend that embedding into the learned patch token sequence → regression head. Time-series patches as Q, clinical text embedding as K/V.

---

## VM setup

SSH host: `ben`

```bash
# Data
/home/ben/data/imu_biokin_data/s3_backup_30Sep2025/   # raw JSON recordings

# Pretrained models
/home/ben/pretrained/gemma-3-270m-it

# Example train command (ridge baseline, correct split)
python train_mfars_regression.py \
  --metadata-csv /Users/bnguyen/projects/d/aims-exp/ml-aims/data/master_adults.csv \
  --json-root /home/ben/data/imu_biokin_data/s3_backup_30Sep2025/ \
  --split-manifest <path-to-manifest-derived-from-split_adults.csv> \
  --model-backend ridge_window \
  --output-dir results/mfars_regression/my_run
```
