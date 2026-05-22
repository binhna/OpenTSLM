#!/usr/bin/env bash
# FRDA mFARS experiment runner.
# Trains multiple model configurations, then evaluates each on the test split.
# All results are appended to a single JSONL log: results/experiments_log.jsonl
#
# Usage (from project root, inside a tmux session on VM ben):
#   bash run_frda_experiments.sh
#
# Paths are hard-coded for VM ben. Edit the DATA_* variables if needed.

set -uo pipefail

# ── Python path ───────────────────────────────────────────────────────────────
# Ensure 'opentslm' package is importable when running from project root.
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# ── Paths ────────────────────────────────────────────────────────────────────
METADATA_CSV="/home/ben/projects/claude/ml-aims/data/master_adults.csv"
SPLIT_ADULTS_CSV="/home/ben/projects/claude/ml-aims/results/split_adults.csv"
JSON_ROOT="/home/ben/data/imu_biokin_data/s3_backup_30Sep2025/"
PRETRAINED="/home/ben/pretrained"
RESULTS_DIR="results/mfars_experiments"
LOG_FILE="results/experiments_log.jsonl"

mkdir -p "${RESULTS_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"

# ── Shared split manifest (generated once, reused by all runs) ────────────────
SHARED_MANIFEST="${RESULTS_DIR}/shared_split_manifest.json"

# ── Helper: append one entry to the master log ───────────────────────────────
log_run() {
    local run_name="$1"
    local train_dir="$2"
    local test_dir="$3"

    python3 - "${run_name}" "${train_dir}" "${test_dir}" "${LOG_FILE}" <<'PYEOF'
import json, datetime, sys, os

run_name, train_dir, test_dir, log_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

cfg = {}
cfg_path = os.path.join(train_dir, "run_config.json")
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)

test_metrics = {}
test_path = os.path.join(test_dir, "test_metrics.json")
if os.path.exists(test_path):
    with open(test_path) as f:
        test_metrics = json.load(f)

file_m = test_metrics.get("file", {})
entry = {
    "run_name": run_name,
    "timestamp": datetime.datetime.utcnow().isoformat(),
    "train_dir": train_dir,
    "test_dir": test_dir,
    "val_file_r2": cfg.get("best_val_file_r2"),
    "val_file_mae": cfg.get("best_val_file_mae"),
    "test_file_r2": file_m.get("r2"),
    "test_file_mae": file_m.get("mae"),
    "test_file_rmse": file_m.get("rmse"),
    "test_file_pearson_r": file_m.get("pearson_r"),
    "config": cfg,
}
with open(log_file, "a") as f:
    f.write(json.dumps(entry) + "\n")
print(f"  logged: val_r2={entry['val_file_r2']}, test_r2={entry['test_file_r2']}, test_pearson={entry['test_file_pearson_r']}")
PYEOF
}

# ── Helper: train then evaluate ───────────────────────────────────────────────
run_experiment() {
    local run_name="$1"
    shift
    local train_args=("$@")

    local train_dir="${RESULTS_DIR}/${run_name}"
    local test_dir="${train_dir}/test_results"

    echo ""
    echo "════════════════════════════════════════════════════"
    echo " RUN: ${run_name}"
    echo "════════════════════════════════════════════════════"

    # Skip if already done
    if [[ -f "${train_dir}/run_config.json" && -f "${test_dir}/test_metrics.json" ]]; then
        echo "  Already complete — skipping training & test."
        log_run "${run_name}" "${train_dir}" "${test_dir}"
        return 0
    fi

    # Train
    if ! python train_mfars_regression.py \
        --metadata-csv "${METADATA_CSV}" \
        --split-adults-csv "${SPLIT_ADULTS_CSV}" \
        --json-root "${JSON_ROOT}" \
        --target mfars_total \
        --split-manifest "${SHARED_MANIFEST}" \
        --output-dir "${train_dir}" \
        "${train_args[@]}"; then
        echo "  ERROR: training failed for ${run_name} — skipping test."
        return 1
    fi

    # Evaluate on test split
    if ! python test_mfars_regression.py \
        --checkpoint "${train_dir}/best_model.pt" \
        --split-manifest "${SHARED_MANIFEST}" \
        --metadata-csv "${METADATA_CSV}" \
        --json-root "${JSON_ROOT}" \
        --output-dir "${test_dir}"; then
        echo "  ERROR: test evaluation failed for ${run_name}"
        return 1
    fi

    log_run "${run_name}" "${train_dir}" "${test_dir}"
}

# ════════════════════════════════════════════════════════════════════
# EXPERIMENT GRID
# ════════════════════════════════════════════════════════════════════

# ── 1. Ridge baseline (fast, no GPU needed) ──────────────────────────────────
echo "=== RIDGE BASELINE EXPERIMENTS ==="

run_experiment "ridge_alpha_grid_default" \
    --model-backend ridge_window \
    --ridge-alpha-grid "0.01,0.1,1,3,10,30,100,300,1000" \
    --ridge-use-hgbr-blend \
    --ridge-include-test-id

run_experiment "ridge_no_hgbr" \
    --model-backend ridge_window \
    --ridge-alpha-grid "0.01,0.1,1,3,10,30,100,300,1000" \
    --ridge-no-hgbr-blend \
    --ridge-include-test-id

run_experiment "ridge_no_test_id" \
    --model-backend ridge_window \
    --ridge-alpha-grid "0.01,0.1,1,3,10,30,100,300,1000" \
    --ridge-use-hgbr-blend \
    --ridge-no-test-id

# ── 2. OpenTSLM: Gemma-3-270M (frozen backbone, encoder+head only) ──────────
echo "=== GEMMA-3-270M EXPERIMENTS ==="
GEMMA="${PRETRAINED}/gemma-3-270m-it"

run_experiment "opentslm_gemma270m_lr2e4" \
    --model-backend opentslm \
    --llm-id "${GEMMA}" \
    --lr-encoder 2e-4 --lr-projector 1e-4 --lr-regression-head 1e-4 \
    --epochs 60 --patience 15 --batch-size 8 \
    --encoder-patch-size 50

run_experiment "opentslm_gemma270m_lr1e3" \
    --model-backend opentslm \
    --llm-id "${GEMMA}" \
    --lr-encoder 1e-3 --lr-projector 5e-4 --lr-regression-head 5e-4 \
    --epochs 60 --patience 15 --batch-size 8 \
    --encoder-patch-size 50

run_experiment "opentslm_gemma270m_patch100" \
    --model-backend opentslm \
    --llm-id "${GEMMA}" \
    --lr-encoder 2e-4 --lr-projector 1e-4 --lr-regression-head 1e-4 \
    --epochs 60 --patience 15 --batch-size 8 \
    --encoder-patch-size 100

# ── 3. OpenTSLM: Qwen2.5-0.5B (frozen backbone) ─────────────────────────────
echo "=== QWEN2.5-0.5B EXPERIMENTS ==="
QWEN05="${PRETRAINED}/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit"

run_experiment "opentslm_qwen05b_lr2e4" \
    --model-backend opentslm \
    --llm-id "${QWEN05}" \
    --lr-encoder 2e-4 --lr-projector 1e-4 --lr-regression-head 1e-4 \
    --epochs 60 --patience 15 --batch-size 8 \
    --encoder-patch-size 50

run_experiment "opentslm_qwen05b_lr1e3" \
    --model-backend opentslm \
    --llm-id "${QWEN05}" \
    --lr-encoder 1e-3 --lr-projector 5e-4 --lr-regression-head 5e-4 \
    --epochs 60 --patience 15 --batch-size 8 \
    --encoder-patch-size 50

# ── 4. OpenTSLM: Llama-3.2-1B (frozen backbone) ──────────────────────────────
echo "=== LLAMA-3.2-1B EXPERIMENTS ==="
LLAMA1B="${PRETRAINED}/Llama-3.2-1B"

run_experiment "opentslm_llama1b_lr2e4" \
    --model-backend opentslm \
    --llm-id "${LLAMA1B}" \
    --lr-encoder 2e-4 --lr-projector 1e-4 --lr-regression-head 1e-4 \
    --epochs 60 --patience 15 --batch-size 4 \
    --encoder-patch-size 50

run_experiment "opentslm_llama1b_lr1e3" \
    --model-backend opentslm \
    --llm-id "${LLAMA1B}" \
    --lr-encoder 1e-3 --lr-projector 5e-4 --lr-regression-head 5e-4 \
    --epochs 60 --patience 15 --batch-size 4 \
    --encoder-patch-size 50

# ════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════"
echo " ALL EXPERIMENTS COMPLETE"
echo " Log file: ${LOG_FILE}"
echo "════════════════════════════════════════════════════"

python3 - "${LOG_FILE}" <<'PYEOF'
import json, sys
log_file = sys.argv[1]
entries = [json.loads(l) for l in open(log_file)]
print(f"{'Run':<45} {'val_r2':>8} {'test_r2':>8} {'test_pearson':>13}")
print('-' * 80)
for e in entries:
    vr2 = e.get('val_file_r2')
    tr2 = e.get('test_file_r2')
    tp  = e.get('test_file_pearson_r')
    vr2s = f"{vr2:.4f}" if isinstance(vr2, float) else str(vr2)
    tr2s = f"{tr2:.4f}" if isinstance(tr2, float) else str(tr2)
    tps  = f"{tp:.4f}"  if isinstance(tp,  float) else str(tp)
    print(f"{e['run_name']:<45} {vr2s:>8} {tr2s:>8} {tps:>13}")
PYEOF
