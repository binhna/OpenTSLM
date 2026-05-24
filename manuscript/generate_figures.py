#!/usr/bin/env python3
"""Generate all manuscript figures as PDFs in manuscript/figures/."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import pearsonr

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PRED_DIR = Path("/tmp")

DEVICE_NAMES = {1: "Cup (AIM-C)", 2: "Spoon (AIM-S)", 3: "Pendant (AIM-P)"}
DEVICE_SHORT = {1: "Cup", 2: "Spoon", 3: "Pendant"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "ridge": "#2166ac",
    "gemma": "#d73027",
    "qwen": "#f46d43",
    "moment": "#1a9641",
}

MODEL_LABELS = {
    "ridge_alpha_grid_default": "Ridge+HGBR",
    "opentslm_gemma270m_lr1e3": "Gemma-270M",
    "opentslm_qwen05b_lr2e4": "Qwen-0.5B",
    "moment_lr2e4": "MOMENT (lr=2e-4)",
    "moment_lr1e3": "MOMENT (lr=1e-3)",
}

MODEL_COLORS = {
    "ridge_alpha_grid_default": COLORS["ridge"],
    "opentslm_gemma270m_lr1e3": COLORS["gemma"],
    "opentslm_qwen05b_lr2e4": COLORS["qwen"],
    "moment_lr2e4": COLORS["moment"],
    "moment_lr1e3": "#4daf4a",
}


def load_preds(run_name):
    path = PRED_DIR / f"{run_name}_preds.jsonl"
    if not path.exists():
        return None
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def metrics(preds, targets):
    preds, targets = np.array(preds, float), np.array(targets, float)
    r2 = 1 - np.sum((targets - preds)**2) / np.sum((targets - np.mean(targets))**2)
    mae = np.mean(np.abs(preds - targets))
    pr, _ = pearsonr(targets, preds)
    return r2, mae, pr


# ── Figure 1: Scatter plots for the four key models (2×2) ────────────────────
def fig_scatter():
    runs = [
        "ridge_alpha_grid_default",
        "opentslm_gemma270m_lr1e3",
        "opentslm_qwen05b_lr2e4",
        "moment_lr2e4",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    lim = (5, 92)

    for ax, run in zip(axes.flat, runs):
        rows = load_preds(run)
        if rows is None:
            ax.set_visible(False)
            continue
        y_true = np.array([r["target"] for r in rows])
        y_pred = np.array([r["prediction"] for r in rows])
        test_ids = np.array([r["test_id"] for r in rows])
        r2, mae, pr = metrics(y_pred, y_true)

        device_colors = {1: "#2166ac", 2: "#f46d43", 3: "#1a9641"}
        for tid, dc in device_colors.items():
            mask = test_ids == tid
            ax.scatter(y_true[mask], y_pred[mask], c=dc, s=18, alpha=0.7,
                       label=DEVICE_SHORT[tid], linewidths=0)

        lims = np.array(lim)
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("True mFARS")
        ax.set_ylabel("Predicted mFARS")
        ax.set_title(MODEL_LABELS[run])
        ax.text(0.04, 0.95, f"$R^2$={r2:.2f}  $r$={pr:.2f}\nMAE={mae:.1f}",
                transform=ax.transAxes, va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    handles = [mpatches.Patch(color="#2166ac", label="Cup"),
               mpatches.Patch(color="#f46d43", label="Spoon"),
               mpatches.Patch(color="#1a9641", label="Pendant")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    out = FIGURES_DIR / "fig1_scatter.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 2: Bar chart — overall test R² and Pearson r by model ─────────────
def fig_model_comparison():
    results_path = Path(__file__).parent.parent / "results" / "experiments_log.jsonl"
    entries = {}
    for l in results_path.read_text().splitlines():
        if l.strip():
            e = json.loads(l)
            entries[e["run_name"]] = e  # keep last

    select = [
        ("ridge_alpha_grid_default", "Ridge+HGBR", COLORS["ridge"]),
        ("opentslm_gemma270m_lr2e4", "Gemma-270M\n(lr=2e-4)", COLORS["gemma"]),
        ("opentslm_gemma270m_lr1e3", "Gemma-270M\n(lr=1e-3)", "#e85d4a"),
        ("opentslm_qwen05b_lr2e4", "Qwen-0.5B\n(lr=2e-4)", COLORS["qwen"]),
        ("opentslm_llama1b_lr2e4", "Llama-1B\n(lr=2e-4)", "#984ea3"),
        ("moment_lr2e4", "MOMENT\n(lr=2e-4)", COLORS["moment"]),
        ("moment_lr1e3", "MOMENT\n(lr=1e-3)", "#4daf4a"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    x = np.arange(len(select))
    w = 0.65

    val_r2 = [entries.get(r, {}).get("val_file_r2", 0) for r, _, _ in select]
    test_r2 = [entries.get(r, {}).get("test_file_r2", 0) for r, _, _ in select]
    pearson = [entries.get(r, {}).get("test_file_pearson_r", 0) for r, _, _ in select]
    labels = [lbl for _, lbl, _ in select]
    colors = [c for _, _, c in select]

    bars1 = ax1.bar(x - w/4, val_r2, w/2, label="Val $R^2$", color=colors, alpha=0.5, edgecolor="white")
    bars2 = ax1.bar(x + w/4, test_r2, w/2, label="Test $R^2$", color=colors, alpha=0.95, edgecolor="white")
    ax1.axhline(0, color="black", lw=0.6)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=7.5)
    ax1.set_ylabel("File-level $R^2$")
    ax1.set_title("Val vs. Test $R^2$")
    ax1.legend(fontsize=7.5)
    ax1.set_ylim(-0.15, 0.85)

    bars3 = ax2.bar(x, pearson, w, color=colors, alpha=0.9, edgecolor="white")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7.5)
    ax2.set_ylabel("Pearson $r$ (test)")
    ax2.set_title("Test Pearson Correlation")
    ax2.set_ylim(-0.2, 1.0)
    ax2.axhline(0, color="black", lw=0.6)

    fig.tight_layout()
    out = FIGURES_DIR / "fig2_model_comparison.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 3: Device-stratified heatmap ──────────────────────────────────────
def fig_device_heatmap():
    ds_path = Path(__file__).parent.parent / "results" / "device_stratified_results.json"
    data = json.loads(ds_path.read_text())

    select_runs = [
        ("ridge_alpha_grid_default", "Ridge+HGBR"),
        ("opentslm_gemma270m_lr1e3", "Gemma-270M"),
        ("opentslm_qwen05b_lr2e4", "Qwen-0.5B"),
        ("opentslm_llama1b_lr2e4", "Llama-1B"),
        ("moment_lr2e4", "MOMENT"),
    ]
    devices = ["cup", "spoon", "pendant", "all"]
    device_labels = ["Cup (AIM-C)", "Spoon (AIM-S)", "Pendant (AIM-P)", "All devices"]

    r2_matrix = np.zeros((len(select_runs), len(devices)))
    for i, (run, _) in enumerate(select_runs):
        for j, dev in enumerate(devices):
            v = data.get(run, {}).get(dev, {}).get("r2")
            r2_matrix[i, j] = v if v is not None else np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.0), gridspec_kw={"width_ratios": [3, 1]})

    # Clamp for display
    display = np.clip(r2_matrix, -0.5, 0.8)
    im = ax1.imshow(display, cmap="RdYlGn", vmin=-0.5, vmax=0.8, aspect="auto")
    plt.colorbar(im, ax=ax1, label="$R^2$", shrink=0.85)

    ax1.set_xticks(range(len(devices)))
    ax1.set_xticklabels(device_labels, rotation=20, ha="right")
    ax1.set_yticks(range(len(select_runs)))
    ax1.set_yticklabels([lbl for _, lbl in select_runs])
    ax1.set_title("File-level $R^2$ by device")

    for i in range(len(select_runs)):
        for j in range(len(devices)):
            v = r2_matrix[i, j]
            txt = f"{v:.2f}" if not np.isnan(v) else "—"
            ax1.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                     color="black" if abs(v) < 0.5 else "white")

    # Pearson r panel
    pr_matrix = np.zeros((len(select_runs), len(devices)))
    for i, (run, _) in enumerate(select_runs):
        for j, dev in enumerate(devices):
            v = data.get(run, {}).get(dev, {}).get("pearson_r")
            pr_matrix[i, j] = v if v is not None else np.nan

    im2 = ax2.imshow(np.clip(pr_matrix, -0.5, 1.0), cmap="RdYlGn", vmin=-0.5, vmax=1.0, aspect="auto")
    plt.colorbar(im2, ax=ax2, label="Pearson $r$", shrink=0.85)
    ax2.set_xticks([0]); ax2.set_xticklabels(["All"], rotation=20, ha="right")
    ax2.set_yticks(range(len(select_runs)))
    ax2.set_yticklabels([])
    ax2.set_title("Pearson $r$ (all)")
    for i in range(len(select_runs)):
        v = pr_matrix[i, 3]
        ax2.text(0, i, f"{v:.2f}" if not np.isnan(v) else "—",
                 ha="center", va="center", fontsize=7.5)

    fig.tight_layout()
    out = FIGURES_DIR / "fig3_device_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 4: CV fold variability + val-test gap ──────────────────────────────
def fig_cv_and_gap():
    cv_path = Path(__file__).parent.parent / "results" / "cv_results.json"
    cv = json.loads(cv_path.read_text())

    log_path = Path(__file__).parent.parent / "results" / "experiments_log.jsonl"
    entries = {}
    for l in log_path.read_text().splitlines():
        if l.strip():
            e = json.loads(l)
            entries[e["run_name"]] = e

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # Left: CV fold R² values
    fold_r2 = [f["r2"] for f in cv["fold_results"]]
    folds = [f["fold"] for f in cv["fold_results"]]
    ax1.bar(folds, fold_r2, color=COLORS["ridge"], alpha=0.8, edgecolor="white")
    ax1.axhline(cv["cv_r2_mean"], color="black", lw=1.2, ls="--", label=f"Mean={cv['cv_r2_mean']:.2f}")
    ax1.axhline(entries.get("ridge_alpha_grid_default", {}).get("test_file_r2", 0),
                color="red", lw=1.2, ls=":", label=f"Test R²={entries['ridge_alpha_grid_default']['test_file_r2']:.2f}")
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("File-level $R^2$")
    ax1.set_title("5-Fold CV: Ridge+HGBR")
    ax1.legend(fontsize=7.5)
    ax1.set_ylim(-0.05, 0.55)
    ax1.set_xticks(folds)

    # Right: val-test gap for all models
    select = [
        ("ridge_alpha_grid_default", "Ridge+HGBR", COLORS["ridge"]),
        ("opentslm_gemma270m_lr1e3", "Gemma-270M", COLORS["gemma"]),
        ("opentslm_qwen05b_lr2e4", "Qwen-0.5B", COLORS["qwen"]),
        ("opentslm_llama1b_lr2e4", "Llama-1B", "#984ea3"),
        ("moment_lr2e4", "MOMENT", COLORS["moment"]),
    ]
    labels = [lbl for _, lbl, _ in select]
    val_r2 = [entries.get(r, {}).get("val_file_r2", 0) for r, _, _ in select]
    test_r2 = [entries.get(r, {}).get("test_file_r2", 0) for r, _, _ in select]
    colors = [c for _, _, c in select]

    x = np.arange(len(select))
    w = 0.35
    ax2.bar(x - w/2, val_r2, w, label="Val $R^2$", color=colors, alpha=0.5, edgecolor="white")
    ax2.bar(x + w/2, test_r2, w, label="Test $R^2$", color=colors, alpha=0.95, edgecolor="white")
    for i, (v, t) in enumerate(zip(val_r2, test_r2)):
        ax2.annotate("", xy=(x[i] + w/2, t), xytext=(x[i] - w/2, v),
                     arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))
    ax2.axhline(0, color="black", lw=0.6)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7.5, rotation=15, ha="right")
    ax2.set_ylabel("File-level $R^2$")
    ax2.set_title("Val–Test Gap per Model")
    ax2.legend(fontsize=7.5)
    ax2.set_ylim(-0.1, 0.9)

    fig.tight_layout()
    out = FIGURES_DIR / "fig4_cv_gap.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 5: System diagram (text-based placeholder) ────────────────────────
def fig_system_diagram():
    """Simple pipeline schematic."""
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")

    boxes = [
        (0.3, 1.3, 1.6, 1.4, "AIM Sensor\n(Cup/Spoon/\nPendant)", "#aec7e8"),
        (2.3, 1.3, 1.6, 1.4, "Signal\nPreprocessing\n(filter, norm,\nwindow)", "#ffbb78"),
        (4.3, 1.3, 1.6, 1.4, "Frozen\nEncoder\n(Ridge / LLM /\nMOMENT)", "#98df8a"),
        (6.3, 1.3, 1.6, 1.4, "Regression\nHead\n(MLP)", "#ff9896"),
        (8.3, 1.3, 1.6, 1.4, "mFARS\nPrediction\n(0–93)", "#c5b0d5"),
    ]
    for x, y, w, h, txt, col in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.08", facecolor=col, edgecolor="grey", lw=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center",
                fontsize=7.5, multialignment="center")

    for i in range(len(boxes) - 1):
        x_end = boxes[i][0] + boxes[i][2]
        x_start = boxes[i+1][0]
        y_mid = boxes[i][1] + boxes[i][3]/2
        ax.annotate("", xy=(x_start, y_mid), xytext=(x_end, y_mid),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.0))

    # Labels underneath
    labels = ["Raw IMU\n100 Hz, 7ch", "Median filter\nz-score\n3000-sample\nwindows",
              "Patch tokens\nor features", "Linear(d, 512)\nReLU / Dropout\nLinear(512,1)", "Scalar\noutput"]
    for i, (x, y, w, h, _, _) in enumerate(boxes):
        ax.text(x + w/2, y - 0.25, labels[i], ha="center", va="top",
                fontsize=6.5, color="#444444", multialignment="center")

    ax.set_title("FRDA mFARS Prediction Pipeline", fontsize=10, pad=6)
    fig.tight_layout()
    out = FIGURES_DIR / "fig5_pipeline.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    fig_scatter()
    fig_model_comparison()
    fig_device_heatmap()
    fig_cv_and_gap()
    fig_system_diagram()
    print("\nAll figures saved to", FIGURES_DIR)
