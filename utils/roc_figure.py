"""Test-set ROC figure of the fold ensembles (roc_test.pdf / .png).

Only the hold-out test ROCs of the fold ensembles are shown; per-fold curves are shown in roc_plots.pdf
(utils.evaluation.generate_results), whose layout, fonts and axes this figure follows. Colours avoid
red-green contrasts, and every model additionally has its own line style, so identity never relies on
colour alone. Panel b with the clinical models is added if clinical predictions are present.

Usage (no TensorFlow needed):  python -m utils.roc_figure [results_dir]   (default: config.get_results_dir())
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.metrics import roc, auc

# model key -> (column in test_predictions.csv, bootstrap model name, label, colour, line style, width)
# Colours follow Figure 1 (no green: red-green safe); every model keeps its own line style.
# Layout, fonts and axes match the fold-level supplementary figure (utils.evaluation.generate_results).
MODELS = {
    "rpw": ("p_rpw", "RPW", "RPW", "#4F7CB5", (0, (5, 2)), 1.5),
    "ecg": ("p_ecg", "ECG", "ECG", "#C07A5B", (0, (6, 1.5, 1, 1.5)), 1.5),
    "comb": ("p_comb", "COMB", "Combined", "#000000", "-", 2.2),
    "clin": ("p_clin", "CLIN", "Clinical", "#9BA3AE", (0, (6, 1.5, 1, 1.5)), 1.5),
    "clin_comb": ("p_clin_comb", "CLIN_COMB", "Clinical + Combined", "#4D4D4D", (0, (3, 1.5)), 1.5),
}
CHANCE_COLOR = "#BDBDBD"
CHANCE_STYLE = (0, (1, 1.5))
FONTSIZE = 15  # same as roc_plots.pdf


def _roc_points(df, col):
    sub = df[df[col].notna()]
    p = sub[col].values.astype(float)
    y = sub["y"].values.astype(float)
    preds = np.stack([1 - p, p], axis=1)
    labels = np.stack([1 - y, y], axis=1)
    r = roc(y_true=labels, y_pred=preds, y_record_ids=sub["participant_id"].astype(str).values, c=1,
            calculation_method="subsample_level")
    fprs, tprs = [0.0] + list(r["FPR"]) + [1.0], [0.0] + list(r["TPR"]) + [1.0]
    fprs, tprs = zip(*sorted(zip(fprs, tprs)))
    return np.array(fprs), np.array(tprs), auc(r)


def _label(name, key, df, boot):
    col, boot_name = MODELS[key][0], MODELS[key][1]
    row = boot[(boot["Model"] == boot_name) & (boot["Metric"] == "AUC")] if boot is not None else None
    if row is not None and len(row):
        r = row.iloc[0]
        return f"{name}  {r['Original']:.2f} ({r['CI_2.5']:.2f}–{r['CI_97.5']:.2f})"
    return f"{name}  {_roc_points(df, col)[2]:.2f}"


def _panel(ax, df, boot, keys, title):
    for key in keys:
        col, _, name, colour, ls, lw = MODELS[key]
        if col not in df.columns or df[col].notna().sum() == 0:
            continue
        fpr, tpr, _ = _roc_points(df, col)
        ax.plot(fpr, tpr, color=colour, linestyle=ls, linewidth=lw, alpha=1,
                label=_label(name, key, df, boot), zorder=4 if key == "comb" else 3, clip_on=False)
    ax.plot([0, 1], [0, 1], label="Chance", alpha=1, linewidth=1, linestyle=CHANCE_STYLE, color=CHANCE_COLOR,
            zorder=1)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_aspect("equal")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("1 − specificity", fontsize=FONTSIZE)
    ax.set_ylabel("Sensitivity", fontsize=FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    if title:
        ax.set_title(title, loc="left", fontsize=16, fontweight="bold")
    ax.label_outer()


def plot_test_rocs(results_dir, include_clinical=None):
    """
    Plot the test-set ROC curves of the fold ensembles (roc_test.pdf / .png in results_dir).

    Args:
        results_dir (str): Folder containing test_predictions.csv (and bootstrap.csv for the CIs).
        include_clinical (bool, optional): Add a second panel with the clinical models.
            Default: True if clinical predictions are present.
    """
    df = pd.read_csv(f"{results_dir}/test_predictions.csv")
    boot_path = f"{results_dir}/bootstrap.csv"
    boot = pd.read_csv(boot_path) if os.path.exists(boot_path) else None
    has_clin = "p_clin" in df.columns and df["p_clin"].notna().any()
    if include_clinical is None:
        include_clinical = has_clin
    include_clinical = include_clinical and has_clin

    # Panel size (~5 inch) and font sizes as in roc_plots.pdf
    if include_clinical:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0.18)
        _panel(axes[0], df, boot, ["rpw", "ecg", "comb"], "a) Resting biosignals")
        _panel(axes[1], df, boot, ["comb", "clin", "clin_comb"], "b) Clinical reference")
    else:
        fig, ax = plt.subplots(figsize=(5.6, 5.6))
        axes = [ax]
        _panel(ax, df, boot, ["rpw", "ecg", "comb"], None)

    # One legend to the right of the panels, as in roc_plots.pdf (Combined is shared by both panels)
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels and l != "Chance":
                handles.append(h)
                labels.append(l)
    h_chance = [h for h, l in zip(*axes[0].get_legend_handles_labels()) if l == "Chance"]
    handles += h_chance
    labels += ["Chance"] * len(h_chance)
    leg = fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.04, 0.5),
                     bbox_transform=axes[-1].transAxes, fontsize=FONTSIZE,
                     title="Test AUC (95% CI)", title_fontsize=FONTSIZE, framealpha=1, facecolor="white",
                     handlelength=3)
    leg._legend_box.align = "left"
    for ext in ["pdf", "png"]:
        fig.savefig(f"{results_dir}/roc_test.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Test ROC figure saved to {results_dir}/roc_test.pdf")


if __name__ == "__main__":
    import config
    plot_test_rocs(sys.argv[1] if len(sys.argv) > 1 else config.get_results_dir())
