"""Utilities for bootstrap sampling and metric computation."""
import pandas as pd
import numpy as np
import logging
import os

import config
from utils.loader import load_rpw_labels, load_rpw_ecg_map, find_records
from utils.metrics import auc, f1score


def verify_test_data(csv_path):
    """
    Verify the integrity of test data against directories and labels.

    Args:
        csv_path (str): Path to the CSV file containing test participants.
    """
    df = pd.read_csv(csv_path)
    rpw_labels = load_rpw_labels()
    rpw_ecg_map = load_rpw_ecg_map()
    rpw_test_dir = f"{config.directory}/data/test/rpw/"
    ecg_test_dir = f"{config.directory}/data/test/ecg/"
    rpw_files = find_records(rpw_test_dir)
    ecg_files = find_records(ecg_test_dir)
    expected_len = len(ecg_files)
    if len(df) != expected_len:
        logging.warning("Expected %d test samples, found %d", expected_len, len(df))
    errors = 0
    for index, row in df.iterrows():
        p_id = str(row['participant_id'])
        y_csv = int(row['y'])
        if p_id not in rpw_labels:
            print(f"Error: ID {p_id} not found in RPW-Labels.")
            errors += 1
            continue
        label_val = rpw_labels[p_id]
        y_orig = int(label_val[1] if isinstance(label_val, (list, np.ndarray)) and len(label_val) > 1 else label_val)
        if y_csv != y_orig:
            print(f"Error: Label-Mismatch for {p_id}. CSV: {y_csv}, Original: {y_orig}")
            errors += 1
        if p_id not in rpw_ecg_map:
            print(f"Error: No RPW-ECG Match found for {p_id}.")
            errors += 1
            continue
        ecg_filename = rpw_ecg_map[p_id].split(".")[0]
        rpw_found = any(f.startswith(p_id) for f in rpw_files)
        ecg_found = any(f.startswith(ecg_filename) for f in ecg_files)
        if not rpw_found:
            print(f"Error: No RPW files for {p_id} in Test Directory.")
            errors += 1
        if not ecg_found:
            print(f"Error: No ECG files for {ecg_filename} in Test Directory.")
            errors += 1
    if errors != 0:
        print(f"\nVerification finished with {errors} Errors. Please check Logs.")


_THRESHOLDS = np.arange(100) / 100  # identical to utils.metrics.roc


def _weighted_metrics(y, p, w):
    """
    AUC, sensitivity, specificity and F1 (at config.classification_threshold) with integer sample weights.

    Equivalent to utils.metrics.roc/auc/confusionmatrix on a data set in which every row is duplicated w times
    (same ROC definition: 100 thresholds, prediction > threshold, TPR/FPR sorted, trapezoidal AUC).
    """
    pos = p[None, :] > _THRESHOLDS[:, None]
    wy, wn = w * y, w * (1 - y)
    tp, fp = (pos * wy).sum(1), (pos * wn).sum(1)
    P, N = wy.sum(), wn.sum()
    tpr = np.sort(np.r_[0.0, tp / P, 1.0])
    fpr = np.sort(np.r_[0.0, fp / N, 1.0])
    area = auc({'TPR': tpr, 'FPR': fpr})
    i = int(round(config.classification_threshold * 100))  # index of the classification threshold
    tp5, fp5 = tp[i], fp[i]
    fn5, tn5 = P - tp5, N - fp5
    sens = tp5 / P if P > 0 else 0
    spec = tn5 / N if N > 0 else 0
    f1 = f1score(tp5, fp5, fn5)
    return area, sens, spec, f1


def _load_clusters(ids):
    """Map examination IDs (SessionID) to athletes (PatID) via labels.csv; fallback: one cluster per row."""
    try:
        labels = pd.read_csv(f"{config.directory}/data/labels.csv")
        mapping = dict(zip(labels["SessionID"].astype(str), labels["PatID"].astype(str)))
        clusters = ids.astype(str).map(mapping)
        if clusters.isna().any():
            logging.warning("%d examinations without PatID - treated as separate clusters", clusters.isna().sum())
            clusters = clusters.fillna(ids.astype(str))
        return clusters
    except FileNotFoundError:
        logging.warning("labels.csv not found - bootstrap resamples examinations, not athletes")
        return ids.astype(str)


def run_bootstrap(csv_path, n_boot):
    """
    Run a paired, athlete-level (cluster) bootstrap for all models in the predictions CSV.

    In every iteration, athletes (PatID) are drawn with replacement; all examinations of a drawn athlete enter
    the resample together. All models and all AUC differences are computed on the SAME resample, i.e. the
    difference is recorded per bootstrap sample and its percentile interval is reported (not derived from
    the individual model CIs). Rows with a missing prediction for a model are excluded for that model;
    differences use examinations with predictions from both models.

    Outputs next to csv_path: bootstrap.csv (summary) and bootstrap_samples.csv (all iterations).

    Args:
        csv_path (str): Path to the CSV file containing the predictions (column participant_id = SessionID).
        n_boot (int): The number of bootstrap iterations to perform.
    """
    df = pd.read_csv(csv_path)
    clusters = _load_clusters(df['participant_id'])
    unique_clusters = clusters.unique()
    cluster_index = clusters.map({c: i for i, c in enumerate(unique_clusters)}).values
    y = df['y'].values.astype(float)

    model_cols = {'rpw': 'p_rpw', 'ecg': 'p_ecg', 'comb': 'p_comb', 'clin': 'p_clin', 'clin_comb': 'p_clin_comb'}
    model_cols = {m: c for m, c in model_cols.items() if c in df.columns and df[c].notna().any()}
    diffs = {'DELTA_AUC': ('comb', 'ecg'),                 # Combined - ECG
             'DELTA_AUC_CLIN': ('clin_comb', 'clin'),      # Clinical+Combined - Clinical
             'DELTA_AUC_COMB_VS_CLIN': ('comb', 'clin'),   # Combined - Clinical
             'DELTA_AUC_ECG_VS_CLIN': ('ecg', 'clin'),     # ECG - Clinical
             'DELTA_AUC_CLINCOMB_VS_COMB': ('clin_comb', 'comb'),  # Clinical+Combined - Combined
             'DELTA_AUC_COMB_VS_RPW': ('comb', 'rpw'),     # Combined - RPW
             'DELTA_AUC_ECG_VS_RPW': ('ecg', 'rpw')}       # ECG - RPW
    diffs = {k: v for k, v in diffs.items() if v[0] in model_cols and v[1] in model_cols}

    preds = {m: df[c].values.astype(float) for m, c in model_cols.items()}
    avail = {m: ~np.isnan(v) for m, v in preds.items()}

    def metrics(model, w, mask=None):
        m = avail[model] if mask is None else mask
        return _weighted_metrics(y[m], preds[model][m], w[m])

    def diff(a, b, w):
        mask = avail[a] & avail[b]
        return metrics(a, w, mask)[0] - metrics(b, w, mask)[0]

    ones = np.ones(len(df))
    orig = {m: metrics(m, ones) for m in model_cols}
    orig_diff = {k: diff(a, b, ones) for k, (a, b) in diffs.items()}

    samples = []
    n_discarded = 0
    for it in range(n_boot):
        drawn = np.random.choice(len(unique_clusters), size=len(unique_clusters), replace=True)
        w = np.bincount(drawn, minlength=len(unique_clusters))[cluster_index].astype(float)
        if (w * y).sum() == 0 or (w * (1 - y)).sum() == 0:
            n_discarded += 1  # resample contains only one class -> AUC undefined
            continue
        row = {'iteration': it}
        for m in model_cols:
            a, se, sp, f = metrics(m, w)
            row.update({f'{m}_auc': a, f'{m}_sens': se, f'{m}_spec': sp, f'{m}_f1': f})
        for k, (a, b) in diffs.items():
            row[k] = diff(a, b, w)
        samples.append(row)
    samples = pd.DataFrame(samples)

    summary_data = []
    print(f"\nBootstrap: {n_boot} iterations, resampling {len(unique_clusters)} athletes "
          f"({len(df)} examinations); {n_discarded} single-class resamples discarded")
    for model in model_cols:
        n = int(avail[model].sum())
        print(f"\n--- {model.upper()} Results (n={n}) ---")
        for m_idx, metric in enumerate(['auc', 'sens', 'spec', 'f1']):
            ci_low, ci_high = np.percentile(samples[f'{model}_{metric}'], [2.5, 97.5])
            print(f"{metric.upper()}: {orig[model][m_idx]:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}])")
            summary_data.append({'Model': model.upper(), 'Metric': metric.upper(), 'Original': orig[model][m_idx],
                                 'CI_2.5': ci_low, 'CI_97.5': ci_high, 'n': n})
    for k, (a, b) in diffs.items():
        diff_low, diff_high = np.percentile(samples[k], [2.5, 97.5])
        p_le0 = float((samples[k] <= 0).mean())
        print(f"\n--- Difference ({a.upper()} AUC - {b.upper()} AUC) ---")
        print(f"Delta AUC: {orig_diff[k]:.4f} (95% CI: [{diff_low:.4f}, {diff_high:.4f}]), "
              f"share of resamples with Delta <= 0: {p_le0:.3f}")
        summary_data.append({'Model': 'DIFFERENCE', 'Metric': k, 'Original': orig_diff[k],
                             'CI_2.5': diff_low, 'CI_97.5': diff_high, 'P_delta_le_0': p_le0})

    out_dir = os.path.dirname(csv_path)
    pd.DataFrame(summary_data).to_csv(os.path.join(out_dir, 'bootstrap.csv'), index=False)
    samples.to_csv(os.path.join(out_dir, 'bootstrap_samples.csv'), index=False)
    print(f"Bootstrap Result saved to {os.path.join(out_dir, 'bootstrap.csv')} (+ bootstrap_samples.csv)")
    write_delta_auc_table(out_dir)


# Rows of the AUC-difference table (bootstrap.csv metric, label); rows missing in bootstrap.csv are skipped
DELTA_TABLE_ROWS = [
    ('DELTA_AUC', 'Combined $-$ ECG'),
    ('DELTA_AUC_COMB_VS_RPW', 'Combined $-$ RPW'),
    ('DELTA_AUC_ECG_VS_RPW', 'ECG $-$ RPW'),
    ('DELTA_AUC_ECG_VS_CLIN', 'ECG $-$ Clinical'),
    ('DELTA_AUC_COMB_VS_CLIN', 'Combined $-$ Clinical'),
    ('DELTA_AUC_CLINCOMB_VS_COMB', 'Clinical + Combined $-$ Combined'),
]


def write_delta_auc_table(results_dir):
    """
    Write the paired AUC differences (from bootstrap.csv) as LaTeX tabular: delta_auc_table.tex.
    The whole table is coloured if config.highlight_revision is True.

    Args:
        results_dir (str): Folder containing bootstrap.csv.
    """
    color = config.highlight_color if config.highlight_revision else None
    boot = pd.read_csv(os.path.join(results_dir, 'bootstrap.csv'))
    diff = boot[boot['Model'] == 'DIFFERENCE'].set_index('Metric')
    c = (lambda x: "\\textcolor{" + color + "}{" + x + "}") if color else (lambda x: x)
    fmt = lambda v: f"{v:.2f}".replace("-", "$-$")
    header = c("Comparison") + " & " + c("$\\Delta$AUC") + " & " + c("95\\% CI") + " \\\\"
    lines = ["\\begin{tabular}{lcc}", "\\toprule", header, "\\midrule"]
    for metric, label in DELTA_TABLE_ROWS:
        if metric not in diff.index:
            continue
        r = diff.loc[metric]
        ci = f"{fmt(r['CI_2.5'])} to {fmt(r['CI_97.5'])}"
        lines.append(c(label) + " & " + c(fmt(r['Original'])) + " & " + c(ci) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out = os.path.join(results_dir, 'delta_auc_table.tex')
    with open(out, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Delta AUC table saved to {out}")
