"""Utilities for bootstrap sampling and metric computation."""
import pandas as pd
import numpy as np
import logging
import os
import glob

import config
from utils.loader import load_rpw_labels, load_rpw_ecg_map, find_records
from utils.metrics import auc, roc, confusionmatrix, f1score


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


def compute_metrics_for_bootstrap(df, pred_col):
    """
    Compute evaluation metrics for bootstrap samples.

    Args:
        df (pd.DataFrame): Dataframe containing the samples and predictions.
        pred_col (str): The column name containing the prediction probabilities.

    Returns:
        tuple: A tuple of (AUC, sensitivity, specificity, F1-score).
    """
    preds = np.stack([1 - df[pred_col].values, df[pred_col].values], axis=1)
    labels = np.stack([1 - df['y'].values, df['y'].values], axis=1)
    files = df['file_id'].values
    roc_curve = roc(y_true=labels, y_pred=preds, y_record_ids=files, c=1, calculation_method="subsample_level")
    area_under_curve = auc(roc_curve)
    tp, fp, tn, fn, _ = confusionmatrix(labels, preds, files, threshold=0.5, c=1, calculation_method="subsample_level")
    f1 = f1score(tp, fp, fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return area_under_curve, sensitivity, specificity, f1


def run_bootstrap(csv_path, n_boot):
    """
    Run bootstrap evaluation for RPW, ECG, and Combined models.

    Args:
        csv_path (str): Path to the CSV file containing the predictions.
        n_boot (int): The number of bootstrap iterations to perform.
    """
    df = pd.read_csv(csv_path)
    unique_participants = df['participant_id'].unique()
    results = {'rpw': {'auc': [], 'sens': [], 'spec': [], 'f1': []}, 'ecg': {'auc': [], 'sens': [], 'spec': [], 'f1': []}, 'comb': {'auc': [], 'sens': [], 'spec': [], 'f1': []}, 'diff': []}
    df['file_id'] = df['participant_id'].astype(str)
    orig_r = compute_metrics_for_bootstrap(df, 'p_rpw')
    orig_e = compute_metrics_for_bootstrap(df, 'p_ecg')
    orig_c = compute_metrics_for_bootstrap(df, 'p_comb')
    for i in range(n_boot):
        boot_parts = np.random.choice(unique_participants, size=len(unique_participants), replace=True)
        boot_frames = []
        for idx, p_id in enumerate(boot_parts):
            sub_df = df[df['participant_id'] == p_id].copy()
            sub_df['file_id'] = sub_df['participant_id'].astype(str) + f"_b{idx}"
            boot_frames.append(sub_df)
        boot_df = pd.concat(boot_frames, ignore_index=True)
        a_r, se_r, sp_r, f_r = compute_metrics_for_bootstrap(boot_df, 'p_rpw')
        a_e, se_e, sp_e, f_e = compute_metrics_for_bootstrap(boot_df, 'p_ecg')
        a_c, se_c, sp_c, f_c = compute_metrics_for_bootstrap(boot_df, 'p_comb')
        results['rpw']['auc'].append(a_r)
        results['rpw']['sens'].append(se_r)
        results['rpw']['spec'].append(sp_r)
        results['rpw']['f1'].append(f_r)
        results['ecg']['auc'].append(a_e)
        results['ecg']['sens'].append(se_e)
        results['ecg']['spec'].append(sp_e)
        results['ecg']['f1'].append(f_e)
        results['comb']['auc'].append(a_c)
        results['comb']['sens'].append(se_c)
        results['comb']['spec'].append(sp_c)
        results['comb']['f1'].append(f_c)
        results['diff'].append(a_c - a_e)
    summary_data = []
    for model in ['rpw', 'ecg', 'comb']:
        orig_vals = orig_r if model == 'rpw' else (orig_e if model == 'ecg' else orig_c)
        metrics = ['auc', 'sens', 'spec', 'f1']
        print(f"\n--- {model.upper()} Results ---")
        for m_idx, metric in enumerate(metrics):
            ci_low, ci_high = np.percentile(results[model][metric], [2.5, 97.5])
            print(f"{metric.upper()}: {orig_vals[m_idx]:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}])")
            summary_data.append(
                {'Model': model.upper(), 'Metric': metric.upper(), 'Original': orig_vals[m_idx], 'CI_2.5': ci_low,
                 'CI_97.5': ci_high})
    diff_low, diff_high = np.percentile(results['diff'], [2.5, 97.5])
    orig_diff = orig_c[0] - orig_e[0]
    print(f"\n--- Difference (Combined AUC - ECG AUC) ---")
    print(f"Delta AUC: {orig_diff:.4f} (95% CI: [{diff_low:.4f}, {diff_high:.4f}])")
    summary_data.append(
        {'Model': 'DIFFERENCE', 'Metric': 'DELTA_AUC', 'Original': orig_diff, 'CI_2.5': diff_low, 'CI_97.5': diff_high})
    out_dir = os.path.dirname(csv_path)
    out_file = os.path.join(out_dir, 'bootstrap.csv')
    pd.DataFrame(summary_data).to_csv(out_file, index=False)
    print(f"Bootstrap Result saved to {out_file}")
