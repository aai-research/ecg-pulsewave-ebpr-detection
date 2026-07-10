"""Evaluate trained RPW and ECG models: compute metrics, generate tables, plots, and bootstrap CIs."""

from utils.gpu import set_visible_gpu

set_visible_gpu(0)

import config
from utils.training import enable_reproducibility

enable_reproducibility(config.seed)

import tensorflow as tf
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker
import csv

from utils.evaluation import generate_results
from utils.metrics import auc, roc, confusionmatrix, f1score
from utils.splitting import rpw_patients_split_cross_validation, ecg_patients_split_cross_validation
from utils.loader import find_records, load_rpw_labels, load_ecg_labels, load_rpw_ecg_map
from utils.test import run_ecg_model, run_rpw_model
from utils.bootstrap import verify_test_data, run_bootstrap


class MetricCollector:
    """A class to collect and manage evaluation metrics across folds."""
    
    def __init__(self, name):
        self.name = name
        
        self.hists = []

        self.result_dicts_sample = []
        self.rocs_sample = []
        self.titles_sample = []
        self.aucs_sample = []
        self.sensitivities_sample = []
        self.specificities_sample = []
        self.f1_scores_sample = []


    def add_fold_metrics(self, hist,
                         result_dict_sample, roc_sample, title_sample, auc_sample,
                         sensitivity_sample, specificity_sample, f1_score_sample):
        """Add metrics from a single fold to the collector."""

        if hist is not None:
            self.hists.append(hist)

        self.result_dicts_sample.append(result_dict_sample)
        self.rocs_sample.append(roc_sample)
        self.titles_sample.append(title_sample)
        self.aucs_sample.append(auc_sample)
        self.sensitivities_sample.append(sensitivity_sample)
        self.specificities_sample.append(specificity_sample)
        self.f1_scores_sample.append(f1_score_sample)


    def get_plotting_data(self):
        """Get collected metrics formatted for plotting."""
        return (self.rocs_sample, self.titles_sample, self.aucs_sample, self.sensitivities_sample,
                self.specificities_sample, self.f1_scores_sample)


def calculate_metrics(preds, labels, files):
    """Calculate evaluation metrics from predictions and labels."""
    roc_curve = roc(y_true=labels, y_pred=preds, y_record_ids=files, c=1, calculation_method="subsample_level")
    area_under_curve = auc(roc_curve)
    tp, fp, tn, fn, _ = confusionmatrix(labels, preds, files, threshold=0.5, c=1, calculation_method="subsample_level")

    f1 = f1score(tp=tp, fp=fp, fn=fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return roc_curve, area_under_curve, sensitivity, specificity, f1


def calculate_metrics_from_dict(result_dict):
    """Calculate evaluation metrics from a dictionary of results."""
    labels = []
    preds = []
    files = []

    # result_dict structure: {record_id: [pred, label]}
    for key, val in result_dict.items():
        files.append(key)
        preds.append(val[0])
        labels.append(val[1])

    return calculate_metrics(preds=preds, labels=labels, files=files)


def aggregate_subsamples(results_dict, key_func):
    """Aggregate subsample results based on a key function."""
    aggregated_dict = {}
    groups = {}
    for file, val in results_dict.items():
        k = key_func(file)
        if k not in groups:
            groups[k] = {'preds': [], 'labels': []}
        groups[k]['preds'].append(val[0])
        groups[k]['labels'].append(val[1])
    for k, data in groups.items():
        for i in range(1, len(data['labels'])):
            if not np.array_equal(data['labels'][i], data['labels'][i - 1]):
                logging.error("Inconsistent labels found while grouping predictions.")
        aggregated_dict[k] = [np.mean(data['preds'], axis=0), np.mean(data['labels'], axis=0)]
    return aggregated_dict


def combine_rpw_ecg_models(rpw_dict, ecg_dict):
    """Combine predictions from RPW and ECG models."""
    rpw_ecg_map = load_rpw_ecg_map()

    result_dict = {}

    for rpw_key in rpw_dict.keys():
        if rpw_key not in rpw_ecg_map:
            logging.error("RPW key %s not found in ECG map.", rpw_key)
            continue

        ecg_filename = rpw_ecg_map[rpw_key].split(".")[0]

        # result_dict structure: {record_id: [pred, label]}
        if ecg_filename in ecg_dict:
            if not np.array_equal(rpw_dict[rpw_key][1], ecg_dict[ecg_filename][1]):
                logging.error("Label mismatch between RPW (%s) and ECG (%s).", rpw_key, ecg_filename)

            result_dict[rpw_key] = [np.mean([rpw_dict[rpw_key][0], ecg_dict[ecg_filename][0]], axis=0),
                                    np.mean([rpw_dict[rpw_key][1], ecg_dict[ecg_filename][1]], axis=0)]

    return result_dict


def merge_fold_result_dicts(fold_result_dicts):
    """Merge result dictionaries across multiple folds."""
    if not fold_result_dicts:
        return None

    # Filter out None entries if any
    valid_dicts = [d for d in fold_result_dicts if d is not None]
    if not valid_dicts:
        logging.error("No valid fold dictionaries found to merge.")
        return None

    # Use keys from the first dictionary
    first_dict = valid_dicts[0]
    keys = first_dict.keys()
    merged = {}

    for k in keys:
        preds = []
        label = None

        # result_dict structure: {record_id: [pred, label]}
        for d in valid_dicts:
            if k in d:
                current_label = d[k][1]
                if label is None:
                    label = current_label
                elif not np.array_equal(label, current_label):
                    logging.warning("Inconsistent label for key '%s' across folds. Using first label.", k)

                preds.append(d[k][0])
            else:
                logging.warning("Key %s missing in one of the fold results.", k)

        if len(preds) == config.num_folds:
            mean_pred = np.mean(preds, axis=0)
            merged[k] = [mean_pred, label]
        else:
            logging.error("Missing predictions in some folds for key: %s", k)

    return merged


def combine_fold_models_to_ensemble(collector):
    """Combine fold models into an ensemble."""
    agg_dict_sam = merge_fold_result_dicts(fold_result_dicts=collector.result_dicts_sample)
    if agg_dict_sam:
        roc_sam, auc_sam, sens_sam, spec_sam, f1_sam = calculate_metrics_from_dict(result_dict=agg_dict_sam)
        title_sam = "Ensemble"
    else:
        roc_sam, auc_sam, sens_sam, spec_sam, f1_sam = None, None, None, None, None
        title_sam = None

    collector.add_fold_metrics(
        hist=None,
        result_dict_sample=agg_dict_sam,
        roc_sample=roc_sam,
        title_sample=title_sam,
        auc_sample=auc_sam,
        sensitivity_sample=sens_sam,
        specificity_sample=spec_sam,
        f1_score_sample=f1_sam
    )


def plot_hist(hists, title, filename):
    """Plot and save a histogram of relevance values."""
    if not hists: return
    h = np.concatenate([arr.flatten() for sublist in hists for arr in sublist])
    h = h[h >= 0.5]
    plt.figure()
    plt.hist(h, bins=200)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Amount")
    plt.savefig(filename)
    plt.clf()


def plot_histograms(collectors, main_dir):
    """Plot histograms for RPW and ECG test results."""
    if collectors['rpw_test'].hists:
        plot_hist(collectors['rpw_test'].hists, "Relevance Histogram RPW", f"{main_dir}/rpw_hist.png")
    if collectors['ecg_test'].hists:
        plot_hist(collectors['ecg_test'].hists, "Relevance Histogram ECG", f"{main_dir}/ecg_hist.png")


def get_rpw_validation_file_list(fold, files_train_path):
    """Get the list of RPW validation files for a given fold."""
    labels = load_rpw_labels()
    record_ids = find_records(files_train_path)

    S = []

    for i, recid in enumerate(record_ids):
        c = labels[recid.split("_")[0]]
        S.append([c])

    _, val_record_ids, _, _ = rpw_patients_split_cross_validation(recids=np.array(record_ids),
                                                                  labels=np.array(S),
                                                                  fold=fold)
    return val_record_ids


def get_ecg_validation_file_list(fold, files_train_path):
    """Get the list of ECG validation files for a given fold."""
    labels = load_ecg_labels()
    record_ids = find_records(files_train_path)

    S = []

    for i, recid in enumerate(record_ids):
        c = labels[f"{recid}.npy"]
        S.append([c])

    _, val_record_ids, _, _ = ecg_patients_split_cross_validation(recids=np.array(record_ids),
                                                                  labels=np.array(S),
                                                                  fold=fold)
    return val_record_ids


def compute_results(collector, fold, files_path, files_list, model_path, compute_explanations, plot_dir):
    """Compute and collect results for a given model and data split."""
    if "rpw" in collector.name.lower():
        result_dict_subsample, hist = run_rpw_model(rpw_model_path=f"{model_path}/{fold}", rpw_files_path=files_path,
                                                    rpw_files_list=files_list, plot_explanations=compute_explanations,
                                                    plot_dir=plot_dir)
    else:
        result_dict_subsample, hist = run_ecg_model(ecg_model_path=f"{model_path}/{fold}", ecg_files_path=files_path,
                                                    ecg_files_list=files_list, plot_explanations=compute_explanations,
                                                    plot_dir=plot_dir)

    if "rpw" in collector.name.lower():
        # Aggregate RPW (Group by 4)
        result_dict_sample = aggregate_subsamples(results_dict=result_dict_subsample,
                                                  key_func=lambda f: f.split('_')[0])
    else:
        # Aggregate ECG (Group by 5)
        result_dict_sample = aggregate_subsamples(results_dict=result_dict_subsample,
                                                  key_func=lambda f: f"{f.split('_')[0]}_{f.split('_')[1]}")

    (roc_sample, auc_sample, sensitivity_sample, specificity_sample, f1_score_sample)\
        = calculate_metrics_from_dict(result_dict=result_dict_sample)

    collector.add_fold_metrics(hist=hist, result_dict_sample=result_dict_sample,
                               roc_sample=roc_sample, title_sample=f"Fold {fold}", auc_sample=auc_sample,
                               sensitivity_sample=sensitivity_sample, specificity_sample=specificity_sample,
                               f1_score_sample=f1_score_sample)

    return result_dict_sample


def generate_plots(collectors, main_dir):
    """Generate and save result plots."""
    collectors_to_plot = []
    if len(collectors['rpw_val'].aucs_sample) == config.num_folds:
        collectors_to_plot.append(collectors['rpw_val'])
    else:
        logging.error("Incomplete folds for rpw_val")
    if len(collectors['ecg_val'].aucs_sample) == config.num_folds:
        collectors_to_plot.append(collectors['ecg_val'])
    else:
        logging.error("Incomplete folds for ecg_val")
    if len(collectors['combined_val'].aucs_sample) == config.num_folds:
        collectors_to_plot.append(collectors['combined_val'])
    else:
        logging.error("Incomplete folds for combined_val")
    if len(collectors['rpw_test'].aucs_sample) == config.num_folds + 1:
        collectors_to_plot.append(collectors['rpw_test'])
    else:
        logging.error("Incomplete folds for rpw_test")
    if len(collectors['ecg_test'].aucs_sample) == config.num_folds + 1:
        collectors_to_plot.append(collectors['ecg_test'])
    else:
        logging.error("Incomplete folds for ecg_test")
    if len(collectors['combined_test'].aucs_sample) == config.num_folds + 1:
        collectors_to_plot.append(collectors['combined_test'])
    else:
        logging.error("Incomplete folds for combined_test")

    if not collectors_to_plot:
        logging.warning("No data available for plotting.")
        return

    generate_results(collectors_to_plot, main_dir)


def evaluate_models(main_dir, rpw_model_dir, ecg_model_dir, explanations=False, histogram=False):
    """Evaluate trained models and generate results."""

    if not os.path.exists(main_dir):
        os.makedirs(main_dir)

    collectors = {
        'rpw_val': MetricCollector("RPW Validation"),
        'ecg_val': MetricCollector("ECG Validation"),
        'combined_val': MetricCollector("Combined Validation"),
        'rpw_test': MetricCollector("RPW Test"),
        'ecg_test': MetricCollector("ECG Test"),
        'combined_test': MetricCollector("Combined Test")
    }

    for sub_dir in range(1, config.num_folds + 1):
        print(f"Processing Fold {sub_dir}...")

        # RPW Validation
        rpw_files_train_path = f"{config.directory}/data/train/rpw/"
        rpw_files_val_list = get_rpw_validation_file_list(sub_dir, rpw_files_train_path)

        rpw_val_result_dict = compute_results(collector=collectors['rpw_val'], fold=sub_dir,
                                              files_path=rpw_files_train_path, files_list=rpw_files_val_list,
                                              model_path=rpw_model_dir, compute_explanations=False, plot_dir="")

        # ECG Validation
        ecg_files_train_path = f"{config.directory}/data/train/ecg/"
        ecg_files_val_list = get_ecg_validation_file_list(sub_dir, ecg_files_train_path)

        ecg_val_result_dict = compute_results(collector=collectors['ecg_val'], fold=sub_dir,
                                              files_path=ecg_files_train_path, files_list=ecg_files_val_list,
                                              model_path=ecg_model_dir, compute_explanations=False, plot_dir="")

        # Combined Validation
        combined_val_result_dict = combine_rpw_ecg_models(rpw_dict=rpw_val_result_dict, ecg_dict=ecg_val_result_dict)

        (combined_val_roc, combined_val_auc, combined_val_sensitivity, combined_val_specificity,
         combined_val_f1_score) = calculate_metrics_from_dict(combined_val_result_dict)

        collectors['combined_val'].add_fold_metrics(hist=None,
                                                result_dict_sample=combined_val_result_dict,
                                                roc_sample=combined_val_roc,
                                                title_sample=f"Fold {sub_dir}",
                                                auc_sample=combined_val_auc,
                                                sensitivity_sample=combined_val_sensitivity,
                                                specificity_sample=combined_val_specificity,
                                                f1_score_sample=combined_val_f1_score)

        # RPW Test
        rpw_files_test_path = f"{config.directory}/data/test/rpw/"
        rpw_files_test_list = find_records(rpw_files_test_path)

        rpw_test_result_dict = compute_results(collector=collectors['rpw_test'], fold=sub_dir,
                                               files_path=rpw_files_test_path, files_list=rpw_files_test_list,
                                               model_path=rpw_model_dir, compute_explanations=explanations,
                                               plot_dir=f"{main_dir}/xai_plots/rpw/{sub_dir}")

        # ECG Test
        ecg_files_test_path = f"{config.directory}/data/test/ecg/"
        ecg_files_test_list = find_records(ecg_files_test_path)

        ecg_test_result_dict = compute_results(collector=collectors['ecg_test'], fold=sub_dir,
                                               files_path=ecg_files_test_path, files_list=ecg_files_test_list,
                                               model_path=ecg_model_dir, compute_explanations=explanations,
                                               plot_dir=f"{main_dir}/xai_plots/ecg/{sub_dir}")

        # Combined Test
        combined_test_result_dict = combine_rpw_ecg_models(rpw_dict=rpw_test_result_dict, ecg_dict=ecg_test_result_dict)

        # Metrics for Samples
        (combined_test_roc, combined_test_auc, combined_test_sensitivity, combined_test_specificity,
         combined_test_f1_score) = calculate_metrics_from_dict(combined_test_result_dict)

        collectors['combined_test'].add_fold_metrics(hist=None,
                                                result_dict_sample=combined_test_result_dict,
                                                roc_sample=combined_test_roc,
                                                title_sample=f"Fold {sub_dir}",
                                                auc_sample=combined_test_auc,
                                                sensitivity_sample=combined_test_sensitivity,
                                                specificity_sample=combined_test_specificity,
                                                f1_score_sample=combined_test_f1_score)

    combine_fold_models_to_ensemble(collectors['rpw_test'])
    combine_fold_models_to_ensemble(collectors['ecg_test'])
    combine_fold_models_to_ensemble(collectors['combined_test'])

    rpw_ens = collectors['rpw_test'].result_dicts_sample[-1]
    ecg_ens = collectors['ecg_test'].result_dicts_sample[-1]
    rpw_ecg_map = load_rpw_ecg_map()
    with open(f"{main_dir}/test_predictions.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["participant_id", "y", "p_rpw", "p_ecg", "p_comb"])
        for rpw_key in rpw_ens.keys():
            ecg_filename = rpw_ecg_map[rpw_key].split(".")[0]
            if ecg_filename in ecg_ens:
                y_val = rpw_ens[rpw_key][1]
                p_rpw_val = rpw_ens[rpw_key][0]
                p_ecg_val = ecg_ens[ecg_filename][0]
                y_scalar = y_val[1] if isinstance(y_val, (np.ndarray, list)) and len(y_val) > 1 else y_val
                p_rpw_scalar = p_rpw_val[1] if isinstance(p_rpw_val, (np.ndarray, list)) and len(
                    p_rpw_val) > 1 else p_rpw_val
                p_ecg_scalar = p_ecg_val[1] if isinstance(p_ecg_val, (np.ndarray, list)) and len(
                    p_ecg_val) > 1 else p_ecg_val
                p_comb_scalar = (p_rpw_scalar + p_ecg_scalar) / 2
                writer.writerow([rpw_key, y_scalar, p_rpw_scalar, p_ecg_scalar, p_comb_scalar])

    verify_test_data(f"{main_dir}/test_predictions.csv")
    run_bootstrap(f"{main_dir}/test_predictions.csv", n_boot=10000)

    generate_plots(collectors, main_dir)

    if histogram:
        plot_histograms(collectors, main_dir)


def extract_validation_aucs(output_dir, model_type):
    """Extract validation AUC scores for all models and find the best one."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    models_dir = f"{config.directory}/models/{model_type}"
    if not os.path.exists(models_dir):
        print(f"Directory {models_dir} does not exist.")
        return None

    model_names = os.listdir(models_dir)
    results = []

    for model_name in model_names:
        model_path = os.path.join(models_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        fold_aucs = []
        for fold in range(1, config.num_folds + 1):
            path_excel = f"{model_path}/{fold}/logs/TRUE/sample_level/results_filtered.xlsx"
            if os.path.exists(path_excel):
                try:
                    df = pd.read_excel(path_excel)
                    if 'AUC' in df.columns:
                        auc_val = df['AUC'].values[0]
                        fold_aucs.append(auc_val)
                    else:
                        fold_aucs.append(None)
                except Exception as e:
                    print(f"Error reading {path_excel}: {e}")
                    fold_aucs.append(None)
            else:
                fold_aucs.append(None)

        valid_aucs = [x for x in fold_aucs if x is not None and not np.isnan(x)]
        mean_auc = np.mean(valid_aucs) if len(valid_aucs) > 0 else np.nan

        while len(fold_aucs) < config.num_folds:
            fold_aucs.append(None)

        results.append({
            'model_name': model_name,
            'fold_1': fold_aucs[0],
            'fold_2': fold_aucs[1],
            'fold_3': fold_aucs[2],
            'fold_4': fold_aucs[3],
            'fold_5': fold_aucs[4],
            'mean_auc': mean_auc
        })

    best_model_name = None
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by='mean_auc', ascending=False)
        best_model_name = df_results.iloc[0]['model_name']
        output_csv = f"{output_dir}/val_aucs_{model_type}.csv"
        df_results.to_csv(output_csv, index=False)
        print(f"Saved evaluation to {output_csv}")
    else:
        print(f"No models found for {model_type}")

    return best_model_name


if __name__ == '__main__':
    print("Extracting validation AUCs for RPW models...")
    best_rpw_model = extract_validation_aucs(output_dir=f"{config.directory}/results", model_type="rpw")
    print("Extracting validation AUCs for ECG models...")
    best_ecg_model = extract_validation_aucs(output_dir=f"{config.directory}/results", model_type="ecg")

    evaluate_models(main_dir=f"{config.directory}/results",
                    rpw_model_dir=f"{config.directory}/models/rpw/{best_rpw_model}",
                    ecg_model_dir=f"{config.directory}/models/ecg/{best_ecg_model}",
                    explanations=True, histogram=True)
