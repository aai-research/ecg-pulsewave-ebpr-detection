"""Evaluation utilities for parsing, filtering, formatting and plotting model metrics."""
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from utils.metrics import is_low_value_good_for_metric
from utils.data import load_dict_from_json


def create_row(results_raw, epoch):
    """
    Create a flattened dictionary row from raw result metrics.

    Args:
        results_raw (dict): The raw results dictionary.
        epoch (int): The epoch number.

    Returns:
        dict: A flattened dictionary representing one row of results.
    """
    line = {}

    results_metrics = results_raw['metrics']

    line['e'] = epoch

    for m in results_metrics:
        line[m] = results_metrics[m]

    if 'classification' in results_raw:
        results_classification = results_raw['classification']

        for c in results_classification:
            line['{}_classification'.format(c)] = results_classification[c]

    return line


def load_results_for_experiment(logdir, lbl, calculation_method, epochs):
    """
    Load and parse results from a JSON log file into a DataFrame.

    Args:
        logdir (str): The directory containing 'results_all.json'.
        lbl (str): The label key to extract.
        calculation_method (str): The calculation method key to extract.
        epochs (int): The total number of epochs.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed results per epoch.
    """
    rows = []

    results_all = load_dict_from_json('{}/results_all.json'.format(logdir))

    for e in range(epochs):
        epoch_id = 'e{}'.format(e)
        results_raw = results_all[epoch_id][lbl][calculation_method]
        rows.append(create_row(results_raw, e))

    return pd.DataFrame(rows)


def filter_df_based_on_metric_thresholds(df, metric_thresholds):
    """
    Filter a DataFrame based on given metric thresholds.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        metric_thresholds (dict): A dictionary of metric names and their cutoff thresholds.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if metric_thresholds is not None:
        for m in metric_thresholds:
            thresh = metric_thresholds[m]
            if is_low_value_good_for_metric(m):
                df = df[df[m] < thresh]
            else:
                df = df[df[m] > thresh]

    return df


def filter_results_based_on_metrics(df, target_metric, metric_thresholds):
    """
    Filter results dynamically by adjusting thresholds until a valid configuration is found.

    Args:
        df (pd.DataFrame): The results DataFrame.
        target_metric (str): The primary metric used to sort the filtered results.
        metric_thresholds (dict): A dictionary of starting threshold constraints.

    Returns:
        pd.DataFrame: The top result row matching the constraints.
    """
    df_tmp = pd.DataFrame(df)

    while True:
        df_f = filter_df_based_on_metric_thresholds(df_tmp, metric_thresholds)
        df_f = df_f.sort_values(by=target_metric, ascending=is_low_value_good_for_metric(target_metric))

        if len(df_f) > 0:
            break
        else:
            df_tmp = pd.DataFrame(df)

            for m in metric_thresholds:
                if is_low_value_good_for_metric(target_metric):
                    metric_thresholds[m] += 0.1
                else:
                    metric_thresholds[m] -= 0.1

            logging.info('Adjusting thresholds to: {}'.format(metric_thresholds))

    return df_f.head(1)


def extract_rocs_from_df(df, title=None):
    """
    Extract ROC curves and generate titles from a results DataFrame.

    Args:
        df (pd.DataFrame): The results DataFrame containing a 'ROC' column.
        title (str, optional): An optional title override.

    Returns:
        tuple: A tuple containing a list of ROC dictionaries and a list of string titles.
    """
    rocs = list(df['ROC'])

    if title is None:
        roc_titles = ['{}, AUC: {:.2f}'.format(e, a) for e, a in zip(df['e'], df['AUC'])]
    else:
        roc_titles = ['{}, AUC: {:.2f}'.format(title, a) for a in df['AUC']]

    return rocs, roc_titles


def plot_roc_to_pdf(roc, roc_title, path):
    """
    Plot a single ROC curve and save it as a PDF.

    Args:
        roc (dict): A dictionary containing 'FPR' and 'TPR' lists.
        roc_title (str): The title for the ROC plot.
        path (str): The filepath where the PDF should be saved.
    """
    fprs = list(roc['FPR'])
    tprs = list(roc['TPR'])

    fprs.append(0.0)
    fprs.append(1.0)
    tprs.append(0.0)
    tprs.append(1.0)

    fprs = sorted(fprs)
    tprs = sorted(tprs)

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.plot(fprs, tprs, 'r-')
    plt.xlim((-0.02, 1.02))
    plt.ylim((-0.02, 1.02))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(roc_title)
    plt.tight_layout()

    plt.savefig(path)
    plt.close()


def generate_results(collectors, path):
    """
    Generate and save consolidated result tables and ROC plots from multiple model collectors.

    Args:
        collectors (list): A list of objects containing ROC and metric data per fold.
        path (str): The directory path to save outputs (PDFs and TeX files).
    """

    colors = [
        "indigo",
        "purple",
        "cornflowerblue",
        "lightsalmon",
        "green",
        "red",
        "yellow",
        "lightgray",
        "darkgray"
    ]

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)

    axes = []
    ax0 = None
    for i in range(min(len(collectors), 6)):
        if ax0 is None:
            ax0 = fig.add_subplot(gs[i // 3, i % 3])
            axes.append(ax0)
        else:
            axes.append(fig.add_subplot(gs[i // 3, i % 3], sharex=ax0, sharey=ax0))

    consolidated_rows = []

    for ax, collector in zip(axes, collectors):
        title = collector.name

        def plot_rocs(rocs, titles):
            if not rocs:
                return

            n_folds = min(len(rocs), config.num_folds)

            if n_folds < config.num_folds:
                logging.warning("Number of ROC curves (%d) is less than expected folds (%d)", n_folds, config.num_folds)
            
            tprs_interp = []
            mean_fpr = np.linspace(0, 1, 100)

            for n in range(n_folds):
                roc = rocs[n]
                if roc is None:
                    logging.warning("ROC curve for fold %d is None", n)
                    continue

                fprs = list(roc['FPR'])
                tprs = list(roc['TPR'])

                tpr_interp = np.interp(mean_fpr, fprs, tprs)
                tpr_interp[0] = 0.0
                tprs_interp.append(tpr_interp)
                
                fprs = [0.0] + fprs + [1.0]
                tprs = [0.0] + tprs + [1.0]
                fprs, tprs = zip(*sorted(zip(fprs, tprs)))

                label = titles[n]

                ax.plot(fprs, tprs, label=label, alpha=0.8, linewidth=1, color=colors[n], linestyle='-')

            mean_tpr = np.mean(tprs_interp, axis=0)
            mean_tpr[-1] = 1.0

            ax.plot(mean_fpr, mean_tpr, label="Average", alpha=1, linewidth=3, color=colors[n_folds],
                    linestyle='-')

            if len(rocs) == config.num_folds + 1:
                roc = rocs[config.num_folds]
                if roc is not None:
                    fprs = list(roc['FPR'])
                    tprs = list(roc['TPR'])
                    fprs = [0.0] + fprs + [1.0]
                    tprs = [0.0] + tprs + [1.0]
                    fprs, tprs = zip(*sorted(zip(fprs, tprs)))

                    label = titles[n_folds]
                    ax.plot(fprs, tprs, label=label, alpha=1, linewidth=3, color=colors[n_folds+1],
                            linestyle='-')
                else:
                    logging.warning("Ensemble ROC curve is None")

            std_tpr = np.std(tprs_interp, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[n_folds+2], alpha=0.5,
                            label=f'$\pm$ 1 std. dev.')

            ax.plot([0, 1], [0, 1], label="Chance", alpha=0.8, linestyle='--', color=colors[n_folds+3])

        plot_rocs(collector.rocs_sample, collector.titles_sample)

        ax.set_xlim((-0.05, 1.05))
        ax.set_ylim((-0.05, 1.05))
        ax.set_xlabel('False Positive Rate', fontsize=15)
        ax.set_ylabel('True Positive Rate', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_title(title, fontsize=16)
        ax.label_outer()

        def collect_metrics(suffix):
            aucs = getattr(collector, f"aucs_{suffix}")
            sens = getattr(collector, f"sensitivities_{suffix}")
            specs = getattr(collector, f"specificities_{suffix}")
            f1s = getattr(collector, f"f1_scores_{suffix}")

            if not aucs: return

            metrics_data = {
                'AUC': aucs,
                'Sensitivity': sens,
                'Specificity': specs,
                'F1-Score': f1s
            }

            n_folds = min(len(aucs), config.num_folds)

            for metric_name, values in metrics_data.items():
                if not values: continue

                fold_values = values[:n_folds]
                valid_fold_values = [v for v in fold_values if v is not None]

                if len(valid_fold_values) != config.num_folds:
                    logging.warning("Number of valid fold values (%d) for %s does not match expected folds (%d)", len(valid_fold_values), metric_name, config.num_folds)

                if valid_fold_values:
                    mean_val = np.mean(valid_fold_values)
                    std_val = np.std(valid_fold_values)
                    mean_sd_str = f"{mean_val:.2f} $\pm$ {std_val:.2f}"
                else:
                    mean_sd_str = "-"

                r = {
                    'Model': title,
                    'Metric': metric_name,
                    'Dataset': title.split(" ")[1],
                    'Mean $\pm$ SD': mean_sd_str
                }

                for fold_idx in range(n_folds):
                    if fold_idx < n_folds:
                        val = values[fold_idx]
                        r[str(fold_idx + 1)] = f"{val:.2f}"
                    else:
                        r[str(fold_idx + 1)] = "-"

                if len(values) == config.num_folds + 1:
                    val = values[config.num_folds]
                    r['Ensemble'] = f"{val:.2f}"
                else:
                    r['Ensemble'] = "-"

                consolidated_rows.append(r)

        collect_metrics("sample")

    bootstrap_df = pd.read_csv(f'{path}/bootstrap.csv')
    bootstrap_dict = {}
    metric_map = {'AUC': 'AUC', 'SENS': 'Sensitivity', 'SPEC': 'Specificity', 'F1': 'F1-Score'}
    model_map = {'RPW': 'RPW', 'ECG': 'ECG', 'COMB': 'Combined'}
    for _, row in bootstrap_df.iterrows():
        mod = model_map.get(row['Model'])
        met = metric_map.get(row['Metric'])
        if mod and met:
            val_str = f"{row['CI_2.5']:.2f}--{row['CI_97.5']:.2f}"
            bootstrap_dict[(mod, met)] = val_str

    consolidated_path = f"{path}/results_table.tex"
    with open(consolidated_path, 'w') as f:
        f.write(r"\begin{tabular}{lllcccccccc}" + "\n")
        f.write(r"\toprule" + "\n")
        titles_list = ["RPW", "ECG", "Combined"]
        f.write(
            r" & \multirow{2}{*}{\textbf{Metric}} & \multirow{2}{*}{\textbf{Dataset}} & \multicolumn{5}{c}{\makecell{\textbf{Group-stratified 5-fold} \\ \textbf{Cross-validation Split}}} & \multirow{2}{*}{\textbf{Mean $\pm$ SD}} & \multirow{2}{*}{\textbf{Ensemble}} & \multirow{2}{*}{\textbf{95\% CI}} \\" + "\n")
        f.write(r"\cmidrule(lr){4-8}" + "\n")
        f.write(r" & & & \textbf{1} & \textbf{2} & \textbf{3} & \textbf{4} & \textbf{5} & & & \\" + "\n")
        f.write(r"\midrule" + "\n")

        for i, title in enumerate(titles_list):
            if i > 0:
                f.write(r"\midrule" + "\n")

            model_rows = [r for r in consolidated_rows if r['Model'].split(" ")[0] == title]

            metrics_order = ['AUC', 'Sensitivity', 'Specificity', 'F1-Score']

            total_rows = 0
            for metric in metrics_order:
                metric_rows = [r for r in model_rows if r['Metric'] == metric]
                total_rows += len(metric_rows)

            first_metric = True
            for m_idx, metric in enumerate(metrics_order):
                metric_rows = [r for r in model_rows if r['Metric'] == metric]
                if not metric_rows: continue

                if m_idx > 0:
                    f.write(r"\cmidrule(lr){2-11}" + "\n")

                for r_idx, row in enumerate(metric_rows):
                    dataset = row['Dataset']
                    folds = [row.get(str(k), '-') for k in range(1, 6)]
                    ensemble = row.get('Ensemble', '-')
                    mean_sd = row.get('Mean $\pm$ SD', '-')
                    boot_val = bootstrap_dict.get((title, metric), '-') if ensemble != '-' else '-'
                    if first_metric and r_idx == 0:
                        line_start = rf"\multirow{{{total_rows}}}{{*}}{{\rotatebox[origin=c]{{90}}{{\textbf{{{title}}}}}}} & "
                    else:
                        line_start = "& "
                    if r_idx == 0:
                        line_start += rf"\multirow{{{len(metric_rows)}}}{{*}}{{{metric}}} & "
                    else:
                        line_start += "& "
                    if title == "Combined":
                        ensemble_str = f"\\textbf{{{ensemble}}}"
                    else:
                        ensemble_str = ensemble
                    f.write(
                        f"{line_start} {dataset} & {' & '.join(folds)} & {mean_sd} & {ensemble_str} & {boot_val} \\\\" + "\n")
                
                first_metric = False

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
    print(f"Results Table saved to {consolidated_path}")

    handles, labels = [], []

    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if len(l) > len(labels):
            handles, labels = h, l

    if handles:
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.99, 0.5), fontsize=15)

    plot_path = f"{path}/roc_plots.pdf"
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")


def plot_rocs_to_pdf(rocs, roc_titles, path, filename):
    """
    Plot multiple ROC curves to separate PDF files.

    Args:
        rocs (list): A list of ROC dictionaries.
        roc_titles (list): A list of corresponding titles.
        path (str): The directory to save the plots.
        filename (str): The base filename for the exported PDFs.
    """
    filepaths = []

    for roc, roc_title, i in zip(rocs, roc_titles, range(len(rocs))):
        filepath = '{}/{}_{}.pdf'.format(path, filename, i)
        plot_roc_to_pdf(roc, roc_title, filepath)
        filepaths.append(filepath)


def calculate_mean_values_for_filtered_results(df, metrics):
    """
    Calculate the mean and standard deviation for a set of metrics across rows in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing metric columns.
        metrics (list): A list of metric column names to evaluate.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated mean and standard deviation values.
    """
    means = {'value': 'mean'}
    stddevs = {'value': 'stddev'}

    for m in metrics:
        means[m] = np.mean(df[m])
        stddevs[m] = np.std(df[m])

    df_ret = pd.DataFrame([means, stddevs])
    df_ret = df_ret.set_index('value')

    return df_ret


def evaluate_experiment(logdir, y_labels, calculation_methods, metrics, target_metric, metric_thresholds, save_raw_results, epochs):
    """
    Evaluate experiment results over multiple epochs, saving metrics and ROC plots.

    Args:
        logdir (str): The base directory for experiment logs.
        y_labels (list): A list of labels evaluated.
        calculation_methods (list): A list of evaluation methodologies (e.g., sample_level).
        metrics (list): A list of metrics to track.
        target_metric (str): The main metric used to choose the best epoch.
        metric_thresholds (dict): Threshold requirements for valid epochs.
        save_raw_results (bool): Whether to export raw data to Excel.
        epochs (int): Number of epochs trained.

    Returns:
        dict: A dictionary mapping calculation_methods to their respective best epoch indices.
    """
    keep_e = []
    best_e = {}

    for lbl in y_labels:
        for calculation_method in calculation_methods:
            # Create subdirectories for classes and calculation methods
            subdir = '{}/{}/{}'.format(logdir, lbl, calculation_method)
            os.makedirs(subdir, exist_ok=True)

            # Load raw results from .json files into dataframe
            logging.info('Loading results from disk... ({})'.format(calculation_method))
            df = load_results_for_experiment(logdir, lbl, calculation_method, epochs)
            logging.info('Loading complete')
            if save_raw_results is True:
                logging.info('Saving as .xlsx')
                df.to_excel('{}/results_raw.xlsx'.format(subdir))
                logging.info('Saved raw results as .xlsx')

            # Plot metrics course
            plot_metrics_course_wrapper(df, metrics, subdir)

            # Filter results based on thresholds, target metric and validation method -> best n models
            df = filter_results_based_on_metrics(df, target_metric, metric_thresholds)
            e = df['e'].values[0]
            keep_e.append(e)
            best_e[calculation_method] = e
            logging.info('Filtered raw results based on metrics and thresholds')
            df.to_excel('{}/results_filtered.xlsx'.format(subdir))
            logging.info('Saved filtered results as .xlsx')

            # Calculate mean values for each metric
            df_mean = calculate_mean_values_for_filtered_results(df, metrics)
            logging.info('Calculated mean metrics')
            df_mean.to_excel('{}/mean_metrics.xlsx'.format(subdir))
            logging.info('Saved mean metrics as .xlsx')

            if 'TP' in df.columns and 'TN' in df.columns and 'FP' in df.columns and 'FN' in df.columns:
                # Calculate and plot ROC curves for each of the best n models
                rocs, roc_titles = extract_rocs_from_df(df)
                logging.info('Calculated ROCs')
                plot_rocs_to_pdf(rocs, roc_titles, subdir, 'ROCs')
                logging.info('Saved ROCs as .pdf')

    return best_e


def plot_metrics_course(df, metrics, subdir):
    """
    Plot the trajectory of metrics across epochs.

    Args:
        df (pd.DataFrame): The DataFrame containing epoch and metric data.
        metrics (list): List of metric column names to plot.
        subdir (str): The directory path to save the metrics course plot.
    """
    df.sort_values('e', ascending=True, inplace=True)

    plt.figure(figsize=(10, 10))

    for m in metrics:
        plt.plot(df['e'].values, df[m].values, label=m)

    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.xlim((0, max(df['e'].values)))
    plt.ylim((0, 1))
    plt.title('Metrics course')
    plt.tight_layout()
    plt.savefig('{}/metrics_course.pdf'.format(subdir))
    plt.close()


def plot_metrics_course_wrapper(df, metrics, subdir):
    """
    A wrapper function to plot metric courses, handling optional subgroup filtering by fold/k.

    Args:
        df (pd.DataFrame): The DataFrame containing epoch, subgroup ('k' or 'n'), and metric data.
        metrics (list): List of metric column names to plot.
        subdir (str): The directory path to save plots.
    """
    if 'k' in df.columns:
        filter_column = 'k'
    elif 'n' in df.columns:
        filter_column = 'n'
    else:
        filter_column = None

    if filter_column is not None:
        filter_values = set(df[filter_column].values)

        for fv in filter_values:
            df_filtered = df[df[filter_column] == fv]
            plot_metrics_course(df_filtered, metrics, subdir)

    else:
        plot_metrics_course(df, metrics, subdir)


def format_mean_std(series):
    """
    Format a Pandas Series into a string of its mean and standard deviation.

    Args:
        series (pd.Series): The data series to aggregate.

    Returns:
        str: The formatted string (e.g., "mean \\pm std (n=count)").
    """
    mean = series.mean()
    std = series.std()
    valid_n = series.count()
    return f"${mean:.2f} \\pm {std:.2f}$ (n={valid_n})"


def calculate_metrics(df):
    """
    Calculate summary demographics and baseline metrics from a cohort DataFrame.

    Args:
        df (pd.DataFrame): The cohort DataFrame containing participant variables.

    Returns:
        dict: A dictionary of formatted demographic and clinical metrics.
    """
    metrics = {}
    total_len = len(df)
    metrics['Examinations'] = str(total_len)
    metrics['Participants'] = str(df['PatID'].nunique())
    valid_n_ebpr = df['Label'].count()
    ebpr_pct = (df['Label'].dropna() == 1).mean() * 100
    metrics['Exhibiting eBPR'] = f"{ebpr_pct:.1f}\\% (n={valid_n_ebpr})"
    valid_n_male = df['gender'].count()
    male_pct = df['gender'].dropna().isin(values=['M', 'm', 'Male', 'male', 1]).mean() * 100
    metrics['Male'] = f"{male_pct:.2f}\\% (n={valid_n_male})"
    metrics['SBP/MET Slope'] = format_mean_std(series=df['sbp_met_slope'])
    metrics['Age'] = format_mean_std(series=df['age'])
    metrics['Height'] = format_mean_std(series=df['height'])
    metrics['Weight'] = format_mean_std(series=df['weight'])
    metrics['BMI'] = format_mean_std(series=df['bmi'])
    metrics['BSA'] = format_mean_std(series=df['bsa'])
    metrics['Train Years'] = format_mean_std(series=df['train_years'])
    metrics['Brachial SBP'] = format_mean_std(series=df['sys_rest'])
    metrics['Brachial DBP'] = format_mean_std(series=df['dia_rest'])
    metrics['Aortic SBP'] = format_mean_std(series=df['sys_aortic'])
    metrics['Aortic DBP'] = format_mean_std(series=df['dia_aortic'])
    metrics['Heart Rate Rest'] = format_mean_std(series=df['hr_rest'])
    metrics['Max. Heart Rate Exercise'] = format_mean_std(series=df['hr_max'])
    return metrics


def generate_cohort_table(path):
    """
    Generate a LaTeX table of patient cohort demographics and characteristics.

    Args:
        path (str): The directory containing the source labels.csv and where the TeX file will be saved.
    """
    df = pd.read_csv(f"{path}/labels.csv")
    df_dev = df[df['Split'] == 'trainval']
    df_test = df[df['Split'] == 'test']
    metrics_overall = calculate_metrics(df)
    metrics_dev = calculate_metrics(df_dev)
    metrics_test = calculate_metrics(df_test)
    output_path = f"{path}/cohort_table.tex"
    with open(output_path, 'w') as f:
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write(" & \\textbf{Overall} & \\textbf{Development} & \\textbf{Test} \\\\\n")
        f.write("\\midrule\n")
        for key in metrics_overall.keys():
            if key == "SBP/MET Slope":
                f.write("\\midrule\n")
            f.write(f"\\textbf{{{key}}} & {metrics_overall[key]} & {metrics_dev[key]} & {metrics_test[key]} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"Cohort Table saved to {output_path}")
