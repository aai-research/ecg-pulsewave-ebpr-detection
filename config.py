"""Global configuration for the eBPR detection pipeline."""

seed = 0  # Random seed for reproducibility
directory = "experiments_npj_2026"  # Base directory for storing experiments and data
matching_csv = "matching.csv"  # CSV file used for matching patient records
data_excel = "data.xlsx"  # Main Excel file containing athlete data
sheet_name = "sheet_name"  # Target sheet name in the Excel data file
num_rows = 654  # Expected number of rows to read from the Excel file
rpw_data_csv = "rpw_data.csv"  # CSV file for RPW recording data
ecg_dates_csv = "ecg_dates.csv"  # CSV file containing ECG recording dates
num_folds = 5  # Number of folds for cross-validation
ebpr_threshold = 6.2  # eBPR label: SBP/MET slope > ebpr_threshold (mmHg/MET)
classification_threshold = 0.5  # Predicted probability above which a sample is classified as eBPR
ecg_record_length = 2500  # Length l of the 12-lead ECG recordings (samples, 5 s at 500 Hz)
ecg_window_size = 2000  # Length w of the ECG input windows (samples, 4 s at 500 Hz)
ecg_num_windows = 5  # Number n of overlapping ECG windows per recording
n_bootstrap = 10000  # Number of iterations of the paired athlete-level bootstrap (CIs and AUC differences)
xai_method = "lrpsign_epsilon_0_5_std_x"  # Explainable AI (XAI) method
posthresh = 0.75  # Threshold for XAI
clinical_features = ["age", "bmi", "hr_rest", "sys_rest", "dia_rest", "train_years", "train_weekly"]  # Columns of labels.csv
clinical_missing = "impute"  # "impute": median imputation (fitted on training folds) | "drop": exclude incomplete examinations
clinical_class_weight = "balanced"  # "balanced": class weighting analogous to the CNNs (derive_class_weight) | None: sklearn default
clinical_complete_cases_only = False # True: evaluate ALL models (RPW, ECG, Combined, Clinical) only on examinations with complete clinical features
highlight_revision = False  # True: colour changed content in the LaTeX tables (\textcolor, requires \usepackage{xcolor})
highlight_color = "blue"  # Colour used when highlight_revision is True

# Output folder of evaluate_results.py (inside `directory`). None: name derived from the clinical settings,
# e.g. "results_impute_balanced_all" -> runs with different settings do not overwrite each other.
results_name = None


def get_results_dir():
    """Output directory of evaluate_results.py / input of global_xai.py and segment_xai.py."""
    name = results_name
    if name is None:
        cases = "complete" if clinical_complete_cases_only else "all"
        name = f"results_{clinical_missing}_{clinical_class_weight or 'noweight'}_{cases}"
        base = ["age", "bmi", "hr_rest", "sys_rest", "dia_rest"]
        if list(clinical_features) != base:  # other clinical variables -> own folder
            name += f"_{len(clinical_features)}vars"
    return f"{directory}/{name}"
