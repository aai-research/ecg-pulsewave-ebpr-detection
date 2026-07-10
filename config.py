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
xai_method = "lrpsign_epsilon_0_5_std_x"  # Explainable AI (XAI) method
posthresh = 0.75  # Threshold for XAI