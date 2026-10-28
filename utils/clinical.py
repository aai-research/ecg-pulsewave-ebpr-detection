"""Clinical reference model (logistic regression on routinely available resting variables)."""
import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


def build_clinical_pipeline():
    """
    Build the clinical model pipeline. Imputer and scaler are fitted on training data only.

    Returns:
        sklearn.pipeline.Pipeline: The (unfitted) pipeline.
    """
    steps = []
    if config.clinical_missing == "impute":
        steps.append(("imputer", SimpleImputer(strategy="median")))
    steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", LogisticRegression(class_weight=config.clinical_class_weight)))
    return Pipeline(steps)


def _prepare(df):
    """Apply the configured missing-value strategy to a subset of labels.csv."""
    if config.clinical_missing == "drop":
        return df.dropna(subset=config.clinical_features)
    if config.clinical_missing == "impute":
        return df
    raise ValueError(f"Unknown clinical_missing option: {config.clinical_missing!r} (use 'impute' or 'drop')")


def _to_result_dict(df, probs):
    """Convert predictions to the pipeline format {SessionID: [pred, label]}."""
    result_dict = {}
    for session_id, label, p in zip(df["SessionID"], df["Label"], probs):
        result_dict[session_id] = [np.array([1 - p, p]), [0, 1] if label == 1 else [1, 0]]
    return result_dict


def run_clinical_model(fold):
    """
    Train the clinical model on the training folds and predict the validation fold and the test set.

    Args:
        fold (int): The validation fold number.

    Returns:
        tuple: (val_result_dict, test_result_dict), both {SessionID: [pred, label]}.
    """
    df = pd.read_csv(f"{config.directory}/data/labels.csv")
    df_trainval = df[df["Split"] == "trainval"]

    df_train = _prepare(df_trainval[df_trainval["Fold"] != fold])
    df_val = _prepare(df_trainval[df_trainval["Fold"] == fold])
    df_test = _prepare(df[df["Split"] == "test"])

    model = build_clinical_pipeline()
    model.fit(df_train[config.clinical_features].values, df_train["Label"].astype(int).values)

    val_probs = model.predict_proba(df_val[config.clinical_features].values)[:, 1]
    test_probs = model.predict_proba(df_test[config.clinical_features].values)[:, 1]

    logging.info("Clinical model fold %d (%s): train=%d, val=%d, test=%d examinations",
                 fold, config.clinical_missing, len(df_train), len(df_val), len(df_test))

    return _to_result_dict(df_val, val_probs), _to_result_dict(df_test, test_probs)


def restrict_to_complete_cases(result_dict):
    """
    Keep only examinations with complete clinical features (config.clinical_features).

    Works for RPW/combined/clinical dicts (keyed by SessionID) and ECG dicts (keyed by ecg_file without .npy),
    so that all models are evaluated on the identical set of examinations.

    Args:
        result_dict (dict): {record_id: [pred, label]}

    Returns:
        dict: The filtered result dict.
    """
    df = pd.read_csv(f"{config.directory}/data/labels.csv")
    df = df[df[config.clinical_features].notna().all(axis=1)]
    allowed = set(df["SessionID"]) | {f.split(".")[0] for f in df["ecg_file"]}
    filtered = {k: v for k, v in result_dict.items() if k in allowed}
    if len(filtered) < len(result_dict):
        logging.info("Complete-case restriction: %d of %d examinations kept", len(filtered), len(result_dict))
    return filtered


def write_missing_value_report(path):
    """
    Save the number of missing values per clinical variable (development, test, overall) as .csv.

    Args:
        path (str): Output directory.
    """
    df = pd.read_csv(f"{config.directory}/data/labels.csv")
    parts = {"Development": df[df["Split"] == "trainval"], "Test": df[df["Split"] == "test"], "Overall": df}
    report = pd.DataFrame({name: part[config.clinical_features].isna().sum() for name, part in parts.items()})
    report.loc["complete_cases"] = [part[config.clinical_features].notna().all(axis=1).sum() for part in parts.values()]
    report.loc["examinations"] = [len(part) for part in parts.values()]
    report.to_csv(f"{path}/clinical_missing_values.csv", index_label="variable")
    print(f"Clinical missing values saved to {path}/clinical_missing_values.csv")
