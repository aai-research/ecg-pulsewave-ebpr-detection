"""
Splitting utility module for cross-validation data partitioning.
"""
import pandas as pd

import config


def ecg_patients_split_cross_validation(recids, labels, fold):
    """
    Perform a cross-validation split for ECG patient data.

    Args:
        recids (np.ndarray): Array of record IDs.
        labels (np.ndarray): Array of labels corresponding to the records.
        fold (int): The validation fold number.

    Returns:
        tuple: (train_recids, val_recids, train_labels, val_labels)
    """
    df = pd.read_csv(f"{config.directory}/data/labels.csv")
    df_trainval = df[df["Split"] == "trainval"].copy()

    val_files = df_trainval.loc[df_trainval["Fold"] == fold, "ecg_file"].values
    train_files = df_trainval.loc[df_trainval["Fold"] != fold, "ecg_file"].values

    val_index = [i for i, r in enumerate(recids) if f"{r}.npy" in val_files]
    train_index = [i for i, r in enumerate(recids) if f"{r}.npy" in train_files]

    return recids[train_index], recids[val_index], labels[train_index], labels[val_index]


def rpw_patients_split_cross_validation(recids, labels, fold):
    """
    Perform a cross-validation split for RPW patient data.

    Args:
        recids (np.ndarray): Array of record IDs.
        labels (np.ndarray): Array of labels corresponding to the records.
        fold (int): The validation fold number.

    Returns:
        tuple: (train_recids, val_recids, train_labels, val_labels)
    """
    df = pd.read_csv(f"{config.directory}/data/labels.csv")
    df_trainval = df[df["Split"] == "trainval"].copy()

    val_files = df_trainval.loc[df_trainval["Fold"] == fold, "SessionID"].values
    train_files = df_trainval.loc[df_trainval["Fold"] != fold, "SessionID"].values

    val_index = [i for i, r in enumerate(recids) if r.split("_")[0] in val_files]
    train_index = [i for i, r in enumerate(recids) if r.split("_")[0] in train_files]

    return recids[train_index], recids[val_index], labels[train_index], labels[val_index]
