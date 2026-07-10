"""Data loader utilities for reading signals and labels."""
import numpy as np
import os
import pandas as pd
import wfdb
import logging

import config
from utils.data import subsample_ecg, normalize_sampling_rate_array_based


# Find the records in a folder and its subfolders.
def find_records(folder, header_extension='.hea'):
    """
    Find all records in a folder and its subfolders matching a given header extension.

    Args:
        folder (str): The root folder to search.
        header_extension (str): The file extension to match for header files.

    Returns:
        list: A sorted list of relative record paths without the header extension.
    """
    records = set()
    for root, directories, files in os.walk(folder):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == header_extension:
                record = os.path.relpath(os.path.join(root, file), folder)[:-len(header_extension)]
                records.add(record)
    records = sorted(records)
    return records


# Load the signals for a record.
def load_signals(record):
    """
    Load signals and fields for a given record.

    Args:
        record (str): The path to the record.

    Returns:
        tuple: A tuple containing the signal array and a dictionary of fields.
    """
    signal, fields = wfdb.rdsamp(record)
    return signal, fields


def load_rpw_labels():
    """
    Load resting pulse wave (RPW) labels from the CSV file.

    Returns:
        dict: A dictionary mapping SessionID to Label.
    """
    labels = pd.read_csv(f"{config.directory}/data/labels.csv")
    d = dict(zip(labels["SessionID"], labels[f"Label"]))
    return d


def load_ecg_labels():
    """
    Load ECG labels from the CSV file.

    Returns:
        dict: A dictionary mapping ecg_file to Label.
    """
    labels = pd.read_csv(f"{config.directory}/data/labels.csv")
    d = dict(zip(labels["ecg_file"], labels[f"Label"]))
    return d


def load_rpw_ecg_map():
    """
    Load the mapping between RPW and ECG files.

    Returns:
        dict: A dictionary mapping SessionID to ecg_file.
    """
    labels = pd.read_csv(f"{config.directory}/data/labels.csv")
    d = dict(zip(labels["SessionID"], labels["ecg_file"]))
    return d


def load_timeseries_data(recids, labels, subsampling_factor, window_size, data_folder, num_leads, sampling_rate,
                         num_cls, is_ecg):
    """
    Load timeseries data for a given set of record IDs and labels.

    Args:
        recids (list): List of record IDs.
        labels (dict): Dictionary mapping record IDs to labels.
        subsampling_factor (int): The number of subsamples to draw per record.
        window_size (int): The size of each subsample window.
        data_folder (str): Directory containing the data files.
        num_leads (int): Number of signal leads.
        sampling_rate (int): Target sampling rate to which signals will be resampled if necessary.
        num_cls (int): Number of classes for one-hot encoding.
        is_ecg (bool): Whether the data is ECG data (determines label mapping).

    Returns:
        tuple: A tuple containing the data array (X_split), labels array (Y_split), and a list of generated record IDs.
    """
    X_split = np.zeros((len(recids) * subsampling_factor, window_size, num_leads))
    Y_split = np.zeros((len(recids) * subsampling_factor, num_cls))
    record_ids = []

    for i, recid in enumerate(recids):
        ecg, fields = load_signals('{}/{}'.format(data_folder, recid))
        if is_ecg:
            c = labels[f"{recid}.npy"]
        else:
            c = labels[recid.split("_")[0]]

        # Normalize sampling rate
        if fields['fs'] != sampling_rate:
            ecg = normalize_sampling_rate_array_based(ecg, fields['fs'], sampling_rate)
            logging.info("Resampling signal from %d Hz to %d Hz", fields['fs'], sampling_rate)

        # Draw subsamples, normalize length
        X_split[i * subsampling_factor: i * subsampling_factor + subsampling_factor] = subsample_ecg(ecg, window_size,
                                                                                                     subsampling_factor)

        # One-hot encoding
        Y_split[i * subsampling_factor: i * subsampling_factor + subsampling_factor, c] = 1

        # Collect record-IDs
        if is_ecg:
            record_ids += ['{}_{}'.format(recid, j) for j in range(subsampling_factor)]
        else:
            record_ids.append(recid)

    return X_split, Y_split, record_ids
