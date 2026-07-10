"""Data manipulation and preprocessing utilities for ECG signals."""
import json
import wfdb
import numpy as np
import resampy
import scipy


def normalize_sampling_rate_array_based(ecg, sampling_rate_src, sampling_rate_trgt):
    """
    Resample an ECG array to a new target sampling rate.

    Args:
        ecg (np.ndarray): The original ECG array.
        sampling_rate_src (int or float): Original sampling rate.
        sampling_rate_trgt (int or float): Target sampling rate.

    Returns:
        np.ndarray: The resampled ECG array.
    """
    n_samples, n_leads = np.shape(ecg)

    factor = sampling_rate_trgt / sampling_rate_src

    ecg_new = np.zeros((int(n_samples * factor), n_leads))

    for i in range(n_leads):
        if factor != 0:
            ecg_new[..., i] = resampy.resample(ecg[..., i], sampling_rate_src, sampling_rate_trgt, filter='sinc_window', num_zeros=2, precision=1, window=scipy.signal.hann)

    return ecg_new


def extract_subsample_from_ecg_matrix_based(ecg, start, end):
    """
    Extract a temporal subsample from an ECG matrix.

    Args:
        ecg (np.ndarray): The ECG matrix.
        start (int): The starting index.
        end (int): The ending index.

    Returns:
        np.ndarray: The extracted subsample window.
    """
    return ecg[start:end]


def subsample_ecg(ecg, window_size, factor):
    """
    Draw multiple sliding-window subsamples from an ECG record.

    Args:
        ecg (np.ndarray): The ECG sequence.
        window_size (int): Size of each subsample window in timesteps.
        factor (int): Number of subsamples to extract.

    Returns:
        np.ndarray: An array containing the extracted subsamples.
    """
    length = len(ecg)

    if not length >= window_size:
        raise Exception('Record was shorter ({}) than the configured subsampling window size of {} timesteps. Aborting. Please adjust window size.'.format(length, window_size))

    if factor == 1:
        stride = 0
    else:
        stride = int((length - window_size) / (factor - 1))

    subsamples = np.zeros((factor, window_size, ecg.shape[1]))

    start = 0
    for i in range(factor):
        end = start + window_size

        if end > length:
            break

        subsamples[i,...] = extract_subsample_from_ecg_matrix_based(ecg, start, end)

        start = start + stride

    return subsamples


def perform_shape_switch(X):
    """
    Transpose the 2D array representation of a signal.

    Args:
        X (list or np.ndarray): The input matrix.

    Returns:
        np.ndarray: The transposed matrix.
    """
    X = np.asarray(X)

    dimx, dimy = X.shape

    output = np.zeros((dimy, dimx))

    for i in range(dimx):
        output[:, i] = X[i, :]

    return output

def save_dict_as_json(dct, path):
    """
    Save a dictionary as a JSON file.

    Args:
        dct (dict): The dictionary to save.
        path (str): File path for saving the JSON.
    """
    with open(path, 'w') as fp:
        json.dump(dct, fp)


def load_dict_from_json(path):
    """
    Load a JSON file into a dictionary.

    Args:
        path (str): The file path of the JSON.

    Returns:
        dict: The loaded dictionary.
    """
    string = load_string_from_file(path)
    dct = json.loads(string)

    return dct


def load_string_from_file(path):
    """
    Load the contents of a text file as a string.

    Args:
        path (str): The file path.

    Returns:
        str: The content of the file.
    """
    with open(path, 'r') as fp:
        string = fp.read()

    return string


def load_and_preprocess_signal(record_id, subsampling_window_size, subsample_start, src_dir):
    """
    Load and preprocess an ECG signal from WFDB format.

    Args:
        record_id (str): The record ID to load.
        subsampling_window_size (int): Size of the subsample window.
        subsample_start (int): Starting index for the subsample.
        src_dir (str): Directory containing the source WFDB records.

    Returns:
        np.ndarray: The preprocessed signal matrix.
    """
    raw_signal = load_raw_signal_wfdb('{}{}'.format(src_dir, record_id), verbose=False)
    signal = convert_lead_dict_to_matrix(raw_signal)
    signal = extract_subsample_from_signal(signal, subsample_start, subsample_start + subsampling_window_size)
    return perform_shape_switch(signal)


def load_raw_signal_wfdb(filepath, verbose=True):
    """
    Load a raw signal from a WFDB record.

    Args:
        filepath (str): Path to the WFDB record.
        verbose (bool): Whether to log loading progress.

    Returns:
        dict: A dictionary mapping lead names to their respective signal arrays.
    """
    if verbose:
        print('Loading {}'.format(filepath))

    signal, meta = wfdb.rdsamp(filepath)
    signal = perform_shape_switch(signal)
    signal = np.nan_to_num(signal)
    meta['sig_name'] = list(meta['sig_name'])

    leads = {}

    lead_id_count = {k: meta['sig_name'].count(k) for k in meta['sig_name']}
    lead_id_counter = {k: 1 for k in meta['sig_name']}

    for lead_id, lead_signal in zip(meta['sig_name'], signal):
        if lead_id_count[lead_id] > 1:
            l_id = '{}_{}'.format(lead_id, lead_id_counter[lead_id])
            lead_id_counter[lead_id] += 1
        else:
            l_id = lead_id
        leads[l_id] = lead_signal

    return leads


def convert_lead_dict_to_matrix(leads, shape_switch=False):
    """
    Convert a dictionary of leads to a numerical matrix.

    Args:
        leads (dict): Dictionary mapping lead names to arrays.
        shape_switch (bool): Whether to transpose the resulting matrix.

    Returns:
        np.ndarray: The stacked lead matrix.
    """
    collected = []

    for lead_id in leads:
        collected.append(leads[lead_id])

    collected = np.array(collected)

    if shape_switch:
        collected = perform_shape_switch(collected)

    return collected


def extract_subsample_from_signal(signal, start, end):
    """
    Extract a subsample from a continuous signal array.

    Args:
        signal (np.ndarray): The source signal array.
        start (int): Start index.
        end (int): End index.

    Returns:
        np.ndarray: The sliced signal array.
    """
    return signal[:, start:end]
