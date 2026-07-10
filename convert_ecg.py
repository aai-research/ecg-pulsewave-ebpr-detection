"""Convert 12-lead ECG numpy arrays to WFDB format and split into train/test sets."""

from utils.gpu import set_visible_gpu

set_visible_gpu(0)

import config
from utils.training import enable_reproducibility

enable_reproducibility(config.seed)

import os
import numpy as np
import wfdb
import pandas as pd
import logging

from utils.data import perform_shape_switch


def run(input_path, labels, output_path_trainval, output_path_test):
    """Load, convert, and save ECG signals into trainval and test splits."""
    logging.info("Starting ECG conversion process...")
    # Load and convert the signal data.
    lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    sampling_frequency = 500
    units = 'mV'

    # Define the parameters for the WFDB files.
    gain = 1000
    baseline = 0
    num_bits = 16
    fmt = str(num_bits)

    # Create output dirs
    os.makedirs(output_path_trainval, exist_ok=True)
    os.makedirs(output_path_test, exist_ok=True)

    labels = pd.read_csv(labels)
    trainval_files = labels.loc[labels["Split"] == "trainval", "ecg_file"].values
    test_files = labels.loc[labels["Split"] == "test", "ecg_file"].values

    for f in os.listdir(input_path):
        logging.info("Processing file: %s", f)
        if f in trainval_files:
            output_path = output_path_trainval
        elif f in test_files:
            output_path = output_path_test
        else:
            continue

        physical_signals = np.load('{}/{}'.format(input_path, f))

        # Adjust shape
        if np.shape(physical_signals)[0] < np.shape(physical_signals)[1]:
            physical_signals = perform_shape_switch(physical_signals)

        # Perform basic error checking on the signal
        num_samples, num_leads = np.shape(physical_signals)
        assert (num_leads == 12)

        # Convert the signal to digital units; saturate the signal and represent NaNs as the lowest representable integer.
        digital_signals = gain * physical_signals
        digital_signals = np.round(digital_signals)
        digital_signals = np.clip(digital_signals, -2 ** (num_bits - 1) + 1, 2 ** (num_bits - 1) - 1)
        digital_signals[~np.isfinite(digital_signals)] = -2 ** (num_bits - 1)
        digital_signals = np.asarray(digital_signals, dtype=np.int32)  # We need to promote from 16-bit integers due to an error in the Python WFDB library.

        # Save the signal.
        wfdb.wrsamp(
            f.replace('.npy', ''), fs=sampling_frequency, units=[units] * num_leads,
            sig_name=lead_names, d_signal=digital_signals, fmt=[fmt] * num_leads, adc_gain=[gain] * num_leads,
            baseline=[baseline] * num_leads, write_dir=output_path
        )
    logging.info("ECG conversion process completed.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    run(f"{config.directory}/data/ecg_numpy", f"{config.directory}/data/labels.csv",
        f"{config.directory}/data/train/ecg",
        f"{config.directory}/data/test/ecg")
