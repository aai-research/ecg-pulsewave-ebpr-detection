"""Convert radial pulse wave numpy arrays to WFDB format and split into train/test sets."""

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


def run(input_path, labels, output_path_trainval, output_path_test, number_leads, min_max_normalize):
    """Load, convert, and save RPW signals into trainval and test splits."""
    logging.info("Starting RPW conversion process...")
    # Load and convert the signal data.
    sampling_frequency = 1000
    units = 'mmHg'

    # Define the parameters for the WFDB files.
    gain = 10
    baseline = 0
    num_bits = 16
    fmt = str(num_bits)

    # Create output dirs
    os.makedirs(output_path_trainval, exist_ok=True)
    os.makedirs(output_path_test, exist_ok=True)

    labels = pd.read_csv(labels)
    trainval_files = labels.loc[labels["Split"] == "trainval", "SessionID"].values
    test_files = labels.loc[labels["Split"] == "test", "SessionID"].values

    # Iterate over the input signal files.
    for f in os.listdir(input_path):
        logging.info("Processing file: %s", f)
        # Make sure only the first four RPWs are used
        if int(f.split(".")[0].split("_")[-1]) < 4:
            if f.split("_")[0] in trainval_files:
                output_path = output_path_trainval
            elif f.split("_")[0] in test_files:
                output_path = output_path_test
            else:
                continue
        else:
            continue

        physical_signals = np.load('{}/{}'.format(input_path, f), allow_pickle=True)

        lead_names = []
        if number_leads == 6:
            lead_names = ['Left1', 'Right1', 'Left2', 'Right2', 'Left3', 'Right3']
            physical_signals = [wave[1] for wave in physical_signals]
        elif number_leads == 1:
            lead_names = [f'RPW_{f.split(".")[0].split("_")[-1]}']

        # Adjust shape
        if np.shape(physical_signals)[0] < np.shape(physical_signals)[1]:
            if min_max_normalize:
                arr_min = physical_signals.min()
                arr_max = physical_signals.max()
                physical_signals = (physical_signals - arr_min) / (arr_max - arr_min)
            physical_signals = perform_shape_switch(physical_signals)

        # Perform basic error checking on the signal
        num_samples, num_leads = np.shape(physical_signals)
        while num_leads < number_leads:
            add = np.zeros((np.shape(physical_signals)[0], 1))
            physical_signals = np.append(physical_signals, add, axis=1)
            num_samples, num_leads = np.shape(physical_signals)
        if num_leads > number_leads:
            logging.warning("Skipping %s: exceeds expected number of leads", f)
            continue
        assert (num_leads == number_leads)

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
    logging.info("RPW conversion process completed.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    run(f"{config.directory}/data/rpw_numpy", f"{config.directory}/data/labels.csv",
        f"{config.directory}/data/train/rpw",
        f"{config.directory}/data/test/rpw", number_leads=1, min_max_normalize=False)
