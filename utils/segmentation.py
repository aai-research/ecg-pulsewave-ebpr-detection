"""
Segmentation utility module for signal analysis and peak detection.
"""
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import resampy

from utils.data import perform_shape_switch


def calculate_mean_beat(beats, lead, cutoff=2, norm=True):
    """
    Calculate the average beat morphology across multiple beats for a specific lead.

    Args:
        beats (np.ndarray): An array containing segmented beats.
        lead (int): The index of the lead to compute the mean beat for.
        cutoff (int, optional): The number of edge samples to truncate. Defaults to 2.
        norm (bool, optional): Whether to normalize the beat amplitude. Defaults to True.

    Returns:
        np.ndarray: The computed mean beat.
    """
    lead_values = []
    for v in beats:
        for x in v:
            beat_tmp = np.hstack([x[lead][300:-cutoff], x[lead][cutoff:300]])

            if norm:
                beat_tmp = beat_tmp / np.max(np.abs(np.ravel(beat_tmp)))

            lead_values.append(beat_tmp)

    return np.nanmean(lead_values, axis=0)


def extract_beats(ecg, segmentation_lead=0, sampling_rate=500, extraction_array=None, num_beats=2):
    """
    Extract individual beats from an ECG signal based on R-peak positions.

    Args:
        ecg (np.ndarray): The input ECG signal array.
        segmentation_lead (int, optional): The index of the lead to use for peak detection. Defaults to 0.
        sampling_rate (int, optional): The sampling rate of the signal. Defaults to 500.
        extraction_array (np.ndarray, optional): Array to extract beats from, if different from ecg. Defaults to None.
        num_beats (int, optional): The number of beats to extract. Defaults to 2.

    Returns:
        np.ndarray: Array containing the resampled and extracted beats.
    """
    if extraction_array is None:
        extraction_array = ecg

    if extraction_array.shape[0] > extraction_array.shape[1]:
        extraction_array = perform_shape_switch(extraction_array)

    if ecg.shape[0] > ecg.shape[1]:
        ecg = perform_shape_switch(ecg)

    beats = np.ones((num_beats, ecg.shape[0], sampling_rate)) * np.nan

    try:
        r_peaks = derive_R_peak_positions(ecg, sampling_rate, segmentation_lead)

        for i in range(len(r_peaks)-1):
            if i == num_beats:
                break
            beat = extraction_array[..., r_peaks[i]:r_peaks[i+1]]
            factor = sampling_rate / beat.shape[1]
            beat_norm = np.zeros((beat.shape[0], sampling_rate))
            tmp = resampy.resample(beat, sampling_rate, sampling_rate * factor)
            for j, v in enumerate(tmp):
                beat_norm[j, :len(v)] = v
            beats[i] = beat_norm
    except IndexError:
        pass

    return beats


def derive_R_peak_positions(ecg, sampling_rate, segmentation_lead, debug=False):
    """
    Find R-peak positions in an ECG signal using NeuroKit2.

    Args:
        ecg (np.ndarray): The input ECG signal array.
        sampling_rate (int): The sampling rate of the signal.
        segmentation_lead (int): The index of the lead to use.
        debug (bool, optional): If True, plots the peaks for visual inspection. Defaults to False.

    Returns:
        np.ndarray: Indices of the detected R-peaks.
    """
    shp = np.shape(ecg)
    # Extract first lead
    if shp[0] < shp[1]:
        ecg_lead_I = ecg[segmentation_lead]
    else:
        ecg_lead_I = ecg[..., 0]

    # Calculate R peaks
    R_peaks_I = nk.ecg_peaks(np.abs(ecg_lead_I), sampling_rate=sampling_rate, method='neurokit')[1]['ECG_R_Peaks']

    if debug:
        plt.rcParams['figure.figsize'] = [20, 2]
        nk.events_plot([R_peaks_I], ecg_lead_I)
        plt.tight_layout()
        plt.show()

    return R_peaks_I
