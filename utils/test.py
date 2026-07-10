"""
Test utility module for evaluating models and extracting explainability maps.
"""
import os
import numpy as np
from signxai.methods.wrappers import calculate_relevancemap

from utils.loader import load_ecg_labels, load_rpw_labels
from utils.data import load_and_preprocess_signal
from utils.model import load_models_from_paths
from utils.explainability import normalize_ecg_relevancemap
import config


def run_rpw_model(rpw_model_path, rpw_files_path, rpw_files_list, plot_explanations, plot_dir):
    """
    Run the RPW model on a list of files and optionally plot XAI explanations.

    Args:
        rpw_model_path (str): Path to the RPW model directory.
        rpw_files_path (str): Path to the directory containing input signals.
        rpw_files_list (list): List of file names to process.
        plot_explanations (bool): Whether to calculate and save relevance maps.
        plot_dir (str): Directory where plots and relevance maps should be saved.

    Returns:
        tuple: (rpw_result_dict, rpw_hist) containing predictions and normalized relevance maps.
    """

    if plot_explanations:
        os.makedirs(plot_dir, exist_ok=True)

    rpw_model, rpw_model_wo_softmax = load_models_from_paths(
        modelpath='{}/model.json'.format(rpw_model_path),
        weightspath='{}/weights.h5'.format(rpw_model_path)
    )

    rpw_last_convolution_layer = None
    for layer in reversed(rpw_model_wo_softmax.layers):
        if 'conv1d' in str(type(layer)).lower():
            rpw_last_convolution_layer = layer.name
            break

    rpw_labels = load_rpw_labels()
    rpw_result_dict = {}
    rpw_hist = []

    for rpw_file in sorted(rpw_files_list):
        rpw_signal = load_and_preprocess_signal(record_id=rpw_file, subsampling_window_size=1000, subsample_start=0,
                                                src_dir=rpw_files_path)
        pred = rpw_model.predict(np.expand_dims(rpw_signal, axis=0))[0]
        label = rpw_labels[rpw_file.split("_")[0]]
        if label == 1:
            rpw_result_dict[rpw_file] = [pred, [0, 1]]
        else:
            rpw_result_dict[rpw_file] = [pred, [1, 0]]

        if plot_explanations:
            pred_class = np.argmax(pred)

            if label != 1 or pred_class != 1:
                continue

            # Calculate relevancemap based on SIGN-XAI package
            R = calculate_relevancemap(config.xai_method, rpw_signal, rpw_model_wo_softmax,
                                       last_conv_layer_name=rpw_last_convolution_layer)
            # Use only positives
            R[R < 0] = 0
            # Normalize relevance map
            Rn = normalize_ecg_relevancemap(R, local=False)
            rpw_hist.append(Rn.flatten())

            np.save('{}/{}_{}.npy'.format(plot_dir, rpw_file, config.xai_method), Rn)
            np.save('{}/{}.npy'.format(plot_dir, rpw_file), rpw_signal)

    return rpw_result_dict, rpw_hist


def run_ecg_model(ecg_model_path, ecg_files_path, ecg_files_list, plot_explanations, plot_dir):
    """
    Run the ECG model on a list of files with subsampling and optionally plot XAI explanations.

    Args:
        ecg_model_path (str): Path to the ECG model directory.
        ecg_files_path (str): Path to the directory containing input signals.
        ecg_files_list (list): List of file names to process.
        plot_explanations (bool): Whether to calculate and save relevance maps.
        plot_dir (str): Directory where plots and relevance maps should be saved.

    Returns:
        tuple: (ecg_results_dict, ecg_hist) containing predictions and normalized relevance maps.
    """

    if plot_explanations:
        os.makedirs(plot_dir, exist_ok=True)

    ecg_model, ecg_model_wo_softmax = load_models_from_paths(
        modelpath='{}/model.json'.format(ecg_model_path),
        weightspath='{}/weights.h5'.format(ecg_model_path)
    )

    ecg_last_convolution_layer = None
    for layer in reversed(ecg_model_wo_softmax.layers):
        if 'conv1d' in str(type(layer)).lower():
            ecg_last_convolution_layer = layer.name
            break

    ecg_labels = load_ecg_labels()
    ecg_results_dict = {}
    ecg_hist = []

    for ecg_file in sorted(ecg_files_list):
        for i, start in enumerate(range(0, int(2500 - 2000) + 1,
                       int((2500 - 2000) / 4))):

            subsample = load_and_preprocess_signal(record_id=ecg_file,
                                                   subsampling_window_size=2000, subsample_start=start,
                                                   src_dir=ecg_files_path)
            pred = ecg_model.predict(np.expand_dims(subsample, axis=0))[0]
            label = ecg_labels[f"{ecg_file}.npy"]
            if label == 1:
                ecg_results_dict[f"{ecg_file}_{i}"] = [pred, [0, 1]]
            else:
                ecg_results_dict[f"{ecg_file}_{i}"] = [pred, [1, 0]]

            if plot_explanations:
                pred_class = np.argmax(pred)

                if label != 1 or pred_class != 1:
                    continue

                # Calculate relevancemap based on SIGN-XAI package
                R = calculate_relevancemap(config.xai_method, subsample, ecg_model_wo_softmax,
                                           last_conv_layer_name=ecg_last_convolution_layer)
                # Use only positives
                R[R < 0] = 0
                # Normalize relevance map
                Rn = normalize_ecg_relevancemap(R, local=False)
                ecg_hist.append(Rn.flatten())

                np.save('{}/{}_{}_{}.npy'.format(plot_dir, ecg_file, i, config.xai_method), Rn)
                np.save('{}/{}_{}.npy'.format(plot_dir, ecg_file, i), subsample)

    return ecg_results_dict, ecg_hist
