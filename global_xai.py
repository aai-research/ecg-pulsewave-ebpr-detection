"""Generate global explainability (LRP) visualizations for RPW and ECG models."""

from utils.gpu import set_visible_gpu

set_visible_gpu(0)

import config
from utils.training import enable_reproducibility

enable_reproducibility(config.seed)

import os
import numpy as np
from matplotlib import pyplot as plt

from utils.segmentation import extract_beats, calculate_mean_beat
from utils.visualization import plot_ecg, plot_lead_bubbles
from utils.data import perform_shape_switch


if __name__ == '__main__':
    base_dir = f"{config.directory}/results"

    num_beats = 1

    all_RPW = []
    all_Rn = []

    for subdir in range(1, config.num_folds + 1):

        # Iterate over all explainability (LRP) plots for RPW to calculate global mean
        for file in os.listdir(f"{base_dir}/xai_plots/rpw/{subdir}"):

            if file.endswith(".npy") and config.xai_method not in file:
                rpw = np.load(f"{base_dir}/xai_plots/rpw/{subdir}/{file}")
                rpw = perform_shape_switch(rpw)
                all_RPW.append(rpw)

                Rn = np.load(f"{base_dir}/xai_plots/rpw/{subdir}/{file.split('.npy')[0]}_{config.xai_method}.npy")
                Rn = perform_shape_switch(Rn)
                all_Rn.append(Rn)

    # Calculate global mean of the RPW and explanations
    mean_RPW = np.nanmean(all_RPW, axis=0)
    mean_Rn = np.nanmean(all_Rn, axis=0)

    # Normalize the explanations by the maximum absolute value
    mean_Rn = np.array(mean_Rn / np.max(np.abs(np.ravel(mean_Rn))))

    # Thresholding explanations to highlight only values above the defined threshold
    mean_Rn[mean_Rn <= config.posthresh] = 0
    mean_Rn[mean_Rn > config.posthresh] = mean_Rn[mean_Rn > config.posthresh]

    fig, ax = plt.subplots()
    plot_lead_bubbles(
        x=np.arange(0, len(mean_RPW[0])),
        y=mean_RPW[0],
        z=mean_Rn[0],
        ax=ax,
        linewidth=1.5,
        bubble_size=75,
        color_line=(0, 0, 0),
        clim_min=-1,
        clim_max=1
    )
    plt.xlim(0, len(mean_RPW[0]))
    plt.ylim(65, 135)
    plt.xlabel("Time (ms)", fontsize=16)
    plt.ylabel("Pressure (mmHg)", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(f"{base_dir}/global_xai_rpw_{config.posthresh}.pdf")
    plt.close()


    num_beats = 2

    num_ecg_files = sum(
        1 for subdir in range(1, config.num_folds + 1)
        for file in os.listdir(f"{base_dir}/xai_plots/ecg/{subdir}")
        if file.endswith(".npy") and config.xai_method not in file
    )
    ECG_beats = np.ones((num_ecg_files, num_beats, 12, 500)) * np.nan
    Rn_beats = np.ones((num_ecg_files, num_beats, 12, 500)) * np.nan

    mean_ECG = []
    mean_Rn = []

    n = 0

    for subdir in range(1, config.num_folds + 1):

        # Extract individual beats and explanations from ECG records
        for file in os.listdir(f"{base_dir}/xai_plots/ecg/{subdir}"):

            if file.endswith(".npy") and config.xai_method not in file:
                ecg = np.load(f"{base_dir}/xai_plots/ecg/{subdir}/{file}")
                ECG_beats[n] = extract_beats(ecg, segmentation_lead=1, sampling_rate=500, num_beats=num_beats)

                Rn = np.load(f"{base_dir}/xai_plots/ecg/{subdir}/{file.split('.npy')[0]}_{config.xai_method}.npy")
                Rn_beats[n] = extract_beats(ecg, segmentation_lead=1, sampling_rate=500, extraction_array=Rn,
                                            num_beats=num_beats)

                n += 1

    for i in range(12):
        # Calculate the mean beat for each lead over all ECG files
        mean_lead_ECG = calculate_mean_beat(ECG_beats, lead=i, norm=True)
        mean_lead_Rn = calculate_mean_beat(Rn_beats, lead=i, norm=False)
        
        # Normalize the mean explanations for the lead
        mean_lead_Rn = mean_lead_Rn / np.max(np.abs(np.ravel(mean_lead_Rn)))

        mean_ECG.append(mean_lead_ECG)
        mean_Rn.append(mean_lead_Rn)

    mean_ECG = np.array(mean_ECG)
    mean_Rn = np.array(mean_Rn)

    # Thresholding explanations to highlight only values above the defined threshold
    mean_Rn[mean_Rn <= config.posthresh] = 0
    mean_Rn[mean_Rn > config.posthresh] = mean_Rn[mean_Rn > config.posthresh]

    # Generate and save the global explanation visualization for ECG
    plot_ecg(ecg=mean_ECG[:, 50:400],
             explanation=mean_Rn[:, 50:400],
             bubble_size=5,
             line_width=1,
             row_height=4,
             columns=4,
             title='',
             show_separate_line=True,
             style="fancy",
             save_to=f"{base_dir}/global_xai_ecg_{config.posthresh}.pdf",
             show_lead_name=True,
             display_factor=2,
             display_factor_font=1.25)
