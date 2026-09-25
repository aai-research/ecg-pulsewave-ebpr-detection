"""Quantify ECG explanations (LRP) per lead and per length-normalized waveform segment and plot radar charts."""

import config

import os
import glob
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')  # plot_ecg calls plt.show() when save_to is None, which must not block
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import butter, filtfilt
from matplotlib import pyplot as plt

from utils.radar import radar_factory


FS = 500  # Sampling rate (Hz)
WINDOW = 2000  # Subsample length used by run_ecg_model
STRIDE = 125  # Subsample stride used by run_ecg_model (5 windows over 2500 samples)
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
SEGMENTS = ['P', 'PQ', 'Q', 'R', 'S', 'ST', 'T', 'TP']
FIDUCIALS = ['p_on', 'p_off', 'qrs_on', 'r_on', 'r_off', 'qrs_off', 't_on', 't_off']  # ms relative to R-peak
REFERENCE_LEAD = 'aVF'  # Lead whose Q/R/S split defines the segmentation that is applied to all leads
SEGMENT_COLORS = {'P': '#9ecae1', 'PQ': '#fee391', 'Q': '#fdae6b', 'R': '#e6550d', 'S': '#fd8d3c',
                  'ST': '#c7e9c0', 'T': '#74c476', 'TP': '#dadaeb'}


def ms(samples):
    return samples * 1000 / FS


def smp(milliseconds):
    return int(round(milliseconds * FS / 1000))


def bandpass(x, low=0.5, high=40):
    b, a = butter(2, [low / (FS / 2), high / (FS / 2)], btype='band')
    return filtfilt(b, a, x, axis=0)


def load_windows(base_dir):
    """
    Group all saved ECG windows and relevance maps by ECG recording (patient_session). Every fold model explains
    the same hold-out test set, so windows are keyed by (fold, window index).
    """
    ecgs = defaultdict(dict)
    for fold in range(1, config.num_folds + 1):
        for file in glob.glob(f"{base_dir}/xai_plots/ecg/{fold}/*.npy"):
            name = os.path.basename(file)
            if config.xai_method in name:
                continue
            ecg_id, i = name[:-4].rsplit('_', 1)
            ecgs[ecg_id][(fold, int(i))] = (np.load(file), np.load(file.replace('.npy', f'_{config.xai_method}.npy')))
    return dict(sorted(ecgs.items()))


def reconstruct(windows):
    """Stitch the overlapping windows back into one (partial) recording of shape (T, 12)."""
    starts = [i * STRIDE for _, i in windows]
    full = np.full((min(starts) + WINDOW + max(starts) - min(starts), 12), np.nan)
    for (_, i), (ecg, _) in windows.items():
        full[i * STRIDE:i * STRIDE + WINDOW] = ecg
    offset = min(starts)
    return full[offset:], offset


def detect_r_peaks(ecg_f):
    """R-peaks on the spatial magnitude of all leads, which is robust against single-lead morphology."""
    magnitude = np.sqrt(np.sum(ecg_f ** 2, axis=1))
    peaks = nk.ecg_peaks(magnitude, sampling_rate=FS, method='neurokit')[1]['ECG_R_Peaks']

    # Refine each peak to the local magnitude maximum and drop implausibly short RR intervals (T-waves)
    peaks = np.unique([p - smp(40) + np.argmax(magnitude[max(p - smp(40), 0):p + smp(40)]) for p in peaks])
    keep = [peaks[0]]
    for p in peaks[1:]:
        if p - keep[-1] < smp(300):
            if magnitude[p] > magnitude[keep[-1]]:
                keep[-1] = p
        else:
            keep.append(p)
    return np.array(keep)


def median_beat(ecg_f, r_peaks):
    rr = np.median(np.diff(r_peaks))
    pre, post = int(0.4 * rr), int(0.65 * rr)
    beats = [ecg_f[r - pre:r + post] for r in r_peaks if r - pre >= 0 and r + post <= len(ecg_f)]
    return np.median(beats, axis=0), pre, len(beats)


def crossing(x, start, stop, thresh):
    """First index walking from start towards stop (either direction) where x falls below thresh, else stop."""
    step = 1 if stop >= start else -1
    for i in range(start, stop, step):
        if x[i] < thresh:
            return i
    return stop


def positive_waves(x, amp):
    """
    Positive QRS deflections in temporal order as (on, off) index pairs. A deflection counts if it exceeds 5 % of
    the QRS amplitude, and two deflections are only separate waves if a negative deflection lies in between, so
    that a notched R stays one wave.
    """
    thr = 0.025 * amp
    lobes, i = [], 0
    while i < len(x):
        if x[i] > thr:
            j = i
            while j < len(x) and x[j] > thr:
                j += 1
            if x[i:j].max() >= 0.05 * amp:
                lobes.append([i, j])
            i = j
        else:
            i += 1
    waves = []
    for lobe in lobes:
        if waves and x[waves[-1][1]:lobe[0]].min() > -thr:
            waves[-1][1] = lobe[1]
        else:
            waves.append(lobe)
    return waves


def delineate(beat, pre):
    """
    Heuristic delineation of a 12-lead median beat. Returns one set of fiducials in ms relative to the R-peak,
    which is applied to all leads.

    P on/off, QRS on/off and T on/off are derived from all leads. The QRS is split by the reference lead: Q is
    the part before its first positive wave, R that wave, and S everything after it up to the QRS offset.
    """
    # Isoelectric level from the flattest 20 ms in the PQ segment
    slope = np.sum(np.abs(np.gradient(beat, axis=0)), axis=1)
    slope = np.convolve(slope, np.ones(smp(10)) / smp(10), mode='same')
    search = np.arange(pre - smp(120), pre - smp(20))
    flat = search[np.argmin([slope[i:i + smp(20)].sum() for i in search])]
    beat = beat - beat[flat:flat + smp(20)].mean(axis=0)
    magnitude = np.sqrt(np.sum(beat ** 2, axis=1))

    # QRS on/off where the spatial slope falls below 10 % of its maximum around the R-peak
    qrs_peak = pre - smp(60) + np.argmax(slope[pre - smp(60):pre + smp(60)])
    thresh = 0.1 * slope[qrs_peak]
    qrs_on = crossing(slope, qrs_peak, max(qrs_peak - smp(120), 0), thresh)
    qrs_off = crossing(slope, qrs_peak, min(qrs_peak + smp(150), len(beat) - 1), thresh)

    # T-wave: magnitude peak after the J-point, borders at 30 % (onset, above ST level) and 20 % (offset)
    t_search = magnitude[qrs_off + smp(60):]
    t_peak = qrs_off + smp(60) + np.argmax(t_search[:smp(450)])
    st_min = magnitude[qrs_off:t_peak].min()
    t_on = crossing(magnitude, t_peak, qrs_off, st_min + 0.3 * (magnitude[t_peak] - st_min))
    t_off = crossing(magnitude, t_peak, len(beat) - 1, 0.2 * magnitude[t_peak])

    # P-wave: magnitude peak before the QRS after removing the linear drift between TP and PQ level,
    # borders at 25 % and at most 100 ms from the peak
    p_lo = max(qrs_on - smp(300), 0)
    drift = np.linspace(beat[:smp(20)].mean(axis=0), beat[flat:flat + smp(20)].mean(axis=0), flat + smp(10))
    p_mag = np.sqrt(np.sum((beat[:flat + smp(10)] - drift) ** 2, axis=1))
    p_peak = p_lo + np.argmax(p_mag[p_lo:min(qrs_on - smp(30), len(p_mag))])
    p_thresh = 0.25 * p_mag[p_peak]
    p_on = crossing(p_mag, p_peak, max(p_peak - smp(100), 0), p_thresh)
    p_off = crossing(p_mag, p_peak, min(p_peak + smp(100), len(p_mag) - 1, qrs_on), p_thresh)

    x = beat[qrs_on:qrs_off, LEADS.index(REFERENCE_LEAD)]
    waves = positive_waves(x, np.max(np.abs(x)))
    # QS complex (no positive wave): the whole QRS counts as Q
    r_on, r_off = (qrs_on + waves[0][0], qrs_on + waves[0][1]) if waves else (qrs_off, qrs_off)

    f = {'p_on': p_on, 'p_off': p_off, 'qrs_on': qrs_on, 'r_on': r_on, 'r_off': r_off, 'qrs_off': qrs_off,
         't_on': t_on, 't_off': t_off}
    return {k: int(round(ms(v - pre))) for k, v in f.items()}


def build_mask(f, r_peaks, length):
    """Label every sample of a (length, 12) signal with a segment index (-1 = unassigned), identical for all leads."""
    f = {k: smp(f[k]) for k in FIDUCIALS}
    mask = np.full(length, -1)
    bounds = [('P', 'p_on', 'p_off'), ('PQ', 'p_off', 'qrs_on'), ('Q', 'qrs_on', 'r_on'), ('R', 'r_on', 'r_off'),
              ('S', 'r_off', 'qrs_off'), ('ST', 'qrs_off', 't_on'), ('T', 't_on', 't_off')]
    for k, r in enumerate(r_peaks):
        for seg, a, b in bounds:
            lo, hi = max(r + f[a], 0), min(r + f[b], length)
            if hi > lo:
                mask[lo:hi] = SEGMENTS.index(seg)
        # TP runs from the end of this T-wave up to the next P-wave (only where nothing else is assigned)
        if k + 1 < len(r_peaks):
            lo, hi = max(r + f['t_off'], 0), min(r_peaks[k + 1] + f['p_on'], length)
            if hi > lo:
                tp = mask[lo:hi]
                tp[tp == -1] = SEGMENTS.index('TP')
    return np.tile(mask[:, None], (1, 12))


def enrichment(R, mask):
    """
    Relevance-to-duration ratio (RDR) per segment: RDR_c = (relevance share of c) / (duration share of c).
    RDR_c = 1 corresponds to chance, i.e. the relevance a segment earns from its duration alone.
    Returns the pooled enrichment (over all leads) and the enrichment within each lead.
    """
    def _e(r, m):
        valid = m >= 0
        if r[valid].sum() == 0:
            return np.full(len(SEGMENTS), np.nan)
        rel = np.array([r[m == c].sum() for c in range(len(SEGMENTS))]) / r[valid].sum()
        dur = np.array([(m == c).sum() for c in range(len(SEGMENTS))]) / valid.sum()
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(dur > 0, rel / dur, np.nan)

    pooled = _e(R.ravel(), mask.ravel())
    per_lead = np.array([_e(R[:, lead], mask[:, lead]) for lead in range(12)])
    return pooled, per_lead


def plot_qc(ecg_id, beat, pre, f, save_to):
    """Median beat per lead with the segment mask, to verify (and adjust) the fiducials."""
    t = ms(np.arange(len(beat)) - pre)
    mask = build_mask(f, [pre], len(beat))[:, 0]
    fig, axs = plt.subplots(3, 4, figsize=(14, 7), sharex=True)
    for lead in range(12):
        ax = axs[lead % 3, lead // 3]
        for c, seg in enumerate(SEGMENTS):
            for i in np.where(mask == c)[0]:
                ax.axvspan(t[i], t[i] + ms(1), color=SEGMENT_COLORS[seg], lw=0)
        ax.plot(t, beat[:, lead], 'k', lw=1)
        ax.set_title(LEADS[lead], fontsize=9, loc='left')
        ax.tick_params(labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEGMENT_COLORS[s]) for s in SEGMENTS]
    fig.legend(handles, SEGMENTS, ncol=len(SEGMENTS), loc='upper center', frameon=False)
    fig.suptitle(f"{ecg_id} (Q/R/S from {REFERENCE_LEAD})", y=0.02, fontsize=9)
    fig.supxlabel('Time relative to R-peak (ms)', fontsize=9)
    plt.savefig(save_to, bbox_inches='tight')
    plt.close()


def radar(ax, theta, values, labels, chance, rmax, rticks=(), rticklabels=None, highlight=()):
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.plot(theta, np.full(len(theta), chance), color='grey', lw=0.6, ls='--')
    ax.plot(theta, values, color='r', lw=1)
    ax.fill(theta, values, facecolor='r', alpha=0.25, label='_nolegend_')
    ax.set_ylim(0, rmax)
    ax.set_yticks(rticks)
    ax.set_yticklabels(rticklabels if rticklabels is not None else [f"{t:g}" for t in rticks], fontsize=4,
                       color='grey')
    ax.set_rlabel_position(360 / len(theta) / 2)
    ax.set_varlabels(labels)
    for t in ax.get_xticklabels():
        if t.get_text() in highlight:
            t.set_color('tomato')


def plot_global_segments(ecgs, fiducials, r_peaks, save_to, crop=(50, 400)):
    """
    Reproduce global_xai_ecg_{posthresh}.pdf of global_xai.py and add the segment mask for visual checking.

    The sample-level segment mask of every window (the one used for the enrichment) is one-hot encoded and passed
    through exactly the same code path as the ECG and its explanation (extract_beats with R-to-R resampling,
    calculate_mean_beat with its rotation). Each time point of the global beat is labelled with the segment that
    has the highest mean share there.
    """
    import utils.visualization as vis
    from utils.segmentation import extract_beats, calculate_mean_beat

    num_beats = 2
    channels = SEGMENTS + ['none']
    windows = [(ecg_id, i, ecg, Rn) for ecg_id in ecgs for (_, i), (ecg, Rn) in ecgs[ecg_id].items()]
    ECG_beats = np.ones((len(windows), num_beats, 12, 500)) * np.nan
    Rn_beats = np.ones((len(windows), num_beats, 12, 500)) * np.nan
    Seg_beats = np.ones((len(channels), len(windows), num_beats, 12, 500)) * np.nan

    for n, (ecg_id, i, ecg, Rn) in enumerate(windows):
        # ECG and explanation exactly as in global_xai.py
        ECG_beats[n] = extract_beats(ecg, segmentation_lead=1, sampling_rate=500, num_beats=num_beats)
        Rn_beats[n] = extract_beats(ecg, segmentation_lead=1, sampling_rate=500, extraction_array=Rn,
                                    num_beats=num_beats)

        # Segment mask of this window, one channel per segment, extracted with the same beats
        mask = build_mask(fiducials.loc[ecg_id], r_peaks[ecg_id] - i * STRIDE, WINDOW)
        for c in range(len(channels)):
            onehot = (mask == (c if c < len(SEGMENTS) else -1)).astype(float)
            Seg_beats[c, n] = extract_beats(ecg, segmentation_lead=1, sampling_rate=500, extraction_array=onehot,
                                            num_beats=num_beats)

    mean_ECG, mean_Rn, mean_Seg = [], [], []
    for lead in range(12):
        mean_ECG.append(calculate_mean_beat(ECG_beats, lead=lead, norm=True))
        lead_Rn = calculate_mean_beat(Rn_beats, lead=lead, norm=False)
        mean_Rn.append(lead_Rn / np.max(np.abs(np.ravel(lead_Rn))))
        mean_Seg.append([calculate_mean_beat(Seg_beats[c], lead=lead, norm=False) for c in range(len(channels))])
    mean_ECG, mean_Rn, mean_Seg = np.array(mean_ECG), np.array(mean_Rn), np.array(mean_Seg)
    mean_Rn[mean_Rn <= config.posthresh] = 0
    labels = mean_Seg.argmax(axis=1)  # (12, time), index len(SEGMENTS) = unassigned

    # Draw with the original plot_ecg (kept open instead of saved) and add the segment spans per lead
    close = vis.plt.close
    vis.plt.close = lambda *args, **kwargs: None
    try:
        vis.plot_ecg(ecg=mean_ECG[:, crop[0]:crop[1]], explanation=mean_Rn[:, crop[0]:crop[1]], bubble_size=5,
                     line_width=1, row_height=4, columns=4, title='', show_separate_line=True, style="fancy",
                     save_to=None, show_lead_name=True, display_factor=2, display_factor_font=1.25)
    finally:
        vis.plt.close = close
    fig, ax = plt.gcf(), plt.gca()
    ax.set_axisbelow(True)  # Grid at the bottom, segments above it, ECG and explanation on top

    secs, row_height = (crop[1] - crop[0]) / FS, 4
    for lead in range(12):
        c, i = lead // 3, lead % 3
        x_offset, y_offset = secs * c, -(row_height / 2) * i
        lab = labels[lead, crop[0]:crop[1]].copy()
        # Absorb 1-2 sample fragments at segment borders (resampling ringing) into the preceding segment
        starts = np.flatnonzero(np.diff(np.r_[-2, lab]) != 0)
        ends = np.r_[starts[1:], len(lab)]
        for s, e in zip(starts, ends):
            if e - s < 3 and s > 0:
                lab[s:e] = lab[s - 1]
        starts = np.flatnonzero(np.diff(np.r_[-2, lab]) != 0)
        ends = np.r_[starts[1:], len(lab)]
        for k, (s, e) in enumerate(zip(starts, ends)):
            if lab[s] >= len(SEGMENTS):
                continue
            seg = SEGMENTS[lab[s]]
            x0, x1 = x_offset + s / FS, x_offset + e / FS
            ax.fill_between([x0, x1], y_offset - 0.9, y_offset + 0.9, color=SEGMENT_COLORS[seg], alpha=0.35,
                            lw=0, zorder=0.8)
            if s > 0:
                # Thin dotted border between neighbouring segments, below ECG and explanation
                ax.plot([x0, x0], [y_offset - 0.9, y_offset + 0.9], color=(0.35, 0.35, 0.35), lw=0.3, ls=(0, (1, 1.5)),
                        zorder=0.9)
            ax.text((x0 + x1) / 2, y_offset + 0.9, seg, ha='center',
                    va='bottom', fontsize=3.5, zorder=4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEGMENT_COLORS[s], alpha=0.35) for s in SEGMENTS]
    fig.legend(handles, SEGMENTS, ncol=len(SEGMENTS), loc='lower right', bbox_to_anchor=(0.995, 0.005),
               frameon=False, fontsize=4, handlelength=1, handletextpad=0.4, columnspacing=0.8)
    plt.savefig(save_to, dpi=300)
    plt.close()


def write_table(per_ecg, save_to):
    """LaTeX tabular (booktabs) with lead share and segment enrichment as mean +- SD across ECGs."""
    def fmt(col, scale=1, digits=2, bold=False):
        m, sd = per_ecg[col].mean() * scale, per_ecg[col].std() * scale
        v = f"{m:.{digits}f} $\\pm$ {sd:.{digits}f}"
        return f"\\textbf{{{v}}}" if bold else v

    head = ' & '.join(f"\\textbf{{{s}}}" for s in SEGMENTS)
    lines = [f"\\begin{{tabular}}{{l c {'c' * len(SEGMENTS)}}}", "\\toprule",
             f"\\multirow{{2}}{{*}}{{\\textbf{{Lead}}}} & \\multirow{{2}}{{*}}{{\\makecell{{\\textbf{{Relevance}} \\\\ "
             f"\\textbf{{share (\\%)}}}}}} & \\multicolumn{{{len(SEGMENTS)}}}{{c}}{{\\textbf{{Relevance-to-duration "
             f"ratio (RDR)}}}} \\\\",
             f"\\cmidrule(lr){{3-{len(SEGMENTS) + 2}}}", f" & & {head} \\\\", "\\midrule"]
    for lead in LEADS:
        means = [per_ecg[f"E_{lead}_{s}"].mean() for s in SEGMENTS]
        cells = [fmt(f"E_{lead}_{s}", bold=(m == max(means))) for s, m in zip(SEGMENTS, means)]
        lines.append(f"{lead} & {fmt(f'share_{lead}', 100, 1)} & {' & '.join(cells)} \\\\")
    lines.append("\\midrule")
    lines.append(f"All leads & 100 & {' & '.join(fmt(f'E_{s}') for s in SEGMENTS)} \\\\")
    lines.append(f"Duration share (\\%) & -- & {' & '.join(fmt(f'dur_{s}', 100, 1) for s in SEGMENTS)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(save_to, 'w') as file:
        file.write('\n'.join(lines) + '\n')


def plot_radars(lead_share, seg_pooled, seg_per_lead, out_dir):
    """Radar charts sized for the manuscript figure (per-lead grid left, lead and pooled radars stacked right)."""
    plt.rcParams['font.size'] = 6

    # Relevance distribution over leads (chance = 1/12)
    theta = radar_factory(len(LEADS), frame='polygon')
    fig, ax = plt.subplots(figsize=(1.45, 1.45), subplot_kw=dict(projection='radar'))
    radar(ax, theta, lead_share, LEADS, chance=1 / 12, rmax=0.12, rticks=[0.05, 0.1], rticklabels=['5 %', '10 %'])
    plt.savefig(f"{out_dir}/radar_leads.pdf", bbox_inches='tight')
    plt.close()

    # Length-normalized relevance over segments, pooled over leads and per lead (chance = 1)
    theta = radar_factory(len(SEGMENTS), frame='polygon')
    fig, ax = plt.subplots(figsize=(1.45, 1.45), subplot_kw=dict(projection='radar'))
    radar(ax, theta, seg_pooled, SEGMENTS, chance=1, rmax=np.ceil(np.nanmax(seg_pooled)), rticks=[1, 2])
    plt.savefig(f"{out_dir}/radar_segments_pooled.pdf", bbox_inches='tight')
    plt.close()

    rmax = np.ceil(np.nanmax(seg_per_lead))
    fig, axs = plt.subplots(3, 4, figsize=(3.7, 3.2), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.85, hspace=0.55)
    for lead in range(12):
        ax = axs[lead % 3, lead // 3]
        radar(ax, theta, seg_per_lead[lead], SEGMENTS, chance=1, rmax=rmax, rticks=np.arange(1, rmax))
        ax.tick_params(axis='x', labelsize=5, pad=-3)
        ax.set_title(LEADS[lead], fontsize=6.5, fontweight='bold', x=-0.12, y=1.02)
    plt.savefig(f"{out_dir}/radar_segments_per_lead.pdf", bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--redelineate', action='store_true',
                        help='Overwrite fiducials.csv with the automatic delineation (discards manual edits)')
    args = parser.parse_args()

    base_dir = f"{config.directory}/results"
    out_dir = f"{base_dir}/segment_xai"
    os.makedirs(f"{out_dir}/qc", exist_ok=True)
    fid_path = f"{out_dir}/fiducials.csv"

    ecgs = load_windows(base_dir)

    # 1. Delineate one median beat per ECG. Fiducials are stored as an editable CSV and reused if present.
    beats, r_peaks, fids = {}, {}, []
    for ecg_id, windows in ecgs.items():
        full, offset = reconstruct(windows)
        full_f = bandpass(full)
        r = detect_r_peaks(full_f)
        r_peaks[ecg_id] = r + offset
        beat, pre, n = median_beat(full_f, r)
        beats[ecg_id] = (beat, pre)
        fids.append({'ecg_id': ecg_id, **delineate(beat, pre)})
        print(f"{ecg_id}: {len(windows)} windows ({len({f for f, _ in windows})} folds), {len(r)} R-peaks, median RR {ms(np.median(np.diff(r))):.0f} ms, "
              f"{n} beats in median")

    if os.path.exists(fid_path) and not args.redelineate:
        print(f"Using existing {fid_path}")
        fiducials = pd.read_csv(fid_path)
        if list(fiducials.columns) != ['ecg_id'] + FIDUCIALS:
            raise SystemExit(f"{fid_path} has an outdated format, rerun with --redelineate")
    else:
        fiducials = pd.DataFrame(fids)
        fiducials.to_csv(fid_path, index=False)
    fiducials = fiducials.set_index('ecg_id')

    for ecg_id, (beat, pre) in beats.items():
        plot_qc(ecg_id, beat, pre, fiducials.loc[ecg_id], f"{out_dir}/qc/{ecg_id}.pdf")

    # 2. Per window: lead share and segment enrichment, then averaged per ECG so that each recording counts once
    rows = []
    for ecg_id, windows in ecgs.items():
        for (fold, i), (ecg, R) in windows.items():
            mask = build_mask(fiducials.loc[ecg_id], r_peaks[ecg_id] - i * STRIDE, WINDOW)

            # Sanity check: uniform relevance must yield chance level (1) for every present segment
            u_pooled, _ = enrichment(np.ones_like(R), mask)
            assert np.allclose(u_pooled[~np.isnan(u_pooled)], 1)

            pooled, per_lead = enrichment(R, mask)
            row = {'ecg_id': ecg_id, 'fold': fold, 'window': i, 'unassigned': float((mask < 0).mean())}
            row.update({f"share_{l}": v for l, v in zip(LEADS, R.sum(axis=0) / R.sum())})
            row.update({f"E_{s}": v for s, v in zip(SEGMENTS, pooled)})
            row.update({f"dur_{s}": float((mask[:, 0] == k).sum() / (mask[:, 0] >= 0).sum())
                        for k, s in enumerate(SEGMENTS)})
            row.update({f"E_{l}_{s}": per_lead[j, k] for j, l in enumerate(LEADS) for k, s in enumerate(SEGMENTS)})
            rows.append(row)

    per_window = pd.DataFrame(rows)
    per_ecg = per_window.drop(columns=['fold', 'window']).groupby('ecg_id').mean()
    per_window.to_csv(f"{out_dir}/per_window.csv", index=False)
    per_ecg.to_csv(f"{out_dir}/per_ecg.csv")

    summary = per_ecg.agg(['mean', 'std', 'count']).T  # count = number of ECGs in which the segment exists
    summary.to_csv(f"{out_dir}/summary.csv")

    lead_share = per_ecg[[f"share_{l}" for l in LEADS]].mean().values
    seg_pooled = per_ecg[[f"E_{s}" for s in SEGMENTS]].mean().values
    seg_per_lead = np.array([per_ecg[[f"E_{l}_{s}" for s in SEGMENTS]].mean().values for l in LEADS])

    print(f"\n{len(per_window)} windows from {len(per_ecg)} ECGs, "
          f"{100 * per_window['unassigned'].mean():.1f} % of samples outside the segment mask")
    print("\nRelevance share per lead (chance = 0.083):")
    print(pd.Series(lead_share, index=LEADS).round(3).to_string())
    print(f"\nSegment enrichment (Q/R/S from {REFERENCE_LEAD}), pooled over leads (chance = 1):")
    print(pd.Series(seg_pooled, index=SEGMENTS).round(2).to_string())
    print("\nSegment enrichment per lead (chance = 1):")
    print(pd.DataFrame(seg_per_lead, index=LEADS, columns=SEGMENTS).round(2).to_string())

    plot_radars(lead_share, seg_pooled, seg_per_lead, out_dir)
    write_table(per_ecg, f"{out_dir}/segment_table.tex")
    plot_global_segments(ecgs, fiducials, r_peaks, f"{out_dir}/global_xai_ecg_{config.posthresh}_segments.pdf")
