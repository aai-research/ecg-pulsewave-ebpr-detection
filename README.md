# Resting electrocardiographic and pulse-wave deep learning identifies exaggerated exercise blood pressure response in elite athletes

This repository contains the official code for the paper **"Resting electrocardiographic and pulse-wave deep learning identifies exaggerated exercise blood pressure response in elite athletes"**, which is currently under review at *npj Digital Medicine* (submitted on July 10, 2026).

---

## 📑 Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Availability](#data-availability)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Author Contributions](#author-contributions)
- [Competing Interests](#competing-interests)

<a id="overview"></a>
## 📖 Overview
**Background:**
An exaggerated systolic blood pressure response to exercise (eBPR) is associated with future arterial hypertension and adverse cardiovascular outcomes, yet it remains an exercise-derived phenotype with heterogeneous definitions. We investigated whether resting cardiovascular biosignals can identify eBPR, defined by a workload-indexed systolic blood pressure-to-metabolic equivalent slope (SBP/MET slope) >6.2 mmHg/MET.

**Methods:**
In this retrospective study, 197 predominantly male professional athletes contributed 334 preseason screening examinations comprising resting 12-lead electrocardiograms (ECG), radial pulse waves (RPW), and cycle ergometry. Convolutional neural networks were trained with group-stratified five-fold cross-validation and evaluated on a hold-out test set. Explainable AI (LRP) was used to identify contributing signal regions.

**Results:**
The combined RPW–ECG model, via late fusion, achieved an area under the receiver operating characteristic curve (ROC AUC) of 0.79 (95% CI 0.66–0.91), slightly exceeding single-modality models (ECG: 0.77 [0.63–0.89]; RPW: 0.66 [0.51–0.81]). A clinical reference model based on routinely recorded resting variables reached an AUC of 0.44 (0.23–0.63). As a proof of concept, this shows that resting biosignals could complement, though not replace, exercise testing, pending external validation and prospective outcome studies.

<a id="repository-structure"></a>
## 🗂 Repository Structure
```text
ecg-pulsewave-ebpr-detection/
├── config.py                  # Global configuration for the eBPR detection pipeline
├── create_labels.ipynb        # Label creation and cohort definition (Jupyter Notebook)
├── convert_ecg.py             # Convert ECG numpy arrays to WFDB format and split train/test
├── convert_rpw.py             # Convert RPW numpy arrays to WFDB format and split train/test
├── train_rpw_model.py         # Train the 1D-CNN model for radial pulse wave (RPW) data (5-fold CV)
├── train_ecg_model.py         # Train the 1D-CNN model for 12-lead ECG data (5-fold CV)
├── evaluate_results.py        # Evaluate models (incl. clinical reference model), ensembles, bootstrap, tables and figures
├── global_xai.py              # Generate global explainability (LRP) visualizations
├── segment_xai.py             # Quantify ECG explanations per lead and per waveform segment (radar charts, table)
├── utils/                     # Utility modules
│   ├── __init__.py            # Utility package for the eBPR detection pipeline
│   ├── bootstrap.py           # Paired athlete-level bootstrap (CIs, AUC differences, ΔAUC table)
│   ├── callbacks.py           # Custom Keras callbacks for per-epoch validation metric calculation
│   ├── clinical.py            # Clinical reference model (logistic regression on resting variables)
│   ├── data.py                # Data I/O, signal resampling, subsampling, and shape manipulation utilities
│   ├── evaluation.py          # Result filtering, ROC plotting, and LaTeX table generation
│   ├── explainability.py      # Relevance map normalization for LRP
│   ├── gpu.py                 # GPU device selection
│   ├── loader.py              # Data loading for training/inference
│   ├── logger.py              # Logging configuration
│   ├── metrics.py             # Classification metrics (ROC, AUC, F1, MCC, etc.)
│   ├── model.py               # Model loading utilities for Keras JSON and weights
│   ├── radar.py               # Radar chart projection (adapted from the matplotlib gallery)
│   ├── roc_figure.py          # Test-set ROC figure of the fold ensembles
│   ├── segmentation.py        # ECG beat segmentation using R-peak detection
│   ├── splitting.py           # Group-stratified cross-validation splitting
│   ├── test.py                # Model inference with optional XAI map generation
│   ├── training.py            # Training utilities (reproducibility, optimizer, class weights)
│   └── visualization.py       # ECG and RPW signal visualization with LRP relevance overlay
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md                  # This file
```

<a id="installation"></a>
## ⚙️ Installation
The pipeline requires Python (3.10). Follow the steps below to set up the environment and install dependencies.

```bash
# Clone the repository
git clone https://github.com/aai-research/ecg-pulsewave-ebpr-detection.git
cd ecg-pulsewave-ebpr-detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note (macOS):** TensorFlow 2.8 provides no wheels for Apple Silicon with Python 3.10, so `requirements.txt` installs `tensorflow-macos` 2.9 on macOS instead (selected automatically via environment markers). On macOS, `pyodbc` additionally requires unixODBC (`brew install unixodbc`) to be importable, but it is not needed to run the pipeline.

<a id="usage"></a>
## 🚀 Usage

### 1. Configuration
Check and update `config.py` as necessary before running the pipeline. Key parameters include:
- `seed`: Random seed for reproducibility.
- `directory`: The root directory for the current experiment's data and outputs.
- `num_folds`: Number of cross-validation folds (e.g., 5).
- `n_bootstrap`: Number of bootstrap iterations for confidence intervals and AUC differences (default: 10,000).
- `ebpr_threshold`: Outcome definition, eBPR if SBP/MET slope > threshold (default: 6.2 mmHg/MET).
- `classification_threshold`: Probability threshold for sensitivity, specificity, F1 and the selection of true-positive predictions for XAI (default: 0.5).
- `ecg_record_length`, `ecg_window_size`, `ecg_num_windows`: ECG windowing, i.e. recording length l, window length w and number n of overlapping windows per recording (default: 2500, 2000, 5 samples at 500 Hz).
- `xai_method`, `posthresh`: LRP variant and relevance-visualization threshold of the global explanations (default: 0.75).
- Data files referenced (e.g., `matching.csv`, `data.xlsx`, etc.).
- `clinical_features`, `clinical_missing`, `clinical_class_weight`, `clinical_complete_cases_only`: Settings of the clinical reference model (predictors, handling of missing values, class weighting, evaluation set).
- `results_name`: Name of the results folder inside `directory`. If `None`, it is derived from the clinical settings (e.g., `results_impute_balanced_all_7vars`), so runs with different settings do not overwrite each other.
- `highlight_revision`, `highlight_color`: Optionally colour selected content of the LaTeX tables (`\textcolor`, requires `\usepackage{xcolor}`).

### 2. Data Preparation
Prepare the data directory structure under `{config.directory}/data/`. To run the pipeline from scratch, you must provide the following raw files:
- **`ecg_numpy/`**: Directory containing `.npy` files for 12-lead ECG recordings. Expected shape: `(12, 2500)` (500Hz for 5 seconds).
- **`rpw_numpy/`**: Directory containing `.npy` files for Radial Pulse Wave recordings, one file per exported cardiac cycle (`<SessionID>_<index>.npy`, the first four cycles per examination are used). Expected shape: `(1, 1000)`. The exported cycles have different lengths depending on the cardiac period and were time-normalized to 1,000 samples by linear interpolation.
- **`data.xlsx`**: An Excel file containing manual athlete data (e.g., DateOfBirth, Surname, Firstnames, SBP/MET slope).
- **`matching.csv`**: A CSV file used to match and de-duplicate patient records (PatID, SN, FN, DoB).
- **`rpw_data.csv`**: A CSV file containing RPW recording metadata (DateTime, SessionID, etc.).
- **`ecg_dates.csv`**: A CSV file containing ECG recording dates (PatID, Date, Filename).

These files are used by the `create_labels.ipynb` notebook to automatically generate the final `labels.csv` used in training.

### 3. Pipeline Execution
Run the pipeline scripts in the following order:

**Step 3.1. Generate Labels and Cohort Definition**
Run the Jupyter Notebook `create_labels.ipynb` to process raw clinical data and generate the `labels.csv` file.

**Step 3.2. Data Conversion (NumPy to WFDB)**
Convert the `.npy` files to WFDB format, split automatically into training/validation and testing sets based on `labels.csv`:
```bash
python convert_ecg.py
python convert_rpw.py
```

**Step 3.3. Model Training**
Train the 1D-CNN models for both RPW and ECG data. Each script will automatically run 5-fold cross-validation and save logs/models to `{config.directory}/models/`:
```bash
python train_rpw_model.py
python train_ecg_model.py
```

**Step 3.4. Model Evaluation**
Select the best configuration per modality (highest mean validation AUC across folds), evaluate the fold models and their ensembles on the hold-out test set, train and evaluate the clinical reference model on the identical folds, and compute a paired athlete-level bootstrap (`n_bootstrap` iterations, default 10,000) for confidence intervals and AUC differences:
```bash
python evaluate_results.py
```
All outputs are written to `{config.directory}/{results_name}` (see `config.get_results_dir()`), including:
- `test_predictions.csv`: Ensemble test-set probabilities of all models per examination.
- `bootstrap.csv`, `bootstrap_samples.csv`: Bootstrap summary and all iterations.
- `results_table.tex`, `delta_auc_table.tex`: Per-fold/ensemble metrics and paired AUC differences as LaTeX tables.
- `roc_test.pdf/.png`, `roc_plots.pdf`: Test-set ROC curves of the ensembles and fold-level ROC curves.
- `val_aucs_rpw.csv`, `val_aucs_ecg.csv`, `clinical_missing_values.csv`: Model selection and missing values of the clinical variables.
- `xai_plots/`: Signals and LRP relevance maps of all true-positive test predictions of the individual fold models (input for Steps 3.5 and 3.6).

**Step 3.5. Global Explainability (XAI)**
Generate the Layer-wise Relevance Propagation (LRP) visualizations across the test sets to analyze feature importance:
```bash
python global_xai.py
```

**Step 3.6. Segment-level ECG Explainability**
Quantify the ECG explanations per lead (relevance share) and per waveform segment (relevance-to-duration ratio), averaged per recording and then across recordings:
```bash
python segment_xai.py                 # reuses an existing fiducials.csv
python segment_xai.py --redelineate   # overwrites fiducials.csv with the automatic delineation
```
Segment borders are first delineated automatically on the mean beat of each recording and stored in `segment_xai/fiducials.csv`. They should be checked with the plots in `segment_xai/qc/` and can be corrected manually in the CSV; subsequent runs reuse the edited file. Outputs include the radar charts, `segment_table.tex`, and per-window/per-recording CSV files.

<a id="data-availability"></a>
## 📊 Data Availability
The clinical datasets analyzed in the present study contain potentially identifiable health information from professional athletes and are therefore not publicly available. Deidentified data may be made available from the corresponding author (P.B.) upon reasonable scientific request and subject to institutional approval, data protection regulations and applicable participant consent.

<a id="citation"></a>
## 📜 Citation
If you use this code or find our work helpful, please cite our paper:

```bibtex
@article{eckerle2026ebpr,
  title={Resting electrocardiographic and pulse-wave deep learning identifies exaggerated exercise blood pressure response in elite athletes},
  author={Eckerle, Dominic and Gumpfer, Nils and Guckert, Michael and Bauer, Pascal and Hannig, Jennifer},
  journal={npj Digital Medicine},
  year={2026},
  note={Under review (Submitted July 10, 2026)}
}
```

<a id="license"></a>
## ⚖️ License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

<a id="acknowledgments"></a>
## 🙏 Acknowledgments
This work was supported by THMconnectFCMH Funds funded by the Hessian Ministry of Higher Education, Research, Science and the Arts (HMWK) through the project “BPEAX”. This work was further supported in part by the German Federal Ministry of Research, Technology and Space (BMFTR) through ExperTeam4KI (grant no. 16IS24063). We gratefully acknowledge support from the hessian.AI Service Center (funded by the BMFTR, grant no. 16IS22091) and the hessian.AI Innovation Lab (funded by the Hessian Ministry for Digital Strategy and Innovation, grant no. SDIW04/0013/003). The funders had no role in the design of the study; the collection, analysis, or interpretation of data; the preparation of the manuscript; or the decision to submit the manuscript for publication. We thank Denise Lange for her preliminary work on this topic during her Master’s thesis.

<a id="author-contributions"></a>
## 👥 Author Contributions
Conceptualization: M.G., P.B. and J.H. Methodology: D.E., N.G. and J.H. Formal analysis: D.E. and N.G. Supervision: M.G., P.B. and J.H. Project administration: J.H. Funding acquisition: J.H. Writing – original draft: D.E. and J.H. Writing – review & editing: D.E., N.G., M.G., P.B. and J.H.

<a id="competing-interests"></a>
## 🤝 Competing Interests
The authors declare no financial or non-financial competing interests related to this work.
