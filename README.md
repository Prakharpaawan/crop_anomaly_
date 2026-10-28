# Crop Stress Detection Using Sentinel-2 Imagery

This project uses Sentinel-2 satellite time-series data to detect potential crop stress at the field level. It uses unsupervised anomaly detection, so the models do not require labelled training data.

**Dataset:** 3,328 fields from Lombardy, Italy  
**Period:** 2019  
**Observations:** 83,200 across 25 satellite passes

## Results

The current pipeline uses Local Outlier Factor (LOF), a Generalized Pareto Distribution (GPD) threshold, and a two-consecutive-observation persistence rule.

Evaluation on 832 held-out fields, using simulated stress at 40% severity over two observations, produced the following results:

| Metric | Score |
|---|---:|
| Precision | 0.876 |
| Recall | 0.814 |
| F1-score | 0.844 |
| PR-AUC | 0.904 |

These results are based on simulated anomalies and should not be interpreted as equivalent to performance on independently verified crop stress in real fields.

## Project Structure

```text
PBL Project/
├── README.md
├── HANDOFF.md
├── crop_stress/
│   ├── study/          # Main anomaly detection study
│   ├── engine/         # Shared configuration, utilities and models
│   ├── appendix/       # Additional experiments
│   ├── multispectral/  # Study using four spectral indices
│   ├── outputs/        # Reports, figures and tables
│   └── cache/          # Cached intermediate data
├── paper/              # Research paper draft
├── dataset_clean_2/    # MSAVI raster data
├── archive (1)/        # Original nine-band data archive
└── old/                # Earlier project files
```

## Where to Start

- `crop_stress/README.md` — overview of the code and project setup.
- `crop_stress/outputs/study/REPORT.md` — main study results.
- `crop_stress/outputs/study/PROVENANCE.md` — data sources and experiment details.
- `paper/README.md` — information about the research paper.
- `HANDOFF.md` — notes on the current project state and pending work.

## Running the Project

The project uses Python and relies on shared configuration, utilities, feature tables and satellite raster data. The `.venv` folder at the repository root is currently empty.

The Python interpreter used during development is:

```text
C:/Users/prakh/AppData/Local/Programs/Python/Python314/python.exe
```

To run the main study:

```bash
cd crop_stress/study
python run_all.py
```

This rebuilds the study's tables, figures, report and results file. The first run takes approximately 35 minutes, depending on the machine and available data.

To regenerate the figures from stored results:

```bash
cd crop_stress/study
python make_figures.py
```

Make sure the required dependencies are installed and the expected input files are available before running either command.

## Evaluation Method

The 3,328 fields are divided into three disjoint sets:

- **Training — 1,664 fields:** The anomaly detectors are fitted using clean observations.
- **Calibration — 832 fields:** Used to select the threshold and persistence setting.
- **Testing — 832 fields:** Reserved for the final reported evaluation.

Thresholds are calibrated using clean, pre-injection anomaly scores. The test set is not used to select the detector or its operating parameters.

## Key Findings

LOF performed better than the hybrid ensemble evaluated in this project. The LOF configuration achieved an F1-score of 0.844, compared with 0.354 for the hybrid ensemble. The difference was mainly due to recall, which fell to 0.230 with the hybrid approach.

The evaluation also identified several limitations and trade-offs:

- The recommended configuration performed better in nine of eleven tested severity-duration combinations, but results were weaker for the least severe anomalies.
- The stress events were simulated by scaling six of the sixteen feature columns. This makes the evaluation useful for comparing configurations, but it does not establish equivalent performance on real crop stress.
- The reported metrics depend on the simulation settings, feature representation and threshold calibration protocol.
- The multispectral study has separate evaluation limitations documented in its report.

Further details and outstanding work are documented in `HANDOFF.md` and `crop_stress/README.md`.

## Limitations

The project is an experimental crop stress detection pipeline. Its results demonstrate how unsupervised anomaly detection methods compare under the chosen simulation protocol. Validation against independently labelled real-world crop stress is still needed before drawing conclusions about operational performance in agricultural settings.
