# Engine Degradation Risk Assessment Pipeline

A multi-stage predictive maintenance system built on the NASA CMAPSS turbofan engine dataset. The pipeline automatically derives degradation stages from raw sensor data — no manual labels required — and combines classification and regression models to generate real-time risk scores and maintenance alerts.

---

## Overview

Traditional predictive maintenance approaches rely on manually defined failure thresholds. This project takes a different approach: unsupervised clustering is used to **automatically discover degradation stages** from sensor behaviour, which then become the targets for a supervised ML pipeline. The final output is a normalized risk score per engine per time step, with automated maintenance alerts triggered above a configurable threshold.

---

## Pipeline

```
Raw Sensor Data (NASA CMAPSS)
        │
        ▼
Phase 0: Data Loading & Preprocessing
  - Combines multiple CMAPSS datasets (FD001, FD003)
  - Creates unique engine IDs across datasets
  - Identifies 21 sensor + 3 operational setting features
        │
        ▼
Phase 1: Unsupervised Clustering → Degradation Stage Labels
  - Scales sensor readings
  - Applies KMeans / GMM clustering
  - Interprets clusters as degradation stages (0 → 4) based on time ordering
        │
        ▼
Phase 2: Rolling Feature Engineering
  - Computes rolling mean + std (window = 50 cycles) for all sensor columns
  - Captures temporal degradation trends
        │
        ▼
Phase 3: Stage Classification
  - Predicts current degradation stage
  - Models: XGBoost, Logistic Regression, SVC
        │
        ▼
Phase 4: Time-to-Next-Stage (TTNS) Regression
  - Predicts how many cycles until the next degradation stage
  - Models: XGBoost, HistGradientBoosting, NuSVR
        │
        ▼
Phase 5: Risk Scoring & Maintenance Alerts
  - Combines stage transition probability + TTNS prediction
  - Normalizes into a 0–1 risk score
  - Flags engines above configurable risk threshold (default: 0.7)
```

---

## Key Results

| Phase | Model | Metric |
|---|---|---|
| Stage Classification | SVC | Stage prediction with probability outputs |
| TTNS Regression | XGBoost | Best TTNS prediction performance |
| Risk Pipeline | Combined | Maintenance alerts at configurable threshold |

Tested on combined FD001 + FD003 datasets — **45,351 training samples**, **9,071 validation samples**, 65 output features including risk scores and alert flags.

---

## Dataset

**NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)**
- 21 sensor readings per time step
- 3 operational settings
- Multiple run-to-failure trajectories across engine units
- Datasets: FD001 (single fault, single condition) + FD003 (single fault, multiple conditions)

---

## Tech Stack

`Python` `Scikit-learn` `XGBoost` `NumPy` `Pandas` `Matplotlib` `KMeans` `GMM` `NuSVR` `HistGradientBoosting`

---

## What Makes This Interesting

Most RUL (Remaining Useful Life) prediction work treats this as a pure regression problem — predict cycles until failure. This project reframes it as a **staged degradation problem**, which is more realistic for maintenance scheduling: knowing *which degradation stage* an engine is in, and *how long until the next stage*, gives operators more actionable information than a single RUL number.

The unsupervised stage derivation means the pipeline can be adapted to any sensor dataset without needing manually annotated failure labels.

---

## Authors

Group project — Mahindra University (Semester 5 ML Course)
- **Anushka Mohanty** — clustering pipeline, degradation stage derivation, risk scoring logic
