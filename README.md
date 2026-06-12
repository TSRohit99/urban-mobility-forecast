# Urban Mobility Forecasting

Deep learning models for **hourly bike-sharing demand prediction** using the Seoul public bike dataset (2017–2018). The project compares recurrent and hybrid architectures for **next-hour (1-step)** and **24-hour-ahead (multi-step)** forecasting under a strict chronological evaluation protocol.

---

## Overview

Accurate short-term demand forecasts help operators rebalance fleets, staff stations, and plan capacity. This repository implements an end-to-end pipeline—data cleaning, feature engineering, model training, and systematic comparison—implemented primarily in **`model.ipynb`**.

**Forecasting tasks**

| Task | Horizon | Best model (test) | RMSE | MAE | R² |
|------|---------|-------------------|------|-----|-----|
| Next-hour demand | 1 step | CNN-LSTM (tuned, raw weather) | 254.3 | 178.9 | 0.78 |
| Day-ahead profile | 24 steps | Encoder–decoder LSTM | 359.7* | 250.1* | 0.56* |

\*24-step metrics are overall (all horizons flattened). See [`model_comparison_report.txt`](model_comparison_report.txt) for per-horizon breakdown.

---

## Dataset

- **Source:** Seoul bike-sharing hourly records with weather and calendar features (`dataset.csv` included in this repo).
- **Period:** December 2017 – November 2018 (8,760 raw rows → 8,297 after cleaning and feature lagging).
- **Target:** `Rented Bike Count` (continuous regression).
- **Split:** Chronological **70% / 15% / 15%** (train / validation / test)—no random shuffle, so test data is genuinely future-held-out.

**Input representation:** Each training example uses the previous **24 hours** of features as a sequence tensor of shape `(24, F)`.

---

## Methodology

### Preprocessing & features

- Missing-hour checks, outlier capping (99th percentile on target), `log1p` on rainfall/snowfall.
- Calendar encodings (hour, day, month, season, holiday), cyclic sin/cos time features.
- Demand lags and rolling means; peak-hour flag (07–09, 17–20).
- **Weather ablation:** `raw_weather` (rainfall, snowfall, visibility) vs. composite `weather_severity` index.
- **Scaling:** `MinMaxScaler` fit on **training data only** for features and target; metrics reported after inverse transform to original bike counts.

### Training protocol

- Optimizer: Adam (`lr=1e-3` baselines; tuned model uses `5e-4`).
- Loss: MSE on scaled target.
- Up to **50 epochs**, batch size **256**, early stopping (patience 8), `ReduceLROnPlateau`, best-checkpoint saving.
- Reproducibility: random seed **42**.

---

## Models Compared

| Architecture | Description |
|--------------|-------------|
| **Vanilla LSTM** | Baseline sequence model |
| **BiLSTM** | Bidirectional LSTM over the 24h window |
| **CNN-LSTM** | Causal Conv1D → pooling → LSTM (primary hybrid) |
| **CNN-LSTM + attention** | Temporal attention over LSTM sequence outputs |
| **Peak / off-peak specialists** | Separate CNN-LSTM models trained on rush vs. non-rush target hours |
| **CNN-LSTM (Keras Tuner)** | Hyperparameter search (10 trials) over filters, kernel, units, dropout, LR |
| **Encoder–decoder LSTM** | 24-step vector forecast in one forward pass |

Full architectural notes, cell-by-cell walkthrough, and design rationale: [`project-explanation.md`](project-explanation.md).

---

## Key Findings

1. **Tuned CNN-LSTM with raw weather** achieves the best full-test 1-step performance (RMSE **254.3**, sMAPE **28.6%**).
2. **Raw weather features outperform** the single severity index for CNN-LSTM (~42 bikes lower RMSE on test).
3. **CNN front-end helps** vs. plain LSTM (~264 vs. ~303 RMSE on comparable splits).
4. **Attention did not beat** plain CNN-LSTM on RMSE despite added capacity.
5. **Demand regimes differ:** off-peak specialist RMSE **232** (off-peak test only) vs. peak specialist **337** (peak test only).
6. **24-hour forecasting is harder** than 1-step, as expected; error rises through ~12h then plateaus.

Detailed tables: [`model_comparison_report.txt`](model_comparison_report.txt) · Summary paper: [`demo_paper.pdf`](demo_paper.pdf)

---

## Repository Structure

```
.
├── README.md                      # This file
├── model.ipynb                    # Full pipeline: EDA → training → evaluation
├── dataset.csv                    # Seoul bike-sharing dataset
├── requirements.txt               # Python dependencies
├── project-explanation.md         # In-depth notebook & experiment guide
├── model_comparison_report.txt    # Metrics tables and notes
└── demo_paper.pdf                 # Project write-up / results summary
```

---

## Getting Started

### Prerequisites

- Python **3.10+** recommended
- pip

### Installation

```bash
git clone https://github.com/tsrohit99/urban-mobility-forecast.git
cd urban-mobility-forecast

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook model.ipynb
```

Execute cells **in order** from top to bottom. Training all models may take considerable time depending on hardware (GPU recommended for TensorFlow).

After the final evaluation cell, metrics are written to `model_comparison_report.txt` (a copy is committed in this repo; re-running refreshes it locally).

---

## Evaluation Metrics

This is a **regression** task—there is no classification accuracy.

| Metric | Meaning |
|--------|---------|
| **RMSE** | Root mean squared error (bikes); penalizes large errors |
| **MAE** | Mean absolute error (bikes); typical error magnitude |
| **R²** | Explained variance vs. predicting the mean |
| **sMAPE** | Symmetric percentage error; more stable than MAPE at low demand |

Prefer **RMSE / MAE / R²** for ranking; use **sMAPE** when discussing relative error.

---

## Limitations

- Single temporal split; results are architecture comparisons under one evaluation design, not production guarantees.
- Peak/off-peak specialist metrics are on **subsets** and are not directly rank-comparable to full-test RMSE.
- BiLSTM uses future timesteps within the input window (valid for offline scoring; not strictly causal deployment).
- Point forecasts only—no uncertainty intervals or ensembles.
- Trained model weights (`.keras` checkpoints) are **not** included in this repository; re-run the notebook to regenerate them locally.

---

## References & Data

- Seoul Public Bike Sharing Dataset (hourly demand with weather and calendar attributes).
- Frameworks: [TensorFlow/Keras](https://www.tensorflow.org/), [scikit-learn](https://scikit-learn.org/), [Keras Tuner](https://keras.io/keras_tuner/).

---

## License

Educational / academic project. Dataset rights remain with the original data provider. 