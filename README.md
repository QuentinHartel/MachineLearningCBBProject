# College Basketball Scoring-Margin Prediction

Reproducible final evaluation of two frozen models: Lasso-selected Ridge regression and 30-feature XGBoost. The target is **home points minus away points**, not the sportsbook line.

## Headline result

On the **11,161 market-covered games** in the 2024–25 and 2025–26 held-out seasons:

| Model | MAE | RMSE |
|---|---:|---:|
| Zero-margin baseline | 12.337 | 15.848 |
| Ridge | 9.560 | 12.246 |
| XGBoost | **9.422** | **12.021** |
| Historical sportsbook | 8.896 | 11.304 |

All 11,514 test games: Ridge MAE 9.570, XGBoost MAE 9.435. Values above are rounded to three decimal places. The market benchmark uses available historical lines, whose timing is not independently verified.

## Exact command to reproduce the headline number

Run from the repository root after installing the dependencies and providing the data and saved model files:

```bash
python3 evaluate_final.py --data-dir cbb_data --model-dir cbb_models --output-dir cbb_final_results
```

The script prints the matched-game MAE/RMSE table, calculates a paired date-clustered bootstrap confidence interval with **seed 42**, and saves `matched_test_metrics.csv` and `test_predictions.csv`.

## Required repository layout

```text
repo/
  README.md
  evaluate_final.py
  requirements.txt
  cbb_data/
    games_2024_2025.csv
    games_2025_2026.csv
  cbb_models/
    ridge_model.joblib
    linear_features.json
    xgboost_model.joblib
    xgboost_features.json
```

**Data availability:** Commit the two test CSVs to the repository (if their license and repository size permit). The CSVs must be the same versions used for the reported evaluation. If data cannot be committed, provide an authenticated fetch script and document the precise data snapshot; this evaluation script intentionally does not download data. Never commit an API key. If the CSVs are not present, the command fails with a clear missing-file message.

**Model availability:** Commit the four frozen model artifacts above, or supply a documented way to retrieve the exact artifacts. The model files are required for the command to reproduce the published result; retraining is a different workflow. Only load `.joblib` artifacts from trusted sources.

## Installation

Use the same Python and package versions as the environment that created the `.joblib` models. Capture them with:

```bash
python3 -m pip freeze > requirements.txt
```

Commit that generated `requirements.txt`. In a fresh environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On Windows, activate with `.venv\\Scripts\\activate` instead. The evaluation uses pandas, NumPy, scikit-learn, XGBoost, and joblib. `requirements.txt` is a placeholder until generated from the working environment.

## Reproducibility and data leakage

- Fixed seed: **42** for the bootstrap and any random operations in the evaluation script. The saved models were trained separately with their recorded training settings.
- Development seasons: 2016–17 through 2023–24. Held-out test seasons: 2024–25 and 2025–26.
- Feature selection, normalization, and fitting were performed before final test evaluation. The saved Ridge pipeline contains its fitted imputer and scaler; XGBoost uses its frozen selected-feature list.
- Betting lines and final scores are not model inputs. Betting lines are used only for the matched-game benchmark.
- A historical-market comparison is **not** evidence of profitable betting.

## Training from scratch

This README reproduces the **frozen-model headline evaluation**, not the entire model search. To reproduce model training from scratch, the repository must also include the original data pipeline, training/feature-selection notebook or script, its exact configuration, and the required historical training CSVs or a documented fetch process. Do not claim full training reproducibility until that workflow is checked in a clean environment.
