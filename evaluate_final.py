"""Reproduce frozen-model test results for the CBB margin project."""
import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEED = 42
TARGET = "target_home_margin"
MARKET = "market_home_margin"


def load_test_data(data_dir):
    frames = []
    for year in (2025, 2026):
        path = data_dir / f"games_{year - 1}_{year}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing test CSV: {path}. See README data instructions.")
        frame = pd.read_csv(path, low_memory=False)
        frame["test_season"] = year
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if data[TARGET].isna().any():
        raise ValueError("Test targets contain missing values")
    return data


def evaluate(data, prediction):
    actual = data[TARGET].to_numpy()
    return (mean_absolute_error(actual, prediction),
            np.sqrt(mean_squared_error(actual, prediction)))


def paired_date_bootstrap(data, seed=SEED, repetitions=5000):
    tmp = data.copy()
    tmp["day"] = pd.to_datetime(tmp["date"], utc=True).dt.date
    tmp["difference"] = (np.abs(tmp[TARGET] - tmp["ridge_prediction"])
                         - np.abs(tmp[TARGET] - tmp["xgb_prediction"]))
    daily = tmp.groupby("day").agg(total=("difference", "sum"), n=("difference", "size"))
    sums = daily["total"].to_numpy()
    counts = daily["n"].to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.empty(repetitions)
    for i in range(repetitions):
        ix = rng.integers(0, len(daily), len(daily))
        boot[i] = sums[ix].sum() / counts[ix].sum()
    return float(tmp["difference"].mean()), np.percentile(boot, [2.5, 97.5]), len(daily)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("cbb_data"))
    parser.add_argument("--model-dir", type=Path, default=Path("cbb_models"))
    parser.add_argument("--output-dir", type=Path, default=Path("cbb_final_results"))
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    df = load_test_data(args.data_dir)
    with (args.model_dir / "linear_features.json").open() as f:
        ridge_features = json.load(f)
    with (args.model_dir / "xgboost_features.json").open() as f:
        xgb_features = json.load(f)
    required = set(ridge_features + xgb_features + [TARGET, MARKET, "date"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    ridge = joblib.load(args.model_dir / "ridge_model.joblib")
    xgb = joblib.load(args.model_dir / "xgboost_model.joblib")
    df["ridge_prediction"] = ridge.predict(df[ridge_features])
    df["xgb_prediction"] = xgb.predict(df[xgb_features])
    df["zero_prediction"] = 0.0
    for col in ("ridge_prediction", "xgb_prediction"):
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Nonfinite predictions in {col}")
    covered = df[df[MARKET].notna()].copy()
    results = []
    for name, col in (("Zero Margin", "zero_prediction"), ("Ridge", "ridge_prediction"),
                      ("XGBoost", "xgb_prediction"), ("Sportsbook", MARKET)):
        mae, rmse = evaluate(covered, covered[col].to_numpy())
        results.append({"model": name, "mae": mae, "rmse": rmse, "games": len(covered)})
    result_df = pd.DataFrame(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output_dir / "matched_test_metrics.csv", index=False)
    df.to_csv(args.output_dir / "test_predictions.csv", index=False)
    print(f"Fixed seed: {SEED}")
    print(f"All test games: {len(df):,}; market-covered: {len(covered):,}")
    print(result_df.to_string(index=False, formatters={"mae": "{:.3f}".format, "rmse": "{:.3f}".format}))
    improvement, ci, days = paired_date_bootstrap(covered)
    print(f"XGBoost MAE improvement over Ridge: {improvement:.4f} points")
    print(f"Date-clustered paired bootstrap 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}] ({days} dates)")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
