import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import full_history_features, FEATURES, CAT_FEATURES, TARGET, DATA

MODELS = Path(__file__).parent / "models"
MODELS.mkdir(parents=True, exist_ok=True)
VAL_START = pd.Timestamp("2025-07-01")
FIRST_USABLE = pd.Timestamp("2023-02-19")

PARAMS = dict(objective="tweedie", tweedie_variance_power=1.2, learning_rate=0.04, num_leaves=63,
              min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, seed=7)


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    keep = ~np.isnan(actual) & ~np.isnan(predicted)
    actual, predicted = actual[keep], predicted[keep]
    if actual.size == 0:
        raise ValueError("Cannot calculate forecast metrics for an empty evaluation set.")
    error = actual - predicted
    actual_total = float(actual.sum())
    predicted_total = float(predicted.sum())
    wape = 100 * float(np.abs(error).sum()) / actual_total if actual_total != 0 else float("nan")
    bias = 100 * (predicted_total - actual_total) / actual_total if actual_total != 0 else float("nan")
    return {
        "WAPE_%": round(wape, 2),
        "MAE": round(float(np.mean(np.abs(error))), 3),
        "RMSE": round(float(np.sqrt(np.mean(error ** 2))), 3),
        "Bias_%": round(bias, 2),
    }


def fit(X, y, params, rounds, X_val=None, y_val=None):
    train_set = lgb.Dataset(X, y, categorical_feature=CAT_FEATURES, free_raw_data=False)
    if X_val is None:
        return lgb.train(params, train_set, num_boost_round=rounds)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set, categorical_feature=CAT_FEATURES)
    return lgb.train(params, train_set, num_boost_round=rounds, valid_sets=[val_set],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def quantile_params(alpha):
    params = dict(PARAMS, objective="quantile", alpha=alpha)
    params.pop("tweedie_variance_power")
    return params


def main():
    print("Building features...")
    df, demand, branches = full_history_features()
    df = df[df.date >= FIRST_USABLE]
    labelled = df[df[TARGET].notna()]
    train = labelled[labelled.date < VAL_START]
    val = labelled[labelled.date >= VAL_START].copy()
    print(f"train rows {len(train):,}  validation rows {len(val):,}")
    if train.empty:
        raise ValueError(
            f"No training rows found before {VAL_START.date()}; "
            "check the source data and FIRST_USABLE date."
        )
    if val.empty:
        raise ValueError(
            f"No validation rows found on or after {VAL_START.date()}; "
            "check the source data and validation boundary."
        )
    if train[TARGET].isna().any() or val[TARGET].isna().any():
        raise ValueError("Training and validation targets must not contain missing values.")

    print("Training point model...")
    model = fit(train[FEATURES], train[TARGET], PARAMS, 3000, val[FEATURES], val[TARGET])
    best = model.best_iteration
    val["pred"] = model.predict(val[FEATURES], num_iteration=best)

    print("Training quantile models...")
    for alpha in (0.1, 0.9):
        quantile_model = fit(train[FEATURES], train[TARGET], quantile_params(alpha), best)
        val[f"p{int(alpha * 100)}"] = quantile_model.predict(val[FEATURES])
    coverage = float(((val[TARGET] >= val.p10) & (val[TARGET] <= val.p90)).mean())

    candidates = {
        "LightGBM (this model)": val.pred,
        "Naive: same weekday 4 wks ago": val.lag_28,
        "Naive: avg of last 4 same weekdays": val.dow_mean_4w,
        "Naive: same day last year": val.lag_364,
    }
    results = {name: metrics(val[TARGET], pred) for name, pred in candidates.items()}

    branch_day = (val.groupby(["date", "branch_id"], observed=False)[[TARGET, "pred", "dow_mean_4w"]]
                  .sum().reset_index())
    results_branch_day = {
        "LightGBM (this model)": metrics(branch_day[TARGET], branch_day.pred),
        "Naive: avg of last 4 same weekdays": metrics(branch_day[TARGET], branch_day.dow_mean_4w),
    }
    wape_by_branch = {branch: metrics(g[TARGET], g.pred)["WAPE_%"]
                      for branch, g in branch_day.groupby("branch_id", observed=False)}
    wape_by_service = {str(service): metrics(g[TARGET], g.pred)["WAPE_%"]
                       for service, g in val.groupby("service_type", observed=True)}

    print("\nValidation (Jul-Dec 2025), branch x service x day:")
    print(pd.DataFrame(results).T.to_string())
    print("\nAggregated to branch x day:")
    print(pd.DataFrame(results_branch_day).T.to_string())
    print(f"\nP10-P90 interval coverage: {coverage:.1%} (target 80%)")

    importance = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    importance = (100 * importance / importance.sum()).round(2)

    print("\nRefitting on full history...")
    full_rounds = int(best * 1.1)
    final = fit(labelled[FEATURES], labelled[TARGET], PARAMS, full_rounds)
    final_quantiles = {
        f"p{int(alpha * 100)}": fit(labelled[FEATURES], labelled[TARGET], quantile_params(alpha), full_rounds)
        for alpha in (0.1, 0.9)
    }

    jobs = pd.read_csv(DATA / "jobs.csv", usecols=["service_type", "duration_hours", "invoice_usd"])
    service_stats = jobs.groupby("service_type").agg(
        avg_duration_h=("duration_hours", "mean"),
        avg_invoice_usd=("invoice_usd", "mean"),
    ).round(2)

    final.save_model(str(MODELS / "model.txt"))
    for name, quantile_model in final_quantiles.items():
        quantile_model.save_model(str(MODELS / f"model_{name}.txt"))

    val_out = val[["date", "branch_id", "service_type", TARGET, "pred", "p10", "p90", "dow_mean_4w"]].copy()
    val_out["branch_id"] = val_out.branch_id.astype(str)
    val_out["service_type"] = val_out.service_type.astype(str)
    val_out["date"] = val_out.date.dt.strftime("%Y-%m-%d")
    val_out.round(4).to_csv(MODELS / "validation.csv", index=False)

    meta = {
        "features": FEATURES,
        "cat_features": CAT_FEATURES,
        "categories": {col: [str(x) for x in labelled[col].cat.categories] for col in CAT_FEATURES},
        "metrics": results,
        "metrics_branch_day": results_branch_day,
        "wape_by_branch": wape_by_branch,
        "wape_by_service": wape_by_service,
        "interval_coverage": coverage,
        "best_iteration": int(best),
        "feature_importance": {k: float(v) for k, v in importance.items()},
        "service_stats": {svc: {col: float(x) for col, x in row.items()}
                          for svc, row in service_stats.iterrows()},
        "val_start": str(VAL_START.date()),
        "history_end": str(demand.date.max().date()),
    }
    with open(MODELS / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved models/ (model.txt, model_p10.txt, model_p90.txt, meta.json, validation.csv)")


if __name__ == "__main__":
    main()
