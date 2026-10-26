import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import build_features, load_raw, TARGET, MIN_LAG

SHIFT_HOURS_PER_TECH = 7.5
Z80 = 1.2816


def load_bundle(models_dir):
    folder = Path(models_dir)
    with open(folder / "meta.json") as f:
        bundle = json.load(f)

    bundle["model"] = lgb.Booster(model_file=str(folder / "model.txt"))

    burst_file = folder / "model_burst.txt"
    if burst_file.exists() and bundle.get("burst_service"):
        bundle["burst_model"] = lgb.Booster(model_file=str(burst_file))
    else:
        bundle["burst_model"] = None

    bundle["quantile_models"] = {
        name: lgb.Booster(model_file=str(folder / f"model_{name}.txt")) for name in ("p10", "p90")
    }
    bundle["service_stats"] = pd.DataFrame(bundle["service_stats"]).T
    bundle["validation"] = pd.read_csv(folder / "validation.csv", parse_dates=["date"])
    return bundle


@dataclass
class Scenario:
    temp_shift_c: float = 0.0
    cold_snap_start: int | None = None
    cold_snap_days: int = 3
    cold_snap_depth_c: float = 15.0
    storm_day: int | None = None
    storm_mm: float = 50.0
    promo_days: list = field(default_factory=list)
    marketing_mult: float = 1.0


def climatology(weather):
    w = weather.copy()
    w["doy"] = w.date.dt.dayofyear.clip(upper=365)
    cols = ["temp_avg_c", "temp_min_c", "temp_max_c", "precip_mm"]
    daily_avg = w.groupby(["branch_id", "doy"])[cols].mean()

    frames = []
    for branch, group in daily_avg.groupby(level=0):
        year = group.droplevel(0).reindex(range(1, 366)).interpolate(limit_direction="both")
        wrapped = pd.concat([year.iloc[-15:], year, year.iloc[:15]])
        smooth = wrapped.rolling(15, center=True, min_periods=1).mean().iloc[15:-15]
        smooth["branch_id"] = branch
        frames.append(smooth.reset_index().rename(columns={"index": "doy"}))
    return pd.concat(frames, ignore_index=True)


def future_weather(weather, branch_ids, dates, sc: Scenario):
    normals = climatology(weather)
    frames = []
    for branch in branch_ids:
        f = pd.DataFrame({"date": dates, "branch_id": branch,
                          "doy": np.minimum(dates.dayofyear.values, 365)})
        f = f.merge(normals, on=["branch_id", "doy"], how="left")

        shift = np.full(len(f), sc.temp_shift_c)
        if sc.cold_snap_start is not None:
            start, length = sc.cold_snap_start, sc.cold_snap_days
            profile = np.array([0.7, 1.0, 1.0, 0.9, 0.8, 0.6, 0.5][:length] + [0.5] * max(0, length - 7))
            shift[start:start + length] -= sc.cold_snap_depth_c * profile[:max(0, min(length, len(f) - start))]
        for col in ("temp_avg_c", "temp_min_c", "temp_max_c"):
            f[col] = f[col] + shift

        if sc.storm_day is not None and 0 <= sc.storm_day < len(f):
            f.loc[sc.storm_day, "precip_mm"] = sc.storm_mm
        f["snow_cm"] = np.where(f.temp_avg_c < 0, f.precip_mm * 0.11, 0.0)
        frames.append(f.drop(columns="doy"))
    return pd.concat(frames, ignore_index=True)


def future_marketing(marketing, branch_ids, dates, sc: Scenario):
    m = marketing.copy()
    m["month"] = m.date.dt.month
    last_year = m[m.date.dt.year == m.date.dt.year.max()]
    monthly_spend = last_year.groupby(["branch_id", "month"]).marketing_spend_usd.mean()

    frames = []
    for branch in branch_ids:
        f = pd.DataFrame({"date": dates, "branch_id": branch})
        f["marketing_spend_usd"] = [
            monthly_spend.get((branch, day.month), monthly_spend.loc[branch].mean()) for day in dates
        ]
        f["marketing_spend_usd"] *= np.where(dates.weekday >= 5, 0.6, 1.15) * sc.marketing_mult
        f["promo_active"] = [1 if i in sc.promo_days else 0 for i in range(len(dates))]
        f.loc[f.promo_active == 1, "marketing_spend_usd"] *= 1.8
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def apply_categories(X, categories):
    X = X.copy()
    for col, levels in categories.items():
        X[col] = pd.Categorical(X[col].astype(str), categories=levels)
    return X


def make_forecast(bundle, branch_ids, horizon=14, scenario: Scenario | None = None, raw=None):
    sc = scenario or Scenario()
    horizon = int(min(max(horizon, 1), MIN_LAG))
    demand, weather, marketing, branches = raw or load_raw()

    last_day = demand.date.max()
    start = last_day + pd.Timedelta(days=1)
    dates = pd.date_range(start, periods=horizon, freq="D")
    services = sorted(demand.service_type.unique())

    history = demand[demand.branch_id.isin(branch_ids)][["date", "branch_id", "service_type", TARGET]]
    history = history[history.date >= last_day - pd.Timedelta(days=420)]
    future = pd.MultiIndex.from_product(
        [dates, branch_ids, services], names=["date", "branch_id", "service_type"]
    ).to_frame(index=False)
    future[TARGET] = np.nan
    panel = pd.concat([history, future], ignore_index=True)

    all_weather = pd.concat([weather[weather.branch_id.isin(branch_ids)],
                             future_weather(weather, branch_ids, dates, sc)], ignore_index=True)
    all_marketing = pd.concat([marketing[marketing.branch_id.isin(branch_ids)],
                               future_marketing(marketing, branch_ids, dates, sc)], ignore_index=True)

    df = build_features(panel, all_weather, all_marketing, branches)
    df = df[df.date >= start].reset_index(drop=True)
    X = apply_categories(df[bundle["features"]], bundle["categories"])

    out = df[["date", "branch_id", "service_type", "temp_min_c", "precip_mm", "promo_active"]].copy()
    out["branch_id"] = out.branch_id.astype(str)
    out["service_type"] = out.service_type.astype(str)

    pred = bundle["model"].predict(X)
    if bundle.get("burst_model") is not None:
        is_burst = (out.service_type == bundle["burst_service"]).values
        if is_burst.any():
            pred[is_burst] = bundle["burst_model"].predict(X[is_burst])

    out["forecast"] = np.clip(pred, 0, None)
    out["p10"] = np.clip(bundle["quantile_models"]["p10"].predict(X), 0, None)
    out["p90"] = np.clip(bundle["quantile_models"]["p90"].predict(X), 0, None)
    out["p10"] = np.minimum(out.p10, out.forecast)
    out["p90"] = np.maximum(out.p90, out.forecast)

    stats = bundle["service_stats"]
    out["tech_hours"] = out.forecast * out.service_type.map(stats.avg_duration_h)
    out["exp_revenue_usd"] = out.forecast * out.service_type.map(stats.avg_invoice_usd)
    return out


def agg_interval(fc, by):
    with_var = fc.assign(_var=((fc.p90 - fc.p10) / (2 * Z80)) ** 2)
    totals = with_var.groupby(by).agg(
        forecast=("forecast", "sum"),
        _var=("_var", "sum"),
        tech_hours=("tech_hours", "sum"),
        exp_revenue_usd=("exp_revenue_usd", "sum"),
    ).reset_index()
    spread = np.sqrt(totals.pop("_var"))
    totals["p10"] = np.clip(totals.forecast - Z80 * spread, 0, None)
    totals["p90"] = totals.forecast + Z80 * spread
    return totals


def staffing_plan(fc, demand):
    plan = agg_interval(fc, ["date", "branch_id"])
    peak_ratio = (plan.p90 / plan.forecast.replace(0, np.nan)).fillna(1)
    plan["techs_needed"] = np.ceil(plan.tech_hours / SHIFT_HOURS_PER_TECH).astype(int)
    plan["techs_needed_p90"] = np.ceil(plan.tech_hours * peak_ratio / SHIFT_HOURS_PER_TECH).astype(int)

    recent = demand[demand.date > demand.date.max() - pd.Timedelta(weeks=8)]
    recent = recent.drop_duplicates(["date", "branch_id"]).assign(dow=lambda x: x.date.dt.weekday)
    roster = recent.groupby(["branch_id", "dow"]).technicians_on_shift.median().rename("usual_roster")

    plan["dow"] = plan.date.dt.weekday
    plan = plan.merge(roster.reset_index(), on=["branch_id", "dow"], how="left").drop(columns="dow")
    plan["gap_vs_roster"] = plan.techs_needed - plan.usual_roster
    return plan
