from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
MIN_LAG = 28
TARGET = "calls_received"

CAT_FEATURES = ["branch_id", "service_type", "climate_zone"]
COLD_SERVICE_WEIGHTS = {"burst_frozen_pipe": 1.0, "leak_repair": 0.15, "water_heater": 0.05}

FEATURES = [
    "branch_id", "service_type", "climate_zone", "households_k", "avg_housing_age_yrs",
    "days_since_open",
    "dow", "month", "doy_sin", "doy_cos", "is_weekend", "is_holiday", "is_brown_friday",
    "is_christmas_week", "is_school_holiday", "days_to_holiday", "days_after_holiday",
    "temp_min_c", "temp_avg_c", "precip_mm", "snow_cm", "freeze_severity", "thaw_day",
    "temp_min_lag1", "rain_3d", "freeze_3d",
    "marketing_spend_usd", "promo_active", "mkt_adstock",
    "lag_28", "lag_35", "lag_42", "lag_364", "dow_mean_4w", "roll_mean_7_l28",
    "roll_mean_28_l28", "roll_std_28_l28", "branch_roll_28_l28", "yoy_ratio",
]


def load_raw():
    demand = pd.read_csv(DATA / "daily_demand.csv", parse_dates=["date"])
    weather = pd.read_csv(DATA / "weather_daily.csv", parse_dates=["date"])
    marketing = pd.read_csv(DATA / "marketing_daily.csv", parse_dates=["date"])
    branches = pd.read_csv(DATA / "branches.csv", parse_dates=["open_date"])
    return demand, weather, marketing, branches


def thanksgiving(year):
    november = pd.date_range(f"{year}-11-01", f"{year}-11-30")
    return november[november.weekday == 3][3]


def nth_weekday(year, month, weekday, n):
    days = pd.date_range(f"{year}-{month:02d}-01", periods=31)
    days = days[(days.month == month) & (days.weekday == weekday)]
    return days[n] if n >= 0 else days[-1]


def us_holidays(years):
    holidays = {}
    for year in years:
        holidays[pd.Timestamp(f"{year}-01-01")] = "New Year's Day"
        holidays[nth_weekday(year, 1, 0, 2)] = "MLK Day"
        holidays[nth_weekday(year, 2, 0, 2)] = "Presidents Day"
        holidays[nth_weekday(year, 5, 0, -1)] = "Memorial Day"
        holidays[pd.Timestamp(f"{year}-07-04")] = "Independence Day"
        holidays[nth_weekday(year, 9, 0, 0)] = "Labor Day"
        holidays[thanksgiving(year)] = "Thanksgiving"
        holidays[pd.Timestamp(f"{year}-12-25")] = "Christmas"
    return holidays


def build_calendar(dates):
    dates = pd.DatetimeIndex(sorted(set(dates)))
    years = range(dates.min().year - 1, dates.max().year + 2)
    holidays = us_holidays(years)
    holiday_dates = pd.DatetimeIndex(sorted(holidays))

    cal = pd.DataFrame({"date": dates})
    cal["dow"] = dates.weekday
    cal["month"] = dates.month
    day_of_year = dates.dayofyear
    cal["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    cal["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    cal["is_weekend"] = (cal.dow >= 5).astype(int)
    cal["is_holiday"] = dates.isin(holiday_dates).astype(int)
    cal["holiday_name"] = [holidays.get(day, "") for day in dates]

    brown_fridays = {thanksgiving(year) + pd.Timedelta(days=1) for year in years}
    cal["is_brown_friday"] = dates.isin(list(brown_fridays)).astype(int)
    cal["is_christmas_week"] = ((dates.month == 12) & (dates.day >= 22)).astype(int)

    summer_break = (
        (dates.month == 7)
        | ((dates.month == 6) & (dates.day > 10))
        | ((dates.month == 8) & (dates.day < 20))
    )
    cal["is_school_holiday"] = (summer_break | (cal.is_christmas_week == 1)).astype(int)

    pos = np.searchsorted(holiday_dates.values, dates.values)
    next_holiday = holiday_dates[np.clip(pos, 0, len(holiday_dates) - 1)]
    prev_holiday = holiday_dates[np.clip(pos - 1, 0, len(holiday_dates) - 1)]
    cal["days_to_holiday"] = np.clip((next_holiday - dates).days, 0, 30)
    cal["days_after_holiday"] = np.clip((dates - prev_holiday).days, 0, 30)
    return cal


def adstock(spend, decay=0.55):
    carried = np.zeros(len(spend))
    for i, value in enumerate(spend):
        carried[i] = value + (decay * carried[i - 1] if i else 0.0)
    return carried


def prepare_weather(weather):
    w = weather.sort_values(["branch_id", "date"]).copy()
    for col in ["temp_avg_c", "temp_min_c", "temp_max_c", "precip_mm"]:
        w[col] = w.groupby("branch_id")[col].transform(lambda s: s.interpolate(limit_direction="both"))
    w["snow_cm"] = w["snow_cm"].fillna(0)

    by_branch = w.groupby("branch_id")
    w["freeze_severity"] = np.clip(-6 - w.temp_min_c, 0, None)
    w["temp_min_lag1"] = by_branch.temp_min_c.shift(1)
    w["thaw_day"] = ((w.temp_min_lag1 <= -8) & (w.temp_min_c > -2)).astype(int)
    w["rain_3d"] = by_branch.precip_mm.transform(lambda s: s.rolling(3, min_periods=1).sum())
    w["freeze_3d"] = w.groupby("branch_id").freeze_severity.transform(
        lambda s: s.rolling(3, min_periods=1).sum())

    return w[["date", "branch_id", "temp_min_c", "temp_avg_c", "precip_mm", "snow_cm",
              "freeze_severity", "thaw_day", "temp_min_lag1", "rain_3d", "freeze_3d"]]


def prepare_marketing(marketing):
    m = marketing.sort_values(["branch_id", "date"]).copy()
    m["mkt_adstock"] = m.groupby("branch_id").marketing_spend_usd.transform(lambda s: adstock(s.values))
    return m[["date", "branch_id", "marketing_spend_usd", "promo_active", "mkt_adstock"]]


def build_features(panel, weather, marketing, branches):
    df = panel.sort_values(["branch_id", "service_type", "date"]).reset_index(drop=True).copy()
    series_keys = ["branch_id", "service_type"]

    df["_y"] = df.groupby(series_keys)[TARGET].transform(lambda s: s.interpolate(limit_area="inside"))
    history = df.groupby(series_keys)["_y"]

    for lag in (28, 35, 42, 49, 364):
        df[f"lag_{lag}"] = history.shift(lag)
    df["dow_mean_4w"] = df[["lag_28", "lag_35", "lag_42", "lag_49"]].mean(axis=1)

    shifted = history.shift(MIN_LAG).groupby([df.branch_id, df.service_type])
    df["roll_mean_7_l28"] = shifted.transform(lambda s: s.rolling(7, min_periods=3).mean())
    df["roll_mean_28_l28"] = shifted.transform(lambda s: s.rolling(28, min_periods=7).mean())
    df["roll_std_28_l28"] = shifted.transform(lambda s: s.rolling(28, min_periods=7).std())

    last_year_mean = history.shift(364).groupby([df.branch_id, df.service_type]).transform(
        lambda s: s.rolling(28, min_periods=7).mean())
    df["yoy_ratio"] = (df.roll_mean_28_l28 / last_year_mean).replace([np.inf, -np.inf], np.nan)

    branch_totals = (df.groupby(["branch_id", "date"])["_y"].sum(min_count=1)
                     .rename("_total").reset_index()
                     .sort_values(["branch_id", "date"]))
    branch_totals["branch_roll_28_l28"] = branch_totals.groupby("branch_id")["_total"].transform(
        lambda s: s.shift(MIN_LAG).rolling(28, min_periods=7).mean())
    df = df.merge(branch_totals[["branch_id", "date", "branch_roll_28_l28"]],
                  on=["branch_id", "date"], how="left")

    df = df.merge(build_calendar(df.date.unique()), on="date", how="left")
    df = df.merge(prepare_weather(weather), on=["date", "branch_id"], how="left")
    df = df.merge(prepare_marketing(marketing), on=["date", "branch_id"], how="left")

    branch_info = branches[["branch_id", "climate_zone", "households_k", "avg_housing_age_yrs", "open_date"]]
    df = df.merge(branch_info, on="branch_id", how="left")
    df["days_since_open"] = (df.date - df.open_date).dt.days.clip(lower=0)

    df["baseline"] = df["dow_mean_4w"]
    df["roll_mean_7"] = df["roll_mean_7_l28"]
    df["roll_mean_28"] = df["roll_mean_28_l28"]
    df["roll_std_28"] = df["roll_std_28_l28"]

    cold_weight = df.service_type.astype(str).map(COLD_SERVICE_WEIGHTS).fillna(0)
    df["svc_freeze"] = df.freeze_severity * cold_weight
    df["svc_thaw"] = df.thaw_day * cold_weight

    for col in CAT_FEATURES:
        df[col] = df[col].astype("category")
    return df.drop(columns=["_y", "lag_49", "open_date"])


def full_history_features():
    demand, weather, marketing, branches = load_raw()
    panel = demand[["date", "branch_id", "service_type", TARGET]]
    return build_features(panel, weather, marketing, branches), demand, branches
