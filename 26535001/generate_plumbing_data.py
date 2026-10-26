"""
Synthetic demand dataset for a multi-branch residential plumbing chain
(modelled loosely on franchises like Mr. Rooter Plumbing).

Outputs
  branches.csv            branch master data
  weather_daily.csv       daily weather per branch
  marketing_daily.csv     daily marketing spend / promos per branch
  calendar.csv            date features + US holidays
  daily_demand.csv        MAIN TABLE: branch x service x day (target = calls_received)
  jobs.csv                job-level transactions (one row per completed job)

Demand drivers baked in (so a good model can discover them):
  * trend growth per branch, and one branch opening mid-2024 (cold start)
  * day-of-week pattern (Mon peak, Sun trough; emergencies flatter)
  * annual seasonality that differs by service type
  * freeze events -> burst pipe / leak spikes, with an extra spike on the thaw day
  * heavy rain -> sewer & drain backups (same day + 1-2 day lag)
  * "Brown Friday" (day after Thanksgiving) and Christmas-week drain spike
  * holidays suppress scheduled work, not emergencies
  * marketing spend with a decaying carry-over (adstock) effect, plus promos
  * technician capacity -> completed jobs can be lower than calls (backlog)
  * negative-binomial noise, a few missing weather values, a 3-day data outage
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(".")
START, END = "2023-01-01", "2025-12-31"
dates = pd.date_range(START, END, freq="D")
N = len(dates)
doy = dates.dayofyear.values
t_years = (dates - dates[0]).days.values / 365.25


branches = pd.DataFrame([
    
    ["BR01", "Minneapolis", "MN", "cold",      34, -9,  27, 1.0, "2015-04-01", 410],
    ["BR02", "Chicago",     "IL", "cold",      48, -4,  28, 1.1, "2012-06-15", 620],
    ["BR03", "Denver",      "CO", "cold-dry",  29, -1,  31, 0.6, "2017-03-01", 350],
    ["BR04", "Atlanta",     "GA", "humid",     38,  7,  32, 1.4, "2014-09-10", 480],
    ["BR05", "Dallas",      "TX", "hot",       44,  9,  35, 1.0, "2013-02-20", 560],
    ["BR06", "Phoenix",     "AZ", "desert",    31, 13,  41, 0.3, "2016-11-05", 390],
    ["BR07", "Seattle",     "WA", "wet",       27,  5,  25, 1.6, "2018-05-12", 330],
    ["BR08", "Charlotte",   "NC", "humid",     22,  6,  31, 1.3, "2024-07-01", 260],  
], columns=["branch_id", "city", "state", "climate_zone", "base_daily_calls",
            "winter_temp_c", "summer_temp_c", "rain_factor", "open_date", "households_k"])
branches["open_date"] = pd.to_datetime(branches["open_date"])
branches["annual_growth"] = rng.uniform(0.03, 0.10, len(branches)).round(3)
branches.loc[branches.branch_id == "BR08", "annual_growth"] = 0.0  
branches["technicians_fte"] = (branches.base_daily_calls / 3.2).round().astype(int)
branches["avg_housing_age_yrs"] = rng.integers(22, 55, len(branches))


def thanksgiving(y):
    nov = pd.date_range(f"{y}-11-01", f"{y}-11-30")
    return nov[nov.weekday == 3][3]

def nth_weekday(y, m, wd, n):
    d = pd.date_range(f"{y}-{m:02d}-01", periods=31)
    d = d[(d.month == m) & (d.weekday == wd)]
    return d[n] if n >= 0 else d[-1]

hol = {}
for y in range(2023, 2026):
    hol[pd.Timestamp(f"{y}-01-01")] = "New Year's Day"
    hol[nth_weekday(y, 1, 0, 2)] = "MLK Day"
    hol[nth_weekday(y, 2, 0, 2)] = "Presidents Day"
    hol[nth_weekday(y, 5, 0, -1)] = "Memorial Day"
    hol[pd.Timestamp(f"{y}-07-04")] = "Independence Day"
    hol[nth_weekday(y, 9, 0, 0)] = "Labor Day"
    hol[thanksgiving(y)] = "Thanksgiving"
    hol[pd.Timestamp(f"{y}-12-25")] = "Christmas"

cal = pd.DataFrame({"date": dates})
cal["day_of_week"] = dates.day_name()
cal["dow"] = dates.weekday
cal["month"] = dates.month
cal["year"] = dates.year
cal["week_of_year"] = dates.isocalendar().week.values.astype(int)
cal["is_weekend"] = (cal.dow >= 5).astype(int)
cal["holiday_name"] = cal.date.map(hol).fillna("")
cal["is_holiday"] = (cal.holiday_name != "").astype(int)
tg = {thanksgiving(y) for y in range(2023, 2026)}
cal["is_brown_friday"] = cal.date.isin({d + pd.Timedelta(days=1) for d in tg}).astype(int)
cal["is_christmas_week"] = ((cal.month == 12) & (dates.day >= 22) & (dates.day <= 31)).astype(int)
cal["is_school_holiday"] = (((cal.month == 7) | ((cal.month == 6) & (dates.day > 10)) |
                             ((cal.month == 8) & (dates.day < 20)) | (cal.is_christmas_week == 1))).astype(int)


weather_rows = []
for _, b in branches.iterrows():
    amp = (b.summer_temp_c - b.winter_temp_c) / 2
    mean = (b.summer_temp_c + b.winter_temp_c) / 2
    seasonal = mean - amp * np.cos(2 * np.pi * (doy - 15) / 365.25)
    
    anom = np.zeros(N)
    for i in range(1, N):
        anom[i] = 0.8 * anom[i - 1] + rng.normal(0, 2.6)
    
    for d0 in ["2023-02-02", "2024-01-14", "2025-01-20", "2025-12-18"]:
        idx = dates.get_loc(pd.Timestamp(d0))
        sev = {"cold": 14, "cold-dry": 15, "hot": 13, "humid": 10, "wet": 8, "desert": 4}[b.climate_zone]
        anom[idx:idx + 4] -= sev * np.array([0.7, 1.0, 0.9, 0.5])
    tavg = seasonal + anom
    tmin = tavg - rng.uniform(4, 8, N)
    tmax = tavg + rng.uniform(4, 8, N)
    wet_season = 1 + 0.5 * np.sin(2 * np.pi * (doy - 60) / 365.25)
    if b.climate_zone == "wet":
        wet_season = 1 + 0.8 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    rain_prob = np.clip(0.28 * b.rain_factor * wet_season, 0.02, 0.75)
    rained = rng.random(N) < rain_prob
    precip = np.where(rained, rng.gamma(1.1, 6.5 * b.rain_factor, N), 0.0)
    snow = np.where((tavg < 0) & rained, precip * 1.1 / 10, 0.0)  
    w = pd.DataFrame({
        "date": dates, "branch_id": b.branch_id,
        "temp_avg_c": tavg.round(1), "temp_min_c": tmin.round(1), "temp_max_c": tmax.round(1),
        "precip_mm": precip.round(1), "snow_cm": snow.round(1),
    })
    weather_rows.append(w)
weather = pd.concat(weather_rows, ignore_index=True)
weather["is_freeze"] = (weather.temp_min_c <= -6).astype(int)
weather["is_hard_freeze"] = (weather.temp_min_c <= -12).astype(int)
weather["heavy_rain"] = (weather.precip_mm >= 25).astype(int)
weather["heatwave"] = (weather.temp_max_c >= 38).astype(int)

gap = rng.random(len(weather)) < 0.006
weather.loc[gap, ["temp_avg_c", "temp_min_c", "temp_max_c"]] = np.nan
gap2 = rng.random(len(weather)) < 0.004
weather.loc[gap2, "precip_mm"] = np.nan


mk_rows = []
for _, b in branches.iterrows():
    base_spend = b.base_daily_calls * 9
    spend = base_spend * (1 + 0.25 * np.sin(2 * np.pi * (doy - 80) / 365.25)) * rng.lognormal(0, 0.25, N)
    spend[dates.weekday >= 5] *= 0.6
    
    promo = np.zeros(N, dtype=int)
    for _ in range(9):
        s = rng.integers(0, N - 14)
        L = rng.integers(5, 15)
        promo[s:s + L] = 1
        spend[s:s + L] *= 1.8
    
    for y in range(2023, 2026):
        m = (dates >= f"{y}-10-01") & (dates <= f"{y}-10-21")
        promo[m] = 1
    closed = dates < b.open_date
    spend[closed] = 0
    promo[closed] = 0
    mk_rows.append(pd.DataFrame({"date": dates, "branch_id": b.branch_id,
                                 "marketing_spend_usd": spend.round(2), "promo_active": promo}))
marketing = pd.concat(mk_rows, ignore_index=True)


services = pd.DataFrame([
    
    ["drain_cleaning",      "drain",     0.27, 0.35, 260,  1.3],
    ["leak_repair",         "repair",    0.18, 0.45, 340,  1.6],
    ["water_heater",        "install",   0.14, 0.40, 1450, 3.5],
    ["toilet_fixture",      "repair",    0.13, 0.15, 290,  1.2],
    ["sewer_line",          "sewer",     0.08, 0.55, 2400, 4.5],
    ["burst_frozen_pipe",   "emergency", 0.05, 0.95, 780,  2.4],
    ["sump_pump",           "install",   0.05, 0.50, 950,  2.2],
    ["inspection_camera",   "inspect",   0.06, 0.05, 220,  1.0],
    ["repipe_remodel",      "install",   0.04, 0.02, 5200, 8.0],
], columns=["service_type", "category", "base_share", "emergency_rate", "avg_ticket_usd", "avg_duration_h"])

def annual_profile(svc, climate):
    x = 2 * np.pi * (doy - 15) / 365.25
    winter = np.cos(x)  
    cold = climate in ("cold", "cold-dry")
    p = {
        "drain_cleaning":    1 + 0.08 * winter,
        "leak_repair":       1 + (0.15 if cold else 0.06) * winter,
        "water_heater":      1 + 0.30 * winter,
        "toilet_fixture":    1 + 0.05 * np.sin(x),
        "sewer_line":        1 + 0.15 * np.sin(2 * np.pi * (doy - 80) / 365.25),
        "burst_frozen_pipe": 0.4 + (1.6 if cold else 0.6) * np.clip(winter, 0, None) ** 2,
        "sump_pump":         1 + 0.35 * np.sin(2 * np.pi * (doy - 60) / 365.25),
        "inspection_camera": 1 + 0.20 * np.sin(2 * np.pi * (doy - 90) / 365.25),
        "repipe_remodel":    1 - 0.25 * winter,
    }[svc]
    return np.clip(p, 0.05, None)

DOW = {  
    "scheduled": np.array([1.28, 1.12, 1.06, 1.03, 0.98, 0.42, 0.22]),
    "emergency": np.array([1.10, 1.02, 1.00, 1.00, 1.02, 0.90, 0.86]),
}

def adstock(x, decay=0.55):
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        out[i] = x[i] + (decay * out[i - 1] if i else 0)
    return out

rows = []
for _, b in branches.iterrows():
    w = weather[weather.branch_id == b.branch_id].reset_index(drop=True)
    m = marketing[marketing.branch_id == b.branch_id].reset_index(drop=True)
    tmin = w.temp_min_c.interpolate().bfill().values
    precip = w.precip_mm.fillna(0).values
    open_mask = dates >= b.open_date

    
    if b.branch_id == "BR08":
        days_open = np.clip((dates - b.open_date).days.values, 0, None)
        level = 1 / (1 + np.exp(-(days_open - 120) / 45))
    else:
        level = (1 + b.annual_growth) ** t_years
    
    level = level * np.where((dates >= "2024-02-01") & (dates <= "2024-04-30"), 0.93, 1.0)

    ad = adstock(m.marketing_spend_usd.values)
    ad_norm = ad / (np.median(ad[ad > 0]) if (ad > 0).any() else 1)
    mkt_mult = 1 + 0.10 * np.log1p(ad_norm) + 0.06 * m.promo_active.values

    freeze_sev = np.clip(-6 - tmin, 0, None)  
    prev_cold = np.roll(tmin, 1) <= -8
    thaw = (prev_cold & (tmin > -2)).astype(float)
    thaw[0] = 0
    rain_lag = precip + 0.6 * np.roll(precip, 1) + 0.3 * np.roll(precip, 2)

    for _, s in services.iterrows():
        base = b.base_daily_calls * s.base_share
        ann = annual_profile(s.service_type, b.climate_zone)
        dow_s = DOW["scheduled"][dates.weekday]
        dow_e = DOW["emergency"][dates.weekday]
        dow = s.emergency_rate * dow_e + (1 - s.emergency_rate) * dow_s

        holiday = cal.is_holiday.values
        hol_mult = np.where(holiday == 1, s.emergency_rate * 0.95 + (1 - s.emergency_rate) * 0.15, 1.0)

        event = np.ones(N)
        if s.service_type == "burst_frozen_pipe":
            event += 0.9 * freeze_sev + 6 * thaw
        if s.service_type == "leak_repair":
            event += 0.12 * freeze_sev + 1.5 * thaw
        if s.service_type == "water_heater":
            event += 0.05 * freeze_sev
        if s.service_type in ("sewer_line", "sump_pump"):
            event += 0.035 * rain_lag
        if s.service_type == "drain_cleaning":
            event += 0.01 * rain_lag
            event += 1.6 * cal.is_brown_friday.values + 0.35 * cal.is_christmas_week.values
        if s.service_type == "repipe_remodel":
            event *= 1 + 0.15 * (b.avg_housing_age_yrs > 40)
        if s.service_type in ("toilet_fixture", "drain_cleaning"):
            event *= 1 + 0.05 * cal.is_school_holiday.values

        mkt = mkt_mult if s.emergency_rate < 0.6 else 1 + 0.4 * (mkt_mult - 1)
        lam = base * ann * dow * hol_mult * event * level * mkt
        lam = np.where(open_mask, lam, 0)

        k = 6.0  
        p = k / (k + lam + 1e-9)
        calls = rng.negative_binomial(k, p)
        calls = np.where(open_mask, calls, 0)

        rows.append(pd.DataFrame({
            "date": dates, "branch_id": b.branch_id, "service_type": s.service_type,
            "service_category": s.category, "calls_received": calls,
            "_emerg_rate": s.emergency_rate, "_ticket": s.avg_ticket_usd, "_dur": s.avg_duration_h,
        }))

demand = pd.concat(rows, ignore_index=True)


cap_rows = []
for _, b in branches.iterrows():
    techs = np.full(N, b.technicians_fte, dtype=float)
    if b.branch_id != "BR08":
        techs = techs * (1 + b.annual_growth * 0.8) ** t_years
    else:
        days_open = np.clip((dates - b.open_date).days.values, 0, None)
        techs = 3 + (b.technicians_fte - 3) * np.clip(days_open / 240, 0, 1)
    techs = techs * np.where(dates.weekday == 5, 0.55, np.where(dates.weekday == 6, 0.35, 1.0))
    techs = techs * np.where(cal.is_holiday.values == 1, 0.35, 1.0)
    techs = techs * np.where(cal.is_christmas_week.values == 1, 0.8, 1.0)
    techs = np.maximum(0, np.round(techs - rng.binomial(2, 0.15, N)))  
    techs = np.where(dates >= b.open_date, techs, 0)
    cap_rows.append(pd.DataFrame({"date": dates, "branch_id": b.branch_id, "technicians_on_shift": techs.astype(int)}))
cap = pd.concat(cap_rows, ignore_index=True)
demand = demand.merge(cap, on=["date", "branch_id"])


demand["_hours_needed"] = demand.calls_received * demand._dur
tot = demand.groupby(["date", "branch_id"])["_hours_needed"].transform("sum")
avail_h = demand.technicians_on_shift * 7.5
ratio = np.clip(avail_h / tot.replace(0, np.nan), 0, 1).fillna(1)
emerg = demand.calls_received * demand._emerg_rate
sched = demand.calls_received - emerg
fill_e = np.clip(ratio * 1.4, 0, 1)
fill_s = np.clip(ratio * 0.9, 0, 1)
demand["jobs_completed"] = np.round(emerg * fill_e + sched * fill_s).astype(int).clip(upper=demand.calls_received)
demand["emergency_calls"] = rng.binomial(demand.calls_received, demand._emerg_rate)
demand["jobs_deferred"] = demand.calls_received - demand.jobs_completed
demand["cancellations"] = rng.binomial(demand.jobs_completed, 0.06)
demand["jobs_completed"] -= demand.cancellations
price_infl = (1.035 ** t_years)
pi = pd.Series(price_infl, index=dates)
demand["avg_ticket_usd"] = (demand._ticket * demand.date.map(pi) *
                            rng.lognormal(0, 0.12, len(demand))).round(2)
demand.loc[demand.jobs_completed == 0, "avg_ticket_usd"] = np.nan
demand["revenue_usd"] = (demand.jobs_completed * demand.avg_ticket_usd).fillna(0).round(2)


outage = (demand.branch_id == "BR04") & demand.date.between("2025-03-10", "2025-03-12")
demand.loc[outage, ["calls_received", "emergency_calls"]] = np.nan

demand = demand.drop(columns=[c for c in demand.columns if c.startswith("_")])
demand = demand[demand.date >= demand.branch_id.map(branches.set_index("branch_id").open_date)]  
demand = demand[["date", "branch_id", "service_type", "service_category", "calls_received",
                 "emergency_calls", "jobs_completed", "jobs_deferred", "cancellations",
                 "technicians_on_shift", "avg_ticket_usd", "revenue_usd"]]


svc_idx = services.set_index("service_type")
jb = demand[demand.jobs_completed > 0][["date", "branch_id", "service_type", "jobs_completed", "avg_ticket_usd"]]
rep = jb.loc[jb.index.repeat(jb.jobs_completed)].reset_index(drop=True)
n = len(rep)
rep["job_id"] = [f"J{i:07d}" for i in range(1, n + 1)]
er = rep.service_type.map(svc_idx.emergency_rate).values
rep["is_emergency"] = (rng.random(n) < er).astype(int)
hours = np.where(rep.is_emergency == 1,
                 rng.choice(np.arange(24), n, p=np.r_[np.full(6, .015), np.full(6, .05), np.full(6, .06), np.full(6, .041)] / 1.0 / np.r_[np.full(6, .015), np.full(6, .05), np.full(6, .06), np.full(6, .041)].sum()),
                 rng.choice(np.arange(8, 18), n))
rep["request_ts"] = rep.date + pd.to_timedelta(hours, "h") + pd.to_timedelta(rng.integers(0, 60, n), "m")
lead = np.where(rep.is_emergency == 1, rng.gamma(2, 1.2, n), rng.gamma(2, 14, n))  
rep["response_hours"] = lead.round(2)
dur = rep.service_type.map(svc_idx.avg_duration_h).values
rep["duration_hours"] = np.clip(rng.lognormal(np.log(dur), 0.35), 0.25, None).round(2)
rep["invoice_usd"] = (rep.avg_ticket_usd * rng.lognormal(0, 0.3, n) * np.where(rep.is_emergency == 1, 1.25, 1)).round(2)
rep["customer_type"] = rng.choice(["residential", "commercial", "property_mgmt"], n, p=[0.78, 0.12, 0.10])
rep["booking_channel"] = rng.choice(["phone", "website", "app", "google_lsa", "referral"], n, p=[0.46, 0.22, 0.08, 0.16, 0.08])
rep["repeat_customer"] = (rng.random(n) < 0.31).astype(int)
techs_per_branch = branches.set_index("branch_id").technicians_fte * 1.4
rep["technician_id"] = [f"{b}-T{rng.integers(1, int(techs_per_branch[b]) + 1):02d}" for b in rep.branch_id]
rep["customer_rating"] = np.where(rng.random(n) < 0.42, rng.choice([1, 2, 3, 4, 5], n, p=[.03, .04, .09, .27, .57]), np.nan)
jobs = rep[["job_id", "branch_id", "date", "request_ts", "service_type", "is_emergency", "customer_type",
            "booking_channel", "repeat_customer", "technician_id", "response_hours", "duration_hours",
            "invoice_usd", "customer_rating"]].rename(columns={"date": "job_date"})


branches_out = branches.drop(columns=["base_daily_calls", "winter_temp_c", "summer_temp_c", "rain_factor", "annual_growth"])
branches_out.to_csv(OUT / "branches.csv", index=False)
weather.to_csv(OUT / "weather_daily.csv", index=False)
marketing.to_csv(OUT / "marketing_daily.csv", index=False)
cal.drop(columns=["dow"]).to_csv(OUT / "calendar.csv", index=False)
demand.to_csv(OUT / "daily_demand.csv", index=False)
jobs.to_csv(OUT / "jobs.csv", index=False)

print("daily_demand:", demand.shape, " jobs:", jobs.shape, " weather:", weather.shape)
print(demand.groupby("branch_id").calls_received.sum().astype(int))
