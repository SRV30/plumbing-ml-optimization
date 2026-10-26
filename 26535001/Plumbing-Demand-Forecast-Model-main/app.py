import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from features import load_raw
from forecast import make_forecast, staffing_plan, agg_interval, load_bundle as _load_bundle, Scenario

st.set_page_config(page_title="Plumbing Demand Forecaster", page_icon="🔧", layout="wide")

ROOT = Path(__file__).parent
SERVICE_LABELS = {
    "drain_cleaning": "Drain cleaning", "leak_repair": "Leak repair", "water_heater": "Water heater",
    "toilet_fixture": "Toilet / fixture", "sewer_line": "Sewer line", "burst_frozen_pipe": "Burst / frozen pipe",
    "sump_pump": "Sump pump", "inspection_camera": "Camera inspection", "repipe_remodel": "Repipe / remodel",
}
ACCENT, BAND, ACTUAL = "#1f6feb", "rgba(31,111,235,0.18)", "#6e7781"

st.markdown("""
<style>
.block-container {padding-top: 3.2rem;}
div[data-testid="stMetric"] {background: rgba(127,127,127,0.07); border-radius: 10px; padding: 10px 14px;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model…")
def load_bundle():
    if not (ROOT / "models" / "meta.json").exists():
        st.error("Model not found. Run `python train.py` first.")
        st.stop()
    return _load_bundle(ROOT / "models")


@st.cache_resource(show_spinner="Loading data…")
def load_data():
    return load_raw()


bundle = load_bundle()
demand, weather, marketing, branches = load_data()
branch_name = dict(zip(branches.branch_id, branches.branch_id + " · " + branches.city))
all_branches = list(branches.branch_id)
history_end = demand.date.max()


@st.cache_data(show_spinner="Forecasting…")
def run_forecast(branch_ids, horizon, settings):
    scenario = Scenario(*settings[:6], promo_days=list(settings[6]), marketing_mult=settings[7])
    return make_forecast(bundle, list(branch_ids), horizon, scenario, raw=(demand, weather, marketing, branches))


with st.sidebar:
    st.title("🔧 Demand Forecaster")
    st.caption(f"History: {demand.date.min():%d %b %Y} – {history_end:%d %b %Y}")
    selected = st.multiselect("Branches", all_branches, default=["BR01", "BR02"],
                         format_func=lambda b: branch_name[b])
    if not selected:
        st.warning("Pick at least one branch.")
        st.stop()
    horizon = st.slider("Forecast horizon (days)", 7, 28, 14)

    st.subheader("What-if scenario")
    temp_shift = st.slider("Temperature vs normal (°C)", -10.0, 10.0, 0.0, 0.5)
    snap = st.checkbox("Cold snap")
    snap_start, snap_days, snap_depth = None, 3, 15.0
    if snap:
        snap_start = st.slider("Cold snap starts on day", 1, horizon, min(5, horizon)) - 1
        snap_days = st.slider("Cold snap length (days)", 1, 7, 3)
        snap_depth = st.slider("How much colder (°C)", 5.0, 25.0, 15.0, 1.0)
    storm = st.checkbox("Heavy rainstorm")
    storm_day, storm_mm = None, 50.0
    if storm:
        storm_day = st.slider("Storm on day", 1, horizon, min(3, horizon)) - 1
        storm_mm = st.slider("Rainfall (mm)", 20, 120, 50, 5)
    promo = st.checkbox("Run a promotion")
    promo_days = ()
    if promo:
        first, last = st.slider("Promo days", 1, horizon, (1, min(7, horizon)))
        promo_days = tuple(range(first - 1, last))
    mkt = st.slider("Marketing budget vs usual", 0.5, 2.0, 1.0, 0.1, format="%.1fx")

scenario_settings = (temp_shift, snap_start, snap_days, snap_depth, storm_day, storm_mm, promo_days, mkt)
normal_settings = (0.0, None, 3, 15.0, None, 50.0, (), 1.0)
fc = run_forecast(tuple(selected), horizon, scenario_settings)
fc_base = run_forecast(tuple(selected), horizon, normal_settings)
scenario_on = scenario_settings != normal_settings

tab_fc, tab_staff, tab_hist, tab_perf = st.tabs(
    ["🔮 Forecast", "👷 Staffing plan", "📊 Demand history", "✅ Model performance"])

with tab_fc:
    st.header(f"Next {horizon} days: {fc.date.min():%d %b} – {fc.date.max():%d %b %Y}")
    total, total_normal = fc.forecast.sum(), fc_base.forecast.sum()
    previous = demand[(demand.branch_id.isin(selected)) & (demand.date > history_end - pd.Timedelta(days=horizon))]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected service calls", f"{total:,.0f}",
              f"{100*(total/total_normal-1):+.1f}% vs normal" if scenario_on else None)
    total_range = agg_interval(fc.assign(_all=1), ["_all"]).iloc[0]
    c2.metric("Likely range (P10–P90)", f"{total_range.p10:,.0f} – {total_range.p90:,.0f}")
    c3.metric(f"Same {horizon} days before", f"{previous.calls_received.sum():,.0f}",
              f"{100*(total/max(previous.calls_received.sum(),1)-1):+.1f}% change")
    c4.metric("Expected revenue", f"${fc.exp_revenue_usd.sum()/1e3:,.0f}k")

    daily = agg_interval(fc, ["date"])
    hist = demand[(demand.branch_id.isin(selected)) & (demand.date > history_end - pd.Timedelta(days=60))]
    hist = hist.groupby("date").calls_received.sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.date, y=hist.calls_received, name="Actual", line=dict(color=ACTUAL)))
    fig.add_trace(go.Scatter(x=pd.concat([daily.date, daily.date[::-1]]),
                             y=pd.concat([daily.p90, daily.p10[::-1]]), fill="toself", fillcolor=BAND,
                             line=dict(width=0), name="P10–P90 range", hoverinfo="skip"))
    if scenario_on:
        normal_daily = fc_base.groupby("date").forecast.sum()
        fig.add_trace(go.Scatter(x=normal_daily.index, y=normal_daily.values, name="Normal conditions",
                                 line=dict(color=ACCENT, dash="dot", width=1.5)))
    fig.add_trace(go.Scatter(x=daily.date, y=daily.forecast, name="Forecast" + (" (scenario)" if scenario_on else ""),
                             line=dict(color=ACCENT, width=3)))
    fig.add_vline(x=history_end, line_dash="dash", line_color="gray")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      yaxis_title="Calls per day", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("By service type")
        bs = fc.groupby("service_type").agg(Forecast=("forecast", "sum")).reset_index()
        bs["Normal"] = bs.service_type.map(fc_base.groupby("service_type").forecast.sum())
        bs["Service"] = bs.service_type.map(SERVICE_LABELS)
        bs = bs.sort_values("Forecast")
        f2 = px.bar(bs, x="Forecast", y="Service", orientation="h", color_discrete_sequence=[ACCENT])
        if scenario_on:
            f2.add_trace(go.Scatter(x=bs.Normal, y=bs.Service, mode="markers", name="Normal",
                                    marker=dict(color="black", symbol="line-ns-open", size=16)))
        f2.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="", xaxis_title="Calls")
        st.plotly_chart(f2, width="stretch")
    with right:
        st.subheader("By branch")
        bb = agg_interval(fc, ["branch_id"]).rename(columns={"forecast": "calls", "p10": "low", "p90": "high",
                                                             "exp_revenue_usd": "revenue"})
        bb["Branch"] = bb.branch_id.map(branch_name)
        st.dataframe(bb[["Branch", "calls", "low", "high", "revenue"]], hide_index=True, width="stretch",
                     column_config={"calls": st.column_config.NumberColumn("Calls", format="%.0f"),
                                    "low": st.column_config.NumberColumn("Low (P10)", format="%.0f"),
                                    "high": st.column_config.NumberColumn("High (P90)", format="%.0f"),
                                    "revenue": st.column_config.NumberColumn("Revenue", format="dollar")})
        heat = fc.pivot_table(index="service_type", columns="date", values="forecast", aggfunc="sum")
        heat.index = heat.index.map(SERVICE_LABELS)
        f3 = px.imshow(heat, aspect="auto", color_continuous_scale="Blues",
                       labels=dict(color="Calls", x="", y=""))
        f3.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
        f3.update_xaxes(tickformat="%d %b")
        st.plotly_chart(f3, width="stretch")

    out = fc[["date", "branch_id", "service_type", "forecast", "p10", "p90", "tech_hours", "exp_revenue_usd"]].round(2)
    st.download_button("⬇️ Download forecast (CSV)", out.to_csv(index=False), "forecast.csv", "text/csv")

with tab_staff:
    st.header("Technicians needed")
    st.caption("Hours of work = forecast calls × average job duration per service; one technician ≈ 7.5 hours. "
               "'Usual roster' is the median staff on shift for that weekday over the last 8 weeks.")
    plan = staffing_plan(fc, demand)
    staff_branch = st.selectbox("Branch", selected, format_func=lambda x: branch_name[x], key="staff_branch")
    s = plan[plan.branch_id == staff_branch]
    short_days = int((s.gap_vs_roster > 0).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg technicians needed / day", f"{s.techs_needed.mean():.1f}")
    c2.metric("Usual roster / day", f"{s.usual_roster.mean():.1f}")
    c3.metric("Days short-staffed", f"{short_days} of {len(s)}")
    f4 = go.Figure()
    f4.add_trace(go.Bar(x=s.date, y=s.techs_needed, name="Needed (expected)", marker_color=ACCENT))
    f4.add_trace(go.Scatter(x=s.date, y=s.techs_needed_p90, name="Needed on a busy day (P90)",
                            mode="markers", marker=dict(color=ACCENT, symbol="triangle-up", size=9)))
    f4.add_trace(go.Scatter(x=s.date, y=s.usual_roster, name="Usual roster", mode="lines+markers",
                            line=dict(color="#d1242f", width=2)))
    f4.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Technicians",
                     hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(f4, width="stretch")
    tbl = s.assign(Day=s.date.dt.strftime("%a %d %b"))[
        ["Day", "forecast", "techs_needed", "techs_needed_p90", "usual_roster", "gap_vs_roster"]]
    st.dataframe(tbl, hide_index=True, width="stretch",
                 column_config={"forecast": st.column_config.NumberColumn("Calls", format="%.0f"),
                                "techs_needed": "Techs needed", "techs_needed_p90": "Busy-day techs",
                                "usual_roster": st.column_config.NumberColumn("Usual roster", format="%.0f"),
                                "gap_vs_roster": st.column_config.NumberColumn("Gap (+ = hire/overtime)",
                                                                               format="%+.0f")})

with tab_hist:
    st.header("Historical demand")
    c1, c2 = st.columns([2, 1])
    svc = c1.multiselect("Services", list(SERVICE_LABELS), default=list(SERVICE_LABELS),
                         format_func=SERVICE_LABELS.get)
    freq = c2.radio("Group by", ["Day", "Week", "Month"], index=1, horizontal=True)
    h = demand[demand.branch_id.isin(selected) & demand.service_type.isin(svc)]
    rule = {"Day": "D", "Week": "W", "Month": "MS"}[freq]
    ts = (h.groupby(["branch_id", pd.Grouper(key="date", freq=rule)]).calls_received.sum().reset_index())
    ts["Branch"] = ts.branch_id.map(branch_name)
    f5 = px.line(ts, x="date", y="calls_received", color="Branch")
    f5.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=f"Calls per {freq.lower()}",
                     xaxis_title="", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(f5, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Weekly pattern")
        dw = h.assign(Day=h.date.dt.day_name().str[:3], d=h.date.dt.weekday).groupby(
            ["d", "Day", "date"]).calls_received.sum().groupby(["d", "Day"]).mean().reset_index()
        st.plotly_chart(px.bar(dw, x="Day", y="calls_received", color_discrete_sequence=[ACCENT])
                        .update_layout(height=300, yaxis_title="Avg calls / day", xaxis_title="",
                                       margin=dict(l=10, r=10, t=10, b=10)), width="stretch")
    with c2:
        st.subheader("Seasonality by service")
        sm = h.assign(Month=h.date.dt.month).groupby(["service_type", "Month", "date"]).calls_received.sum() \
            .groupby(["service_type", "Month"]).mean().reset_index()
        sm["idx"] = sm.calls_received / sm.groupby("service_type").calls_received.transform("mean")
        pv = sm.pivot(index="service_type", columns="Month", values="idx")
        pv.index = pv.index.map(SERVICE_LABELS)
        pv.columns = [pd.Timestamp(2025, m, 1).strftime("%b") for m in pv.columns]
        st.plotly_chart(px.imshow(pv, aspect="auto", color_continuous_scale="RdBu_r", zmin=0, zmax=2,
                                  labels=dict(color="vs avg")).update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10)), width="stretch")

    st.subheader("Weather drives demand")
    wj = h.groupby(["date", "branch_id"]).calls_received.sum().reset_index().merge(weather, on=["date", "branch_id"])
    wj["Min temp band"] = pd.cut(wj.temp_min_c, [-40, -15, -6, 0, 10, 20, 40],
                                 labels=["< −15°", "−15 to −6°", "−6 to 0°", "0 to 10°", "10 to 20°", "> 20°"])
    wb = wj.groupby(["Min temp band", "branch_id"], observed=True).calls_received.mean().reset_index()
    wb["Branch"] = wb.branch_id.map(branch_name)
    st.plotly_chart(px.bar(wb, x="Min temp band", y="calls_received", color="Branch", barmode="group")
                    .update_layout(height=320, yaxis_title="Avg calls / day", margin=dict(l=10, r=10, t=10, b=10)),
                    width="stretch")

with tab_perf:
    st.header("How accurate is the model?")
    st.caption(f"Trained on data up to {pd.Timestamp(bundle['val_start']) - pd.Timedelta(days=1):%d %b %Y}, "
               f"tested on {bundle['val_start']} – {history_end:%Y-%m-%d} (data the model never saw). "
               "WAPE = total absolute error ÷ total actual calls (lower is better).")
    mb = bundle["metrics_branch_day"]
    lg, nv = mb["LightGBM (this model)"]["WAPE_%"], mb["Naive: avg of last 4 same weekdays"]["WAPE_%"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("WAPE, branch-level daily", f"{lg:.1f}%")
    c2.metric("Best naive baseline", f"{nv:.1f}%", f"{100*(1-lg/nv):.0f}% less error", delta_color="normal")
    c3.metric("Bias", f"{mb['LightGBM (this model)']['Bias_%']:+.1f}%")
    c4.metric("P10–P90 coverage", f"{100*bundle['interval_coverage']:.0f}%", "target 80%", delta_color="off")

    st.subheader("Model vs baselines (branch × service × day)")
    st.dataframe(pd.DataFrame(bundle["metrics"]).T, width="stretch")
    st.caption("Per-service daily counts are small (often 0–5), so much of the remaining error is pure randomness; "
               "accuracy improves a lot once totals are aggregated per branch.")

    va = bundle["validation"]
    c1, c2 = st.columns(2)
    vb = c1.selectbox("Branch", all_branches, format_func=branch_name.get, key="vb")
    vs = c2.selectbox("Service", ["All services"] + list(SERVICE_LABELS),
                      format_func=lambda s: SERVICE_LABELS.get(s, s), key="vs")
    v = va[va.branch_id == vb]
    if vs != "All services":
        v = v[v.service_type == vs]
    v = agg_interval(v.rename(columns={"pred": "forecast"}).assign(tech_hours=0, exp_revenue_usd=0), ["date"]) \
        .merge(v.groupby("date").calls_received.sum().reset_index()).rename(columns={"forecast": "pred"})
    f6 = go.Figure()
    f6.add_trace(go.Scatter(x=pd.concat([v.date, v.date[::-1]]), y=pd.concat([v.p90, v.p10[::-1]]),
                            fill="toself", fillcolor=BAND, line=dict(width=0), name="P10–P90", hoverinfo="skip"))
    f6.add_trace(go.Scatter(x=v.date, y=v.calls_received, name="Actual", line=dict(color=ACTUAL)))
    f6.add_trace(go.Scatter(x=v.date, y=v.pred, name="Model", line=dict(color=ACCENT, width=2)))
    f6.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                     legend=dict(orientation="h", y=1.1), yaxis_title="Calls per day")
    st.plotly_chart(f6, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Error by branch")
        eb = pd.Series(bundle["wape_by_branch"]).rename("WAPE %").reset_index()
        eb["Branch"] = eb["index"].map(branch_name)
        st.plotly_chart(px.bar(eb, x="WAPE %", y="Branch", orientation="h", color_discrete_sequence=[ACCENT])
                        .update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=""),
                        width="stretch")
    with c2:
        st.subheader("What the model relies on")
        fi = pd.Series(bundle["feature_importance"]).head(12).sort_values().rename("Importance %").reset_index()
        st.plotly_chart(px.bar(fi, x="Importance %", y="index", orientation="h", color_discrete_sequence=[ACCENT])
                        .update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=""),
                        width="stretch")
