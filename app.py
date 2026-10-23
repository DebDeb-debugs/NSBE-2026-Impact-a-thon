"""Carbon-Aware Job Scheduler -- Streamlit app.  Run:  streamlit run app.py"""
import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import scheduler as S

st.set_page_config(page_title="Carbon-Aware Job Scheduler", page_icon="☀️", layout="wide")

EXAMPLE = (
    "Retrain the recommendation model (8 GPUs, 6 hours, due in 40 hours), "
    "nightly database backup (1 hour, due tomorrow 6am), "
    "and batch video encoding (3 hours, due tomorrow noon)"
)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    city = st.selectbox(
        "Data center location",
        ["Los Angeles, CA", "San Francisco, CA", "Phoenix, AZ", "Custom"],
    )
    coords = {
        "Los Angeles, CA": (34.05, -118.24),
        "San Francisco, CA": (37.77, -122.42),
        "Phoenix, AZ": (33.45, -112.07),
    }
    if city == "Custom":
        lat = st.number_input("Latitude", value=34.05)
        lon = st.number_input("Longitude", value=-118.24)
    else:
        lat, lon = coords[city]
    cap = st.slider("Site power cap (kW)", 2, 40, 10, help="Max combined draw of scheduled jobs in any hour.")
    api_key = st.text_input(
        "Anthropic API key (optional)",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="With a key, Claude parses your jobs and explains the schedule. Without one, a simple rule-based parser is used.",
    )
    st.caption(
        f"Assumptions: grid carbon {S.CI_MIN_G_PER_KWH:.0f}-{S.CI_MAX_G_PER_KWH:.0f} g CO₂/kWh scaled by solar output; "
        f"illustrative time-of-use prices (${S.SOLAR_HOURS_PRICE}/${S.OFFPEAK_PRICE}/${S.PEAK_PRICE} per kWh). "
        "Replace with sourced figures before presenting."
    )

st.title("☀️ Carbon-Aware Job Scheduler")
st.write(
    "Describe your compute jobs in plain English. The app finds the sunniest, cleanest hours "
    "before each deadline and shows what you save versus running everything right away."
)

text = st.text_area("Your jobs", value=EXAMPLE, height=110)
go_btn = st.button("Schedule my jobs", type="primary")

if go_btn:
    st.session_state["run"] = True

if st.session_state.get("run"):
    now = datetime.now()
    df, source = S.get_solar_forecast(lat, lon, now)
    curves = S.carbon_and_price_curves(df)

    parser_used = "rule-based parser"
    jobs = None
    if api_key:
        try:
            with st.spinner("Claude is reading your job list..."):
                jobs = S.parse_jobs_with_claude(text, now, api_key)
            parser_used = "Claude"
        except Exception as e:  # keep the demo alive
            st.warning(f"Claude parsing failed ({type(e).__name__}); using the rule-based parser instead.")
    if jobs is None:
        jobs = S.parse_jobs_fallback(text)
    if not jobs:
        st.error("I couldn't find any jobs in that text.")
        st.stop()

    opt_jobs, opt_load, warn_o = S.schedule_jobs(jobs, curves, cap, "optimized")
    nai_jobs, nai_load, warn_n = S.schedule_jobs(jobs, curves, cap, "naive")
    m_opt = S.compute_metrics(opt_jobs, curves)
    m_nai = S.compute_metrics(nai_jobs, curves)

    st.caption(f"Forecast: {source} · Jobs parsed by: {parser_used}")
    for w in dict.fromkeys(warn_o + warn_n):
        st.warning(w)

    # ---------------- headline metrics
    d_co2 = m_nai["co2_kg"] - m_opt["co2_kg"]
    pct = 100 * d_co2 / m_nai["co2_kg"] if m_nai["co2_kg"] else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CO₂ avoided", f"{d_co2:.1f} kg", f"{pct:.0f}% vs run-now")
    c2.metric("Cost saved", f"${m_nai['cost'] - m_opt['cost']:.2f}")
    c3.metric("Energy on solar hours", f"{m_opt['solar_share']*100:.0f}%", f"was {m_nai['solar_share']*100:.0f}%")
    c4.metric("Total energy", f"{m_opt['kwh']:.0f} kWh", "unchanged: jobs are only shifted", delta_color="off")

    # ---------------- charts
    t = curves["time"]
    palette = ["#1f6feb", "#8957e5", "#d29922", "#2da44e", "#cf222e", "#0a7f8c", "#bf3989", "#6e7781"]

    def timeline(job_list, title):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=t, y=curves["carbon_g_per_kwh"], name="Grid carbon (g/kWh)", yaxis="y2",
                line=dict(color="#d29922", width=3), fill="tozeroy", fillcolor="rgba(210,153,34,0.12)",
            )
        )
        for i, j in enumerate(job_list):
            y = [j.power_kw if h in j.schedule else 0 for h in range(len(t))]
            fig.add_trace(go.Bar(x=t, y=y, name=j.name, marker_color=palette[i % len(palette)]))
        fig.update_layout(
            title=title, barmode="stack", height=360, margin=dict(t=50, b=10),
            yaxis=dict(title="Job load (kW)"),
            yaxis2=dict(title="g CO₂/kWh", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=-0.2),
        )
        return fig

    left, right = st.columns(2)
    left.plotly_chart(timeline(nai_jobs, "Run immediately (baseline)"), use_container_width=True)
    right.plotly_chart(timeline(opt_jobs, "Carbon-aware schedule"), use_container_width=True)

    # ---------------- schedule table
    rows = []
    for j in opt_jobs:
        if j.schedule:
            start, end = j.schedule[0], j.schedule[-1] + 1
            when = f"{t.iloc[start]:%a %-I%p} - {t.iloc[min(end, len(t)-1)]:%a %-I%p}"
            if j.splittable and (end - start) != len(j.schedule):
                when += " (split across windows)"
        else:
            when = "not schedulable"
        rows.append(
            {
                "Job": j.name, "Power (kW)": j.power_kw, "Hours": j.duration_h,
                "Deadline": f"{t.iloc[min(j.deadline_h, len(t)) - 1]:%a %-I%p}",
                "Can pause?": "yes" if j.splittable else "no", "Scheduled": when,
            }
        )
    st.subheader("Schedule")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ---------------- Claude explanation
    if api_key:
        try:
            with st.spinner("Writing explanation..."):
                st.subheader("Why these hours?")
                st.markdown(S.explain_with_claude(opt_jobs, curves, api_key))
        except Exception as e:
            st.info(f"Explanation unavailable ({type(e).__name__}).")

    with st.expander("How this works / limitations"):
        st.markdown(
            "- **AI role:** Claude converts free-text into structured jobs and explains the plan. "
            "The placement itself is a transparent greedy algorithm (biggest jobs first, lowest-carbon window before the deadline, under a site power cap).\n"
            "- **Carbon curve is modeled, not measured:** intensity is scaled from forecast solar irradiance. "
            "A production version would use a grid-carbon API (e.g. Electricity Maps or WattTime).\n"
            "- **Single site, simulated:** no real job queue is touched. Future work: Kubernetes/Slurm integration, multi-region shifting, pause/resume."
        )
