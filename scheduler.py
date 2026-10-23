"""Core logic for the Carbon-Aware Job Scheduler (no UI code here).

Pieces:
  - get_solar_forecast: hourly irradiance from Open-Meteo (falls back to a synthetic clear-sky curve)
  - carbon_and_price_curves: turns irradiance into an *assumed* grid carbon intensity and TOU price
  - parse_jobs_with_claude / parse_jobs_fallback: plain English -> structured jobs
  - schedule_jobs: greedy placement under a site power cap
  - compute_metrics: kWh, CO2, cost, solar-hour share
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

HORIZON_HOURS = 48

# ---------------------------------------------------------------- assumptions
# Illustrative assumptions -- label these on your slides and swap in sourced figures.
CI_MAX_G_PER_KWH = 450.0  # grid carbon intensity with no solar (evening, gas on the margin)
CI_MIN_G_PER_KWH = 120.0  # grid carbon intensity at peak solar
PEAK_PRICE = 0.55  # $/kWh, 4-9pm
OFFPEAK_PRICE = 0.32  # $/kWh, most hours
SOLAR_HOURS_PRICE = 0.26  # $/kWh, 9am-4pm
SOLAR_HOUR_THRESHOLD = 0.35  # normalized irradiance above this counts as a "solar hour"


@dataclass
class Job:
    name: str
    power_kw: float
    duration_h: int
    deadline_h: int  # hours from start of horizon (exclusive end of the allowed window)
    splittable: bool = False
    schedule: list[int] = field(default_factory=list)  # hour indices where it runs


# ---------------------------------------------------------------- data
def get_solar_forecast(lat: float, lon: float, start: datetime | None = None):
    """Return (df, source). df has columns: time, irradiance (W/m2), cloud, solar_norm (0-1)."""
    start = (start or datetime.now()).replace(minute=0, second=0, microsecond=0)
    try:
        import requests

        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "shortwave_radiation,cloud_cover",
                "forecast_days": 3,
                "timezone": "auto",
            },
            timeout=8,
        )
        r.raise_for_status()
        h = r.json()["hourly"]
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(h["time"]),
                "irradiance": h["shortwave_radiation"],
                "cloud": h["cloud_cover"],
            }
        )
        df = df[df["time"] >= start].head(HORIZON_HOURS).reset_index(drop=True)
        if len(df) < HORIZON_HOURS:
            raise ValueError("short forecast")
        source = "Open-Meteo live forecast"
    except Exception:
        times = [start + timedelta(hours=i) for i in range(HORIZON_HOURS)]
        irr, cloud = [], []
        rng = np.random.default_rng(7)
        for i, t in enumerate(times):
            x = (t.hour + t.minute / 60 - 6.5) / 12.0  # sunrise ~6:30, sunset ~18:30
            clear = max(0.0, math.sin(math.pi * x)) * 850 if 0 <= x <= 1 else 0.0
            c = 15 if i < 24 else 45
            cloud.append(c + rng.integers(-5, 6))
            irr.append(clear * (1 - 0.6 * cloud[-1] / 100))
        df = pd.DataFrame({"time": times, "irradiance": irr, "cloud": cloud})
        source = "Synthetic clear-sky curve (offline fallback)"
    peak = max(df["irradiance"].max(), 1.0)
    df["solar_norm"] = (df["irradiance"] / max(peak, 600.0)).clip(0, 1)
    return df, source


def carbon_and_price_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Add assumed carbon intensity (g/kWh) and TOU price ($/kWh) columns."""
    out = df.copy()
    out["carbon_g_per_kwh"] = CI_MAX_G_PER_KWH - (CI_MAX_G_PER_KWH - CI_MIN_G_PER_KWH) * out["solar_norm"]

    def price(t):
        if 16 <= t.hour < 21:
            return PEAK_PRICE
        if 9 <= t.hour < 16:
            return SOLAR_HOURS_PRICE
        return OFFPEAK_PRICE

    out["price_per_kwh"] = out["time"].apply(price)
    return out


# ---------------------------------------------------------------- parsing
CLAUDE_SYSTEM = """You convert plain-English descriptions of compute jobs into JSON.
Return ONLY a JSON array. Each element has:
  name (string), power_kw (number: average electrical draw), duration_h (integer hours, >=1),
  deadline_hours_from_now (integer, hours from the current time by which the job must FINISH),
  splittable (boolean: true if it can be paused/resumed, e.g. model training with checkpoints, batch jobs).
Typical draws if unspecified: one GPU ~0.7 kW, 8-GPU node ~6 kW, backup/ETL ~1 kW, video encoding ~2 kW.
If no deadline is given use 24. Never exceed 47 for deadline_hours_from_now."""


def parse_jobs_with_claude(text: str, now: datetime, api_key: str, model: str = "claude-sonnet-4-5"):
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=1500,
        system=CLAUDE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Current local time: {now.strftime('%A %Y-%m-%d %H:%M')}\n\nJobs:\n{text}",
            }
        ],
    )
    raw = "".join(b.text for b in msg.content if getattr(b, "text", None))
    raw = raw[raw.find("[") : raw.rfind("]") + 1]
    return _jobs_from_json(json.loads(raw))


def _jobs_from_json(items) -> list[Job]:
    jobs = []
    for it in items:
        jobs.append(
            Job(
                name=str(it["name"]),
                power_kw=float(it["power_kw"]),
                duration_h=max(1, int(round(float(it["duration_h"])))),
                deadline_h=max(1, min(HORIZON_HOURS, int(it.get("deadline_hours_from_now", 24)))),
                splittable=bool(it.get("splittable", False)),
            )
        )
    return jobs


def parse_jobs_fallback(text: str) -> list[Job]:
    """Simple rule-based parser used when no API key is set. One job per line / comma / 'and'."""
    parts = [p.strip() for p in re.split(r"[\n;]|,(?![^()]*\))| and ", text) if p.strip()]
    jobs = []
    for p in parts:
        low = p.lower()
        dur = 2
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:gpu[- ]?)?(?:hours?|hrs?|h)\b", low)
        if m:
            dur = max(1, int(round(float(m.group(1)))))
        power = 0.7
        g = re.search(r"(\d+)\s*gpus?", low)
        if g:
            power = 0.7 * int(g.group(1))
        elif any(k in low for k in ("train", "fine-tun", "finetun")):
            power = 4.0
        elif "encod" in low or "video" in low or "render" in low:
            power = 2.0
        elif "backup" in low or "etl" in low or "index" in low:
            power = 1.0
        deadline = 24
        d = re.search(r"in (\d+)\s*hours?", low)
        if d:
            deadline = int(d.group(1))
        elif "tomorrow" in low:
            deadline = 30
        elif "tonight" in low:
            deadline = 12
        elif "friday" in low or "weekend" in low:
            deadline = 47
        deadline = max(dur, min(HORIZON_HOURS, deadline))
        splittable = any(k in low for k in ("train", "batch", "encod", "render", "index"))
        name = re.sub(r"\s+", " ", p)[:60]
        jobs.append(Job(name, power, dur, deadline, splittable))
    return jobs


# ---------------------------------------------------------------- scheduling
def _free(load, h, power, cap):
    return load[h] + power <= cap + 1e-9


def schedule_jobs(
    jobs: list[Job],
    curves: pd.DataFrame,
    site_cap_kw: float,
    mode: str = "optimized",
) -> tuple[list[Job], np.ndarray, list[str]]:
    """Place jobs. mode='optimized' minimizes carbon; mode='naive' runs each job ASAP.

    Returns (scheduled copies of jobs, hourly load array in kW, list of warnings).
    """
    n = len(curves)
    carbon = curves["carbon_g_per_kwh"].to_numpy()
    load = np.zeros(n)
    warnings: list[str] = []
    placed: list[Job] = []

    # Largest energy first: the biggest jobs have the most to gain.
    order = sorted(jobs, key=lambda j: j.power_kw * j.duration_h, reverse=True)
    for j in order:
        j2 = Job(j.name, j.power_kw, j.duration_h, min(j.deadline_h, n), j.splittable)
        window = range(0, j2.deadline_h)
        hours: list[int] = []

        if mode == "naive":
            # Run immediately: take the earliest free hours, contiguous if possible.
            for h in window:
                if len(hours) == j2.duration_h:
                    break
                if _free(load, h, j2.power_kw, site_cap_kw):
                    hours.append(h)
                elif hours:
                    hours = []  # keep it contiguous
            if len(hours) < j2.duration_h:
                hours = []
        elif j2.splittable:
            ranked = sorted(window, key=lambda h: (carbon[h], h))
            hours = [h for h in ranked if _free(load, h, j2.power_kw, site_cap_kw)][: j2.duration_h]
            if len(hours) < j2.duration_h:
                hours = []
            hours.sort()
        else:
            best, best_cost = None, float("inf")
            for s in range(0, j2.deadline_h - j2.duration_h + 1):
                span = range(s, s + j2.duration_h)
                if all(_free(load, h, j2.power_kw, site_cap_kw) for h in span):
                    cost = float(sum(carbon[h] for h in span))
                    if cost < best_cost - 1e-9:
                        best, best_cost = s, cost
            hours = list(range(best, best + j2.duration_h)) if best is not None else []

        if not hours:
            warnings.append(
                f"'{j.name}' could not be scheduled before its deadline within the {site_cap_kw:g} kW site cap."
            )
        for h in hours:
            load[h] += j2.power_kw
        j2.schedule = hours
        placed.append(j2)
    return placed, load, warnings


def compute_metrics(jobs: list[Job], curves: pd.DataFrame) -> dict:
    carbon = curves["carbon_g_per_kwh"].to_numpy()
    price = curves["price_per_kwh"].to_numpy()
    solar = curves["solar_norm"].to_numpy()
    kwh = co2_g = cost = solar_kwh = 0.0
    for j in jobs:
        for h in j.schedule:
            e = j.power_kw
            kwh += e
            co2_g += e * carbon[h]
            cost += e * price[h]
            if solar[h] >= SOLAR_HOUR_THRESHOLD:
                solar_kwh += e
    return {
        "kwh": kwh,
        "co2_kg": co2_g / 1000.0,
        "cost": cost,
        "solar_kwh": solar_kwh,
        "solar_share": (solar_kwh / kwh) if kwh else 0.0,
    }


def explain_with_claude(jobs: list[Job], curves: pd.DataFrame, api_key: str, model: str = "claude-sonnet-4-5") -> str:
    import anthropic

    lines = []
    for j in jobs:
        if j.schedule:
            hrs = ", ".join(curves["time"].iloc[h].strftime("%a %-I%p") for h in j.schedule)
            lines.append(f"- {j.name}: {j.power_kw:g} kW, runs at {hrs}")
        else:
            lines.append(f"- {j.name}: could not be scheduled")
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": (
                    "In 3-5 short bullet points, explain to a non-expert why these compute jobs were placed at "
                    "these hours. The grid is cleanest and cheapest when solar output is high (midday). "
                    "Mention any job that could not be moved and why.\n\n" + "\n".join(lines)
                ),
            }
        ],
    )
    return "".join(b.text for b in msg.content if getattr(b, "text", None))
