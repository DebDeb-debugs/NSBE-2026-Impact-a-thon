# Carbon-Aware Job Scheduler

Describe compute jobs in plain English. The scheduler places each one in the sunniest, lowest-carbon hours before its deadline and shows what that saves compared with running everything right away.

Built for the 2026 NSBE FRC AI Impact-a-thon (theme: Engineering a Sustainable Tomorrow).

## Two versions

| | Streamlit app (`app.py`) | Static web page (`web/index.html`) |
|---|---|---|
| Solar data | Live forecast from Open-Meteo, with an offline fallback | Modeled clear-sky curve plus a sky-condition setting |
| Claude | Optional, with an Anthropic API key | Optional, where Claude is available |
| Run it | `streamlit run app.py` | Open the file in a browser |

## Run the Streamlit app

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # optional; or paste it in the sidebar
streamlit run app.py
```

Without an API key, a rule-based reader parses the jobs. Without internet, a synthetic solar curve is used.

## How it works

1. Jobs are parsed into structured form: power draw, duration, deadline, and whether the job can pause.
2. Hourly solar output gives an assumed grid carbon intensity and time-of-use price.
3. A greedy scheduler places the biggest jobs first in the lowest-carbon window that meets the deadline, under a site power cap. Jobs that can pause are split across the cleanest hours.
4. The result is compared with a run-immediately baseline for CO2, cost, and share of energy used in solar hours.

## Assumptions (replace with sourced figures)

- Grid carbon intensity falls from 450 to 120 g CO2/kWh as solar output rises.
- Illustrative time-of-use prices: $0.55 (4-9pm), $0.26 (9am-4pm), $0.32 otherwise.
- Job power draws are typical values, for example 0.7 kW per GPU.
- One simulated site. Total energy is identical in both schedules; only timing changes.

## Future work

Real grid carbon APIs (Electricity Maps, WattTime), Kubernetes or Slurm integration, multi-region shifting, and pause-and-resume support.
