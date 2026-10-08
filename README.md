# Risk-aware battery sizing for a PV-and-storage EV charging hub

How large should the battery of a solar-powered EV charging hub in Bielefeld be, when both
sunshine and charging demand are uncertain? This project answers the question with a
probabilistic solar forecast (NGBoost on real weather data), Monte Carlo scenarios and
simulation-based optimisation, and then asks what a cheap but limited wind-power contract
and a 90 % green-energy target change about the decision.

## Key results

Planning horizon: 9–22 October 2026 (14 days), Open-Meteo forecast for Bielefeld.

| Strategy | Battery | Wind activation level | Expected cost |
|---|---|---|---|
| Plan with the average forecast | 200 kWh | – | 3,336 € (planned: 3,218 €) |
| Plan with scenarios | 250 kWh | – | 3,314 € |
| + flexible wind contract | 350 kWh | 0.90 | 1,614 € |
| + penalty for missing the 90 % green target | 360 kWh | 0.95 | 1,615 € |

1. **The flaw of averages is real.** Planning with average solar and demand underestimates
   cost by 118 € (3.7 %). Accounting for uncertainty leads to a larger battery and saves
   21 € (expected value of including uncertainty, evaluated on independent scenarios).
2. **Flexible supply and storage are complements here.** The wind contract halves the cost
   and *increases* the optimal battery, because cheap wind energy (0.15 €/kWh) can be stored
   and used instead of grid power (0.50 €/kWh).
3. **Risk.** At the optimal configuration the 95 % VaR is 1,946 € and the 95 % CVaR is
   2,041 €. The hub covers 93 % of demand without the grid on average, but misses the 90 %
   target in 11 % of scenarios; a shortfall penalty cuts this to 8 % for 1.5 € extra
   expected cost.

The numbers change with every run, because the planning horizon is always the next 14
days. `results/summary.json` holds the latest results.

![Cost vs capacity](results/figures/03_cost_vs_capacity.png)
![Cost surface](results/figures/04_cost_surface.png)

More figures: [daily profiles](results/figures/01_daily_profiles.png),
[solar scenarios](results/figures/02_solar_scenarios.png),
[cost distribution](results/figures/05_cost_distribution.png).

## Method

1. **Data.** Hourly solar irradiance and cloud cover from the [Open-Meteo](https://open-meteo.com)
   API: three years of ERA5 history for training and the weather forecast for the
   planning horizon. PV output = installed kWp × irradiance / 1000 × performance ratio.
2. **Probabilistic solar forecast.** NGBoost (Normal distribution) on solar elevation and
   cloud cover, trained on daytime hours only; generation at night is exactly zero.
3. **Scenarios.** Solar scenarios share a daily random shock, so cloudy days stay cloudy;
   its weight (ρ = 0.42) is estimated from the forecast errors. Demand comes from a
   stochastic simulator: Poisson arrivals with a weekday/weekend commuter profile and
   LogNormal energy per session.
4. **Simulation and optimisation.** An hourly battery model (Numba) computes investment,
   grid and wind costs for each scenario; grid search over capacity C and activation
   level L. Decisions are optimised on 2,000 scenarios and evaluated on 2,000 others.
5. **Risk.** VaR and CVaR on the upper cost tail, green-quota shortfall probability, and a
   penalised objective (100 € per percentage point below 90 %).

## Assumptions and limitations

- **Demand is simulated, not measured.** No open, hub-level charging data exist for
  Germany; the arrival profile and session energy are documented assumptions in
  `config/params.yaml`.
- Solar and demand are independent; weather does not affect charging behaviour.
- The model is trained on reanalysis cloud cover but applied to forecast cloud cover,
  which may understate forecast uncertainty.
- PV output uses horizontal irradiance and a constant performance ratio (no tilt or
  temperature model); charging is assumed to be completed within the arrival hour.
- Battery: no efficiency losses, starts empty. Wind energy counts as green.
- Wind rule: when the battery is below L × C, up to 20 kWh/h of wind covers the deficit
  and refills the battery up to L × C.

## How to run

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; on Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
python scripts/run_pipeline.py --refresh   # download fresh data and run everything
python -m pytest -q                        # tests
```

All parameters (location, PV size, prices, demand profile, number of scenarios) are in
`config/params.yaml`. Set `horizon.mode: historical` to analyse a past period instead of
the forecast.

## Project structure

```
config/params.yaml          all model inputs
src/hubsizing/
    weather.py              Open-Meteo download, solar position, PV output
    demand.py               stochastic charging demand
    forecasting.py          NGBoost forecast and correlated solar scenarios
    simulation.py           hourly battery operation and costs
    optimize.py             grid search over C and (C, L)
    risk.py                 VaR, CVaR, green quota, shortfall penalty
    plots.py                figures
scripts/run_pipeline.py     end-to-end analysis
tests/                      sanity checks
results/                    summary.json and figures
```

## Background

The case is based on a homework assignment from the course *Combining OR and Data Science*
(Prof. Dr. Michael Römer, Bielefeld University, summer term 2026). This version replaces the
course data with real weather data and a documented demand simulator, and extends the
original analysis with correlated scenarios, out-of-sample evaluation and corrected
upper-tail risk measures.

Weather data: [Open-Meteo](https://open-meteo.com) (CC BY 4.0).