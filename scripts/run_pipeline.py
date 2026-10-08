"""Full analysis: data -> solar forecast -> scenarios -> optimisation -> risk -> figures.

Run from the project root:
    python scripts/run_pipeline.py            # uses saved data if it exists
    python scripts/run_pipeline.py --refresh  # downloads fresh data first
"""
import argparse
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from hubsizing import plots, weather
from hubsizing.config import PROJECT_ROOT, load_config
from hubsizing.demand import demand_scenarios, expected_charging_demand
from hubsizing.forecasting import (estimate_daily_correlation, forecast_solar,
                                   solar_scenarios, train_solar_model)
from hubsizing.optimize import best_capacity, best_capacity_and_level
from hubsizing.risk import cvar, green_quotas, var
from hubsizing.simulation import expected_total_cost, scenario_costs

RAW = PROJECT_ROOT / "data" / "raw"
RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"


def load_or_download(path, download, refresh, timezone):
    """Read a saved CSV, or download it (and save it) if needed."""
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df["local_time"] = df.index.tz_convert(timezone)
        print(f"  loaded {path.name} ({len(df)} rows)")
        return df
    df = download()
    df.drop(columns="local_time").to_csv(path)
    print(f"  downloaded {path.name} ({len(df)} rows)")
    return df


def get_data(cfg, refresh):
    site, horizon = cfg["site"], cfg["horizon"]
    if horizon["mode"] == "historical":
        start = date.fromisoformat(horizon["historical_start"])
        end = start + timedelta(days=horizon["days"] - 1)
        training_end = start - timedelta(days=1)
        download_horizon = lambda: weather.fetch_history(site, start, end)  # noqa: E731
    else:
        training_end = date.today() - timedelta(days=3)
        download_horizon = lambda: weather.fetch_forecast(site, horizon["days"])  # noqa: E731
    training_start = training_end - timedelta(days=365 * cfg["training"]["years"])

    planning_context = load_or_download(RAW / "horizon.csv", download_horizon,
                                        refresh, site["timezone"])
    training_data = load_or_download(
        RAW / "history.csv",
        lambda: weather.fetch_history(site, training_start, training_end),
        refresh, site["timezone"])
    return training_data, planning_context


def main(refresh=False):
    cfg = load_config()
    econ, risk, sim = cfg["economics"], cfg["risk"], cfg["simulation"]
    q = risk["alpha"] * 100
    RAW.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # 1. Data ------------------------------------------------------------
    print("1/6 Data")
    training_data, planning_context = get_data(cfg, refresh)
    training_data = weather.add_features(training_data, cfg["site"], cfg["pv"])
    planning_context = weather.add_features(planning_context, cfg["site"])
    hours = planning_context["hour"].to_numpy()
    weekend = planning_context["weekend"].to_numpy()
    dates = planning_context["local_time"].dt.date.to_numpy()

    # 2. Solar forecast ----------------------------------------------------
    print("2/6 Solar forecast (NGBoost)")
    ngb_model_solar = train_solar_model(training_data, seed=sim["seed"])
    rho = estimate_daily_correlation(ngb_model_solar, training_data)
    loc, scale = forecast_solar(ngb_model_solar, planning_context)
    print(f"  share of daily forecast error: rho = {rho:.2f}")

    # 3. Scenarios: one set to optimise, an independent set to evaluate ----
    print("3/6 Scenarios")
    kwp = cfg["pv"]["capacity_kwp"]
    rng = np.random.default_rng(sim["seed"])
    solar_opt = solar_scenarios(loc, scale, dates, sim["n_scenarios"], rho, kwp, rng)
    demand_opt = demand_scenarios(hours, weekend, cfg["demand"], sim["n_scenarios"], rng)
    rng_eval = np.random.default_rng(sim["seed"] + 1)
    solar_eval = solar_scenarios(loc, scale, dates, sim["n_scenarios_eval"], rho, kwp, rng_eval)
    demand_eval = demand_scenarios(hours, weekend, cfg["demand"], sim["n_scenarios_eval"], rng_eval)

    expected_solar = np.clip(loc, 0, kwp)
    expected_demand = expected_charging_demand(hours, weekend, cfg["demand"])

    # 4. Deterministic vs stochastic capacity ------------------------------
    print("4/6 Deterministic vs stochastic capacity")
    capacities = np.arange(0, cfg["search"]["capacity_max_kwh"] + 1,
                           cfg["search"]["capacity_step_kwh"], dtype=float)
    activation_levels = np.round(np.arange(0, 1.0001, cfg["search"]["activation_step"]), 2)

    C_det, deterministic_costs = best_capacity(capacities, expected_solar, expected_demand, econ)
    C_sto, stochastic_costs = best_capacity(capacities, solar_opt, demand_opt, econ)
    eev = expected_total_cost(C_det, 0, solar_eval, demand_eval, econ)   # deterministic choice, real uncertainty
    eiu = expected_total_cost(C_sto, 0, solar_eval, demand_eval, econ)   # stochastic choice
    flaw_of_averages = eev - deterministic_costs.min()
    eviu = eev - eiu

    # 5. Wind contract, risk and green quota -------------------------------
    print("5/6 Wind contract and risk")
    C_star, L_star, cost_surface = best_capacity_and_level(
        capacities, activation_levels, solar_opt, demand_opt, econ)
    cost_samples, grid_energies = scenario_costs(C_star, L_star, solar_eval, demand_eval, econ)
    quotas = green_quotas(grid_energies, demand_eval)

    C_pen, L_pen, _ = best_capacity_and_level(
        capacities, activation_levels, solar_opt, demand_opt, econ,
        target=risk["green_target"], penalty_per_pp=risk["shortfall_penalty_eur_per_pp"])
    pen_costs, pen_grid = scenario_costs(C_pen, L_pen, solar_eval, demand_eval, econ)
    pen_quotas = green_quotas(pen_grid, demand_eval)

    summary = {
        "site": cfg["site"]["name"],
        "horizon_mode": cfg["horizon"]["mode"],
        "horizon_start": str(planning_context["local_time"].iloc[0]),
        "horizon_end": str(planning_context["local_time"].iloc[-1]),
        "rho_daily_forecast_error": round(rho, 3),
        "deterministic": {
            "capacity_kwh": float(C_det),
            "planned_cost_eur": round(float(deterministic_costs.min()), 2),
            "expected_cost_under_uncertainty_eur": round(float(eev), 2),
            "flaw_of_averages_eur": round(float(flaw_of_averages), 2),
        },
        "stochastic": {
            "capacity_kwh": float(C_sto),
            "expected_cost_eur": round(float(eiu), 2),
            "eviu_eur": round(float(eviu), 2),
        },
        "wind_contract": {
            "capacity_kwh": float(C_star),
            "activation_level": float(L_star),
            "expected_cost_eur": round(float(np.mean(cost_samples)), 2),
            "var_eur": round(float(var(cost_samples, q)), 2),
            "cvar_eur": round(float(cvar(cost_samples, q)), 2),
            "mean_green_quota": round(float(np.mean(quotas)), 4),
            "shortfall_probability": round(float(np.mean(quotas < risk["green_target"])), 4),
        },
        "with_shortfall_penalty": {
            "capacity_kwh": float(C_pen),
            "activation_level": float(L_pen),
            "expected_cost_eur": round(float(np.mean(pen_costs)), 2),
            "shortfall_probability": round(float(np.mean(pen_quotas < risk["green_target"])), 4),
        },
    }
    with open(RESULTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 6. Figures -----------------------------------------------------------
    print("6/6 Figures")
    times = planning_context["local_time"].dt.tz_localize(None)
    plots.daily_profiles(hours, expected_solar, expected_demand, FIGURES / "01_daily_profiles.png")
    plots.solar_fan_chart(times, solar_opt, FIGURES / "02_solar_scenarios.png")
    plots.cost_vs_capacity(capacities, stochastic_costs, deterministic_costs, C_det, C_sto,
                           FIGURES / "03_cost_vs_capacity.png")
    plots.cost_surface(capacities, activation_levels, cost_surface, C_star, L_star,
                       FIGURES / "04_cost_surface.png")
    plots.cost_distribution(cost_samples, var(cost_samples, q), cvar(cost_samples, q),
                            risk["alpha"], FIGURES / "05_cost_distribution.png")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="download fresh data")
    main(parser.parse_args().refresh)