"""Sanity checks for the battery simulation, demand model and risk measures."""
import numpy as np

from hubsizing.demand import demand_scenarios
from hubsizing.risk import cvar, var
from hubsizing.simulation import scenario_costs

ECON = {"battery_cost_eur_per_kwh": 1.0, "grid_price_eur_per_kwh": 0.5,
        "wind_price_eur_per_kwh": 0.15, "wind_max_kwh_per_hour": 20}


def test_no_battery_means_all_deficit_from_grid():
    solar = np.array([[0.0, 10.0, 0.0]])
    demand = np.array([[5.0, 2.0, 4.0]])
    costs, grid = scenario_costs(0, 0, solar, demand, ECON)
    assert grid[0] == 9.0               # 5 + 4; the surplus of 8 is curtailed
    assert costs[0] == 0.5 * 9.0


def test_battery_respects_capacity():
    solar = np.array([[10.0, 0.0]])
    demand = np.array([[0.0, 10.0]])
    _, grid_small = scenario_costs(4, 0, solar, demand, ECON)
    _, grid_big = scenario_costs(20, 0, solar, demand, ECON)
    assert grid_small[0] == 6.0         # only 4 kWh could be stored
    assert grid_big[0] == 0.0


def test_wind_never_increases_grid_energy():
    rng = np.random.default_rng(1)
    solar = rng.uniform(0, 30, (50, 48))
    demand = rng.uniform(0, 30, (50, 48))
    _, grid_base = scenario_costs(100, 0, solar, demand, ECON)
    _, grid_wind = scenario_costs(100, 0.5, solar, demand, ECON)
    assert np.all(grid_wind <= grid_base + 1e-9)


def test_demand_mean_matches_profile():
    cfg = {"arrivals_weekday": [2.0] * 24, "arrivals_weekend": [1.0] * 24,
           "session_energy_mean_kwh": 20, "session_energy_sd_kwh": 8}
    hours = np.arange(24)
    demand = demand_scenarios(hours, np.zeros(24), cfg, 5000, np.random.default_rng(2))
    assert abs(demand.mean() - 40.0) < 0.7      # 2 cars * 20 kWh
    assert (demand >= 0).all()


def test_cvar_is_not_below_var():
    costs = np.random.default_rng(3).normal(1000, 100, 10000)
    assert cvar(costs) >= var(costs) > np.mean(costs)