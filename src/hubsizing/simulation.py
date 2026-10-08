"""Battery operation hour by hour (same logic as in the homework).

In every hour:
  - solar covers the demand directly
  - surplus solar fills the battery up to capacity C, the rest is curtailed
  - if the battery level is below L * C, cheap wind power (max 20 kWh)
    covers the deficit; wind that is left over refills the battery up to L * C
  - the remaining deficit comes from the battery, then from the grid

With L = 0 the wind contract is never used, which is the baseline problem.
The battery starts empty.
"""
import numba
import numpy as np


@numba.njit
def total_cost(C, L, solar_generation, charging_demand,
               battery_cost, grid_fee, wind_price, wind_limit):
    battery_level = 0.0
    grid_energy = 0.0
    wind_power = 0.0
    for h in range(len(solar_generation)):
        net_demand = charging_demand[h] - solar_generation[h]
        if battery_level < L * C:
            wind_available = wind_limit
        else:
            wind_available = 0.0

        if net_demand <= 0:
            battery_level += abs(net_demand)
            if battery_level > C:
                battery_level = C
        else:
            wind_usage = min(net_demand, wind_available)
            wind_power += wind_usage
            wind_available -= wind_usage
            net_demand -= wind_usage

            discharge = min(battery_level, net_demand)
            battery_level -= discharge
            net_demand -= discharge

            grid_energy += net_demand

        # leftover wind power buffers the battery up to L * C
        if wind_available > 0 and battery_level < L * C:
            top_up = min(wind_available, L * C - battery_level)
            battery_level += top_up
            wind_power += top_up

    investment_cost = battery_cost * C
    grid_penalty_cost = grid_fee * grid_energy
    wind_power_cost = wind_price * wind_power
    return investment_cost + grid_penalty_cost + wind_power_cost, grid_energy


@numba.njit
def _all_scenarios(C, L, solar_generation_scenarios, charging_demand_scenarios,
                   battery_cost, grid_fee, wind_price, wind_limit):
    n_scenarios = solar_generation_scenarios.shape[0]
    costs = np.zeros(n_scenarios)
    grid_energies = np.zeros(n_scenarios)
    for s in range(n_scenarios):
        costs[s], grid_energies[s] = total_cost(
            C, L, solar_generation_scenarios[s], charging_demand_scenarios[s],
            battery_cost, grid_fee, wind_price, wind_limit)
    return costs, grid_energies


def scenario_costs(C, L, solar_generation_scenarios, charging_demand_scenarios, econ):
    """Total cost and grid energy for every scenario (arrays of length n_scenarios)."""
    solar = np.atleast_2d(solar_generation_scenarios).astype(np.float64)
    demand = np.atleast_2d(charging_demand_scenarios).astype(np.float64)
    return _all_scenarios(float(C), float(L), solar, demand,
                          econ["battery_cost_eur_per_kwh"], econ["grid_price_eur_per_kwh"],
                          econ["wind_price_eur_per_kwh"], float(econ["wind_max_kwh_per_hour"]))


def expected_total_cost(C, L, solar_generation_scenarios, charging_demand_scenarios, econ):
    costs, _ = scenario_costs(C, L, solar_generation_scenarios, charging_demand_scenarios, econ)
    return np.mean(costs)