"""Grid search for the best capacity C and activation level L."""
import numpy as np

from .risk import green_quotas, shortfall_penalty
from .simulation import scenario_costs


def best_capacity(capacities, solar_generation_scenarios, charging_demand_scenarios, econ):
    """Baseline problem (no wind contract): expected cost for every capacity."""
    expected_costs = []
    for C in capacities:
        costs, _ = scenario_costs(C, 0, solar_generation_scenarios, charging_demand_scenarios, econ)
        expected_costs.append(np.mean(costs))
    best_index = np.argmin(expected_costs)
    return capacities[best_index], np.array(expected_costs)


def best_capacity_and_level(capacities, activation_levels,
                            solar_generation_scenarios, charging_demand_scenarios, econ,
                            target=None, penalty_per_pp=0):
    """Joint search over (C, L). With a target, the shortfall penalty is added."""
    expected_costs = []
    for C in capacities:
        for L in activation_levels:
            costs, grid_energies = scenario_costs(C, L, solar_generation_scenarios,
                                                  charging_demand_scenarios, econ)
            if target is not None:
                quotas = green_quotas(grid_energies, charging_demand_scenarios)
                costs = costs + shortfall_penalty(quotas, target, penalty_per_pp)
            expected_costs.append(np.mean(costs))

    best_index = np.argmin(expected_costs)
    C_star = capacities[best_index // len(activation_levels)]
    L_star = activation_levels[best_index % len(activation_levels)]
    cost_surface = np.array(expected_costs).reshape(len(capacities), len(activation_levels))
    return C_star, L_star, cost_surface