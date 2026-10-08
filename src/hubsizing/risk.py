"""Risk measures. For costs the bad outcomes are the HIGH values,
so VaR and CVaR look at the upper tail (e.g. the worst 5 %)."""
import numpy as np


def var(cost_samples, q=95):
    """Value at Risk: 95 % of the scenarios cost less than this."""
    return np.percentile(cost_samples, q)


def cvar(cost_samples, q=95):
    """Conditional VaR: average cost of the worst 5 % of the scenarios."""
    cost_samples = np.asarray(cost_samples)
    return np.mean(cost_samples[cost_samples >= var(cost_samples, q)])


def green_quotas(grid_energies, charging_demand_scenarios):
    """Share of the total demand that is NOT covered by the grid, per scenario."""
    total_demands = np.atleast_2d(charging_demand_scenarios).sum(axis=1)
    return 1 - np.asarray(grid_energies) / total_demands


def shortfall_penalty(green_quotas, target=0.90, penalty_per_pp=100):
    """Penalty for every percentage point below the green target."""
    shortfall_pp = np.maximum(target - green_quotas, 0) * 100
    return shortfall_pp * penalty_per_pp