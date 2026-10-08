"""Figures for the README and the notebook."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def daily_profiles(hours, solar_mean, demand_mean, path):
    hours = np.asarray(hours)
    h = np.arange(24)
    solar = [solar_mean[hours == k].mean() for k in h]
    dm = [demand_mean[hours == k].mean() for k in h]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(h, solar, label="Expected solar generation", color="#e8a317", lw=2)
    ax.plot(h, dm, label="Expected charging demand", color="#1f5f99", lw=2)
    ax.fill_between(h, solar, dm, where=np.array(dm) > np.array(solar), color="#1f5f99",
                    alpha=0.15, label="Deficit (battery or grid)")
    ax.set_xlabel("Hour of day (local time)")
    ax.set_ylabel("kWh per hour")
    ax.set_title("Average daily profile over the planning horizon")
    ax.set_xticks(range(0, 24, 3))
    ax.legend()
    _save(fig, path)


def solar_fan_chart(times, solar_scenarios, path):
    q05, q50, q95 = np.quantile(solar_scenarios, [0.05, 0.5, 0.95], axis=0)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(times, q05, q95, color="#e8a317", alpha=0.3, label="5-95 % of scenarios")
    ax.plot(times, q50, color="#b07400", lw=1, label="Median")
    ax.set_ylabel("Solar generation (kWh per hour)")
    ax.set_title("Solar generation scenarios over the planning horizon")
    ax.legend(loc="upper right")
    _save(fig, path)


def cost_vs_capacity(capacities, stochastic, deterministic, c_det, c_sto, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(capacities, stochastic, lw=2, label="Expected cost (scenarios)")
    ax.plot(capacities, deterministic, lw=2, ls="--", label="Cost with average forecast")
    ax.axvline(c_det, color="grey", ls=":", label=f"Deterministic choice: {c_det:.0f} kWh")
    ax.axvline(c_sto, color="red", ls=":", label=f"Stochastic optimum: {c_sto:.0f} kWh")
    ax.set_xlabel("Battery capacity C (kWh)")
    ax.set_ylabel("Total cost (EUR)")
    ax.set_title("Total cost as a function of battery capacity")
    ax.legend()
    _save(fig, path)


def cost_surface(capacities, levels, surface, c_star, l_star, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(levels, capacities, surface, shading="nearest", cmap="viridis_r")
    ax.plot(l_star, c_star, "r*", ms=14, label=f"Optimum: C={c_star:.0f} kWh, L={l_star:.2f}")
    fig.colorbar(im, ax=ax, label="Expected total cost (EUR)")
    ax.set_xlabel("Activation level L")
    ax.set_ylabel("Battery capacity C (kWh)")
    ax.set_title("Joint choice of capacity and wind activation level")
    ax.legend(loc="upper right")
    _save(fig, path)


def cost_distribution(costs, var, cvar, alpha, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(costs, bins=50, color="#cccccc", edgecolor="#888888")
    ax.axvline(np.mean(costs), color="black", lw=2, label=f"Mean: {np.mean(costs):.0f} EUR")
    ax.axvline(var, color="orange", lw=2, label=f"VaR {alpha:.0%}: {var:.0f} EUR")
    ax.axvline(cvar, color="red", lw=2, label=f"CVaR {alpha:.0%}: {cvar:.0f} EUR")
    ax.set_xlabel("Total cost (EUR)")
    ax.set_ylabel("Number of scenarios")
    ax.set_title("Cost distribution at the optimal configuration")
    ax.legend()
    _save(fig, path)