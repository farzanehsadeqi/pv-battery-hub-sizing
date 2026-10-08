"""Fetch hourly weather from Open-Meteo and convert irradiance to PV output.

Open-Meteo needs no API key. Two endpoints are used:
  - archive  : ERA5 reanalysis history (available up to about two days ago)
  - forecast : model forecast, up to 16 days ahead
Times are requested in UTC and converted to local time afterwards, which
avoids duplicated or missing hours at daylight-saving switches.
"""
import numpy as np
import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
VARIABLES = ["shortwave_radiation", "cloud_cover"]


def _to_frame(payload, timezone):
    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")
    df = df.set_index("time")
    df["local_time"] = df.index.tz_convert(timezone)
    return df


def fetch_history(site, start, end):
    """Hourly irradiance (W/m2) and cloud cover (%) between two dates."""
    params = {
        "latitude": site["latitude"],
        "longitude": site["longitude"],
        "start_date": str(start),
        "end_date": str(end),
        "hourly": ",".join(VARIABLES),
        "timezone": "UTC",
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=120)
    r.raise_for_status()
    return _to_frame(r.json(), site["timezone"])


def fetch_forecast(site, days):
    """Hourly forecast for the next `days` days, starting tomorrow 00:00 local time."""
    params = {
        "latitude": site["latitude"],
        "longitude": site["longitude"],
        "hourly": ",".join(VARIABLES),
        "forecast_days": min(days + 1, 16),
        "timezone": "UTC",
    }
    r = requests.get(FORECAST_URL, params=params, timeout=60)
    r.raise_for_status()
    df = _to_frame(r.json(), site["timezone"])
    start = (df["local_time"].iloc[0] + pd.Timedelta(days=1)).normalize()
    end = start + pd.Timedelta(days=days)
    return df[(df["local_time"] >= start) & (df["local_time"] < end)]


def solar_elevation_sin(index_utc, latitude, longitude):
    """Sine of the sun's elevation at the middle of each hour.

    Open-Meteo radiation is the mean of the preceding hour, so the
    middle of the hour is the timestamp minus 30 minutes. Uses a standard
    approximation (declination + equation of time), accurate enough here.
    """
    t = index_utc - pd.Timedelta(minutes=30)
    doy = t.dayofyear.to_numpy()
    hour = (t.hour + t.minute / 60).to_numpy()
    b = 2 * np.pi * (doy - 81) / 364
    eot_min = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    solar_time = hour + longitude / 15 + eot_min / 60
    hour_angle = np.radians(15 * (solar_time - 12))
    lat = np.radians(latitude)
    return np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)


def pv_energy_kwh(radiation_w_m2, pv):
    """Hourly PV energy (kWh) from horizontal irradiance, simple linear model."""
    return pv["capacity_kwp"] * np.asarray(radiation_w_m2) / 1000 * pv["performance_ratio"]


def add_features(df, site, pv=None):
    """Add model features (and PV output when irradiance is observed)."""
    df = df.copy()
    df["sin_elevation"] = solar_elevation_sin(df.index, site["latitude"], site["longitude"])
    df["hour"] = df["local_time"].dt.hour
    df["weekend"] = (df["local_time"].dt.dayofweek >= 5).astype(int)
    df["cloud_cover"] = df["cloud_cover"].interpolate().fillna(50.0)
    if pv is not None and "shortwave_radiation" in df:
        df["pv_kwh"] = pv_energy_kwh(df["shortwave_radiation"].fillna(0), pv)
    return df