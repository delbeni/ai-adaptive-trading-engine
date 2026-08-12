"""
features.py
------------
Transforme des données OHLCV brutes (exportées depuis MT5) en features
utilisables par le détecteur de régime et le détecteur d'impulsion.
"""

import numpy as np
import pandas as pd


def load_mt5_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {
        "<date>": "date", "<time>": "time", "<open>": "open", "<high>": "high",
        "<low>": "low", "<close>": "close", "<tickvol>": "tick_volume",
        "<vol>": "volume", "<spread>": "spread",
    }
    df = df.rename(columns=rename_map)
    if "date" in df.columns and "time" in df.columns:
        df["time"] = pd.to_datetime(df["date"] + " " + df["time"])
        df = df.drop(columns=["date"])
    else:
        df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def session_of(ts: pd.Timestamp) -> str:
    h = ts.hour
    if 0 <= h < 7:
        return "asia"
    if 7 <= h < 12:
        return "london"
    if 12 <= h < 16:
        return "london_ny_overlap"
    if 16 <= h < 21:
        return "newyork"
    return "off_hours"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ret"] = d["close"].pct_change()
    d["log_ret"] = np.log(d["close"]).diff()

    for w in (5, 10, 20, 50):
        d[f"mom_{w}"] = d["close"].pct_change(w)

    for w in (10, 20, 50):
        d[f"vol_{w}"] = d["log_ret"].rolling(w).std()

    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift()).abs(),
        (d["low"] - d["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    d["atr_14"] = tr.rolling(14).mean()
    d["atr_ratio"] = d["atr_14"] / d["atr_14"].rolling(100).mean()

    d["range_20"] = (d["high"].rolling(20).max() - d["low"].rolling(20).min())
    d["range_compression"] = d["range_20"] / d["range_20"].rolling(100).mean()

    d["ema_20"] = d["close"].ewm(span=20).mean()
    d["ema_50"] = d["close"].ewm(span=50).mean()
    d["trend_bias"] = (d["ema_20"] - d["ema_50"]) / d["atr_14"]

    up_move = d["high"].diff()
    down_move = -d["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_smooth = tr.rolling(14).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=d.index).rolling(14).mean() / atr_smooth
    minus_di = 100 * pd.Series(minus_dm, index=d.index).rolling(14).mean() / atr_smooth
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    d["adx_14"] = dx.rolling(14).mean()

    if "tick_volume" in d.columns:
        d["vol_z"] = (d["tick_volume"] - d["tick_volume"].rolling(50).mean()) / d["tick_volume"].rolling(50).std()

    if "spread" in d.columns:
        d["spread_z"] = (d["spread"] - d["spread"].rolling(200).mean()) / d["spread"].rolling(200).std()

    prior_high = d["high"].rolling(20).max().shift(1)
    prior_low = d["low"].rolling(20).min().shift(1)
    d["liquidity_sweep_high"] = ((d["high"] > prior_high) & (d["close"] < prior_high)).astype(int)
    d["liquidity_sweep_low"] = ((d["low"] < prior_low) & (d["close"] > prior_low)).astype(int)

    d["session"] = d["time"].apply(session_of)

    d = d.dropna().reset_index(drop=True)
    return d


REGIME_FEATURE_COLS = [
    "mom_5", "mom_10", "mom_20", "mom_50",
    "vol_10", "vol_20", "vol_50",
    "atr_ratio", "range_compression", "trend_bias", "adx_14",
]

IMPULSE_FEATURE_COLS = REGIME_FEATURE_COLS + [
    "liquidity_sweep_high", "liquidity_sweep_low",
]
