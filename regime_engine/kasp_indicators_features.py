"""
kasp_indicators_features.py
-----------------------------
Reproduit en Python la logique des indicateurs MQL5 de Kasp
(KASP_FusionOsc et KASP_TradingFusion), pour donner au modèle ces
signaux comme features supplémentaires -- exactement ce que Kasp
utilise déjà visuellement sur ses graphiques, mais appris
statistiquement plutôt que suivi à l'oeil.

Modules reproduits :
1. Heikin Ashi multi-timeframe (KASP_FusionOsc) : tendance HA sur
   H1/H4/D1, alignement des 3 (signal combiné BUY/SELL du fichier
   original).
2. SuperTrend (KASP_TradingFusion, module MTF_Candles) : ATR-based
   trend follower, direction + distance au niveau.
3. SWDL (KASP_TradingFusion, module SWDL_Historique) : distance du
   prix actuel aux plus haut/bas/open de la semaine et du jour.
4. TrendTrader (KASP_TradingFusion, module TT) : ruban EMA50/EMA100 +
   ADX(25), comme dans le fichier original.
"""

import numpy as np
import pandas as pd


# ------------------------- 1. Heikin Ashi MTF -------------------------

def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les bougies Heikin Ashi (formule identique au .mq5 :
    ha_close = OHLC/4, ha_open = moyenne du ha_open/ha_close précédent)."""
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = np.zeros(len(df))
    ha_open[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2.0
    ha_color = np.where(ha_open > ha_close, -1, np.where(ha_open < ha_close, 1, 0))
    return pd.DataFrame({"ha_open": ha_open, "ha_close": ha_close.values, "ha_color": ha_color})


def heikin_ashi_mtf_features(df_h1: pd.DataFrame) -> pd.DataFrame:
    """Tendance HA sur H1 (natif), H4 et D1 (ré-échantillonnés), reportée
    sur chaque bougie H1 -- comme le fait l'indicateur original avec
    security(). Ajoute aussi le signal combiné (alignement des 3)."""
    d = df_h1.reset_index(drop=True)

    ha_h1 = heikin_ashi(d)
    d["ha_h1_color"] = ha_h1["ha_color"].values

    d_h4 = d.set_index("time")[["open", "high", "low", "close"]].resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna().reset_index()
    ha_h4 = heikin_ashi(d_h4)
    d_h4["ha_h4_color"] = ha_h4["ha_color"].values
    # CORRECTIF anti-fuite : le label du bloc H4 est son DEBUT (ex: 00:00),
    # mais ses OHLC ne sont connus qu'a sa CLOTURE (04:00 plus tard). On
    # decale le temps de reference a la cloture reelle pour que merge_asof
    # ne donne jamais au H1 une info H4 pas encore disponible.
    d_h4["time"] = d_h4["time"] + pd.Timedelta(hours=4)

    d_d1 = d.set_index("time")[["open", "high", "low", "close"]].resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna().reset_index()
    ha_d1 = heikin_ashi(d_d1)
    d_d1["ha_d1_color"] = ha_d1["ha_color"].values
    # Meme correctif anti-fuite pour le D1 (cloture 24h apres le label)
    d_d1["time"] = d_d1["time"] + pd.Timedelta(days=1)

    # Report des couleurs HTF -> H1 (merge_asof : la dernière bougie HTF close avant chaque bougie H1)
    d = pd.merge_asof(d.sort_values("time"), d_h4[["time", "ha_h4_color"]].sort_values("time"), on="time")
    d = pd.merge_asof(d.sort_values("time"), d_d1[["time", "ha_d1_color"]].sort_values("time"), on="time")

    # Signal combiné : les 3 alignés (comme le "tousVert"/"tousRouge" du .mq5)
    all_bull = (d["ha_h1_color"] == 1) & (d["ha_h4_color"] == 1) & (d["ha_d1_color"] == 1)
    all_bear = (d["ha_h1_color"] == -1) & (d["ha_h4_color"] == -1) & (d["ha_d1_color"] == -1)
    d["ha_combo_signal"] = np.where(all_bull, 1, np.where(all_bear, -1, 0))
    d["ha_alignment_count"] = d[["ha_h1_color", "ha_h4_color", "ha_d1_color"]].apply(
        lambda row: max((row == 1).sum(), (row == -1).sum()), axis=1
    )

    return d


# ------------------------- 2. SuperTrend -------------------------

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """SuperTrend classique, formule identique au module MTF_Candles du .mq5."""
    d = df.copy()
    hl2 = (d["high"] + d["low"]) / 2.0

    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift()).abs(),
        (d["low"] - d["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    direction = np.zeros(len(d))
    st = np.zeros(len(d))
    direction[0] = 1

    for i in range(1, len(d)):
        if d["close"].iloc[i] > upper_band.iloc[i - 1]:
            direction[i] = 1
        elif d["close"].iloc[i] < lower_band.iloc[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
            if direction[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i - 1]:
                lower_band.iloc[i] = lower_band.iloc[i - 1]
            if direction[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i - 1]:
                upper_band.iloc[i] = upper_band.iloc[i - 1]
        st[i] = lower_band.iloc[i] if direction[i] == 1 else upper_band.iloc[i]

    d["supertrend_direction"] = direction
    d["supertrend_dist"] = (d["close"] - st) / d["close"]
    return d


# ------------------------- 3. SWDL (niveaux hebdo/jour) -------------------------

def swdl_features(df_h1: pd.DataFrame) -> pd.DataFrame:
    """Distance du prix actuel aux plus haut/bas/open de la semaine et
    du jour en cours (calcul causal : uniquement les niveaux déjà connus)."""
    d = df_h1.reset_index(drop=True).copy()
    d["week"] = d["time"].dt.isocalendar().week.astype(str) + "_" + d["time"].dt.isocalendar().year.astype(str)
    d["day"] = d["time"].dt.date

    # High/Low/Open glissants de la semaine/jour EN COURS (pas encore clôturés,
    # donc uniquement ce qui est connu jusqu'à la bougie actuelle -- causal)
    d["week_high_sofar"] = d.groupby("week")["high"].cummax()
    d["week_low_sofar"] = d.groupby("week")["low"].cummin()
    d["week_open"] = d.groupby("week")["open"].transform("first")

    d["day_high_sofar"] = d.groupby("day")["high"].cummax()
    d["day_low_sofar"] = d.groupby("day")["low"].cummin()

    d["dist_week_high"] = (d["week_high_sofar"] - d["close"]) / d["close"]
    d["dist_week_low"] = (d["close"] - d["week_low_sofar"]) / d["close"]
    d["dist_week_open"] = (d["close"] - d["week_open"]) / d["close"]
    d["dist_day_high"] = (d["day_high_sofar"] - d["close"]) / d["close"]
    d["dist_day_low"] = (d["close"] - d["day_low_sofar"]) / d["close"]

    d = d.drop(columns=["week", "day", "week_high_sofar", "week_low_sofar",
                         "week_open", "day_high_sofar", "day_low_sofar"])
    return d


# ------------------------- 4. TrendTrader (ruban EMA + ADX) -------------------------

def trendtrader_features(df: pd.DataFrame, short_period: int = 50, long_period: int = 100,
                          adx_period: int = 25) -> pd.DataFrame:
    """Ruban EMA50/EMA100 + ADX(25), comme le module TT du .mq5."""
    d = df.copy()
    ema_short = d["close"].ewm(span=short_period, adjust=False).mean()
    ema_long = d["close"].ewm(span=long_period, adjust=False).mean()
    d["tt_ribbon_spread"] = (ema_short - ema_long) / d["close"]
    d["tt_ribbon_bull"] = (ema_short > ema_long).astype(int)

    up_move = d["high"].diff()
    down_move = -d["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift()).abs(),
        (d["low"] - d["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_smooth = tr.rolling(adx_period).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=d.index).rolling(adx_period).mean() / atr_smooth
    minus_di = 100 * pd.Series(minus_dm, index=d.index).rolling(adx_period).mean() / atr_smooth
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    d["tt_adx"] = dx.rolling(adx_period).mean()
    return d


def build_kasp_indicator_features(df_h1: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet : applique les 4 modules et retourne le DataFrame enrichi."""
    d = heikin_ashi_mtf_features(df_h1)
    d = supertrend(d)
    d = swdl_features(d)
    d = trendtrader_features(d)
    d = d.iloc[5:].reset_index(drop=True)  # retire les 1eres lignes (SuperTrend pas encore fiable)
    return d


KASP_FEATURE_COLS = [
    "ha_h1_color", "ha_h4_color", "ha_d1_color", "ha_combo_signal", "ha_alignment_count",
    "supertrend_direction", "supertrend_dist",
    "dist_week_high", "dist_week_low", "dist_week_open", "dist_day_high", "dist_day_low",
    "tt_ribbon_spread", "tt_ribbon_bull", "tt_adx",
]
