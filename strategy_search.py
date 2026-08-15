"""
strategy_search.py
--------------------
L'IA teste elle-même plusieurs combinaisons de réglages (horizon de
prédiction, seuil de mouvement, unité de temps) et garde celle qui
marche le mieux -- honnêtement, en deux temps :

1. Chaque combinaison est évaluée sur la période de VALIDATION (jamais
   sur le test) -- c'est ce qui évite de "tricher" en choisissant après
   coup celle qui a eu de la chance sur les données finales.
2. Une fois la meilleure combinaison choisie, on l'évalue UNE SEULE FOIS
   sur le vrai test set réservé -- c'est le chiffre final, honnête.

Usage :
    python strategy_search.py --csv XAUUSD_H1_15ans.csv --symbol XAUUSD
"""

import argparse
import itertools
import pandas as pd

from regime_engine.features import load_mt5_csv, build_features, IMPULSE_FEATURE_COLS
from regime_engine.kasp_indicators_features import build_kasp_indicator_features, KASP_FEATURE_COLS
from regime_engine.regime_detector import make_impulse_labels
from train_kasp import fit_impulse_kasp, backtest_dollars, ALL_FEATURE_COLS


def resample_to_timeframe(df_h1: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = df_h1.set_index("time")
    out = pd.DataFrame({
        "open": d["open"].resample(rule).first(),
        "high": d["high"].resample(rule).max(),
        "low": d["low"].resample(rule).min(),
        "close": d["close"].resample(rule).last(),
        "tick_volume": d["tick_volume"].resample(rule).sum() if "tick_volume" in d.columns else 0,
        "spread": d["spread"].resample(rule).mean() if "spread" in d.columns else 0,
    }).dropna().reset_index()
    return out


def chronological_split(df, train_frac=0.70, val_frac=0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def evaluate_combo(feats_kasp, horizon, atr_multiple, min_proba=0.65, label=""):
    train_df, val_df, test_df = chronological_split(feats_kasp)

    scaler, model = fit_impulse_kasp(train_df, horizon=horizon, atr_multiple=atr_multiple)

    # Évaluation sur VALIDATION (utilisée pour choisir, jamais le test)
    trades_val, balance_val = backtest_dollars(val_df, scaler, model, min_proba=min_proba)
    return_val = 100 * (balance_val - 10000) / 10000

    return {
        "label": label, "horizon": horizon, "atr_multiple": atr_multiple,
        "scaler": scaler, "model": model,
        "val_trades": len(trades_val), "val_return_pct": return_val,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--min-proba", type=float, default=0.65)
    args = parser.parse_args()

    print(f"Chargement {args.csv} ...")
    raw = load_mt5_csv(args.csv)
    print(f"{len(raw)} bougies H1 chargées.\n")

    candidates = []

    # --- Grille de recherche sur H1 : plusieurs horizons et seuils ATR ---
    print("Préparation des features H1 (indicateurs + KASP)...")
    feats_h1 = build_features(raw)
    feats_h1_kasp = build_kasp_indicator_features(feats_h1)

    horizons = [5, 10, 20]
    atr_multiples = [0.75, 1.0, 1.5]

    print(f"\n=== Recherche sur H1 : {len(horizons)*len(atr_multiples)} combinaisons ===")
    for horizon, atr_mult in itertools.product(horizons, atr_multiples):
        label = f"H1 | horizon={horizon} | atr_mult={atr_mult}"
        result = evaluate_combo(feats_h1_kasp, horizon, atr_mult, args.min_proba, label)
        candidates.append(result)
        print(f"{label:45s} -> validation: {result['val_trades']:4d} trades, "
              f"rendement {result['val_return_pct']:+7.2f}%")

    # --- Un essai en H4 (unité de temps différente) ---
    print(f"\n=== Recherche sur H4 (ré-échantillonné depuis H1) ===")
    raw_h4 = resample_to_timeframe(raw, "4h")
    feats_h4 = build_features(raw_h4)
    feats_h4_kasp = build_kasp_indicator_features(feats_h4)
    for horizon, atr_mult in [(5, 1.0), (10, 1.0), (10, 1.5)]:
        label = f"H4 | horizon={horizon} | atr_mult={atr_mult}"
        result = evaluate_combo(feats_h4_kasp, horizon, atr_mult, args.min_proba, label)
        candidates.append(result)
        print(f"{label:45s} -> validation: {result['val_trades']:4d} trades, "
              f"rendement {result['val_return_pct']:+7.2f}%")

    # --- Sélection du meilleur sur la VALIDATION uniquement ---
    # On exige un minimum de trades pour éviter de choisir un candidat
    # qui a eu 1 trade chanceux (pas statistiquement significatif).
    valid_candidates = [c for c in candidates if c["val_trades"] >= 10]
    if not valid_candidates:
        print("\nAucune combinaison n'a assez de trades en validation (minimum 10). "
              "Résultats trop peu fiables pour choisir un gagnant.")
        return

    best = max(valid_candidates, key=lambda c: c["val_return_pct"])

    print(f"\n{'='*70}")
    print(f"MEILLEURE COMBINAISON (choisie sur la VALIDATION, jamais le test) :")
    print(f"  {best['label']}")
    print(f"  Validation : {best['val_trades']} trades, rendement {best['val_return_pct']:+.2f}%")
    print(f"{'='*70}")

    # --- Évaluation finale, UNE SEULE FOIS, sur le vrai test set ---
    test_df = best["test_df"]
    period_start, period_end = test_df["time"].iloc[0], test_df["time"].iloc[-1]
    duration_days = (period_end - period_start).days
    print(f"\n=== ÉVALUATION FINALE sur le TEST SET (jamais vu, jamais utilisé pour choisir) ===")
    print(f"Période : {period_start} -> {period_end} (~{duration_days/365:.1f} ans)")

    trades_test, balance_test = backtest_dollars(test_df, best["scaler"], best["model"], min_proba=args.min_proba)
    if trades_test:
        wins = [t for t in trades_test if t > 0]
        print(f"Nombre de trades : {len(trades_test)}")
        print(f"Trades gagnants  : {len(wins)} ({100*len(wins)/len(trades_test):.1f}%)")
        print(f"Solde final      : {balance_test:.2f} $ (départ: 10000.00 $)")
        print(f"Rendement total  : {100*(balance_test-10000)/10000:+.2f} %")
    else:
        print("Aucun trade sur le test set avec cette combinaison.")


if __name__ == "__main__":
    main()
