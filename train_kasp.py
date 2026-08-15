"""
train_kasp.py
--------------
Entraîne le modèle avec les indicateurs de tes propres fichiers MQL5
(KASP_FusionOsc : Heikin Ashi MTF ; KASP_TradingFusion : SuperTrend,
SWDL, TrendTrader) ajoutés aux features déjà utilisées. Compare
objectivement avec train.py (indicateurs génériques seuls, +5.75% sur
le test set 2024-05-02 -> 2026-08-14).

Usage :
    python train_kasp.py --csv XAUUSD_H1_15ans.csv --symbol XAUUSD
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

from regime_engine.features import load_mt5_csv, build_features, IMPULSE_FEATURE_COLS
from regime_engine.kasp_indicators_features import build_kasp_indicator_features, KASP_FEATURE_COLS
from regime_engine.regime_detector import RegimeDetector, make_impulse_labels
from regime_engine.anomaly_detector import AnomalyDetector

ALL_FEATURE_COLS = IMPULSE_FEATURE_COLS + KASP_FEATURE_COLS


def chronological_split(df, train_frac=0.70, val_frac=0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def fit_impulse_kasp(train_df, horizon=10, atr_multiple=1.0):
    labels = make_impulse_labels(train_df, horizon, atr_multiple)
    valid = labels.notna() & train_df[ALL_FEATURE_COLS].notna().all(axis=1)
    valid &= (train_df.index < len(train_df) - horizon)

    X = train_df.loc[valid, ALL_FEATURE_COLS].values
    y = labels.loc[valid].values

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    scaler = StandardScaler().fit(X_train)
    Xt = scaler.transform(X_train)
    Xv = scaler.transform(X_val)

    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    model.fit(Xt, y_train)
    val_acc = model.score(Xv, y_val)
    print(f"[ImpulseEstimator+KASP] accuracy validation : {val_acc:.3f}")

    return scaler, model


def backtest_dollars(test_df, scaler, model, min_proba=0.65, atr_mult_sl=1.5,
                      rr_ratio=1.5, risk_pct=0.5, starting_balance=10000.0, spread=0.30):
    X = test_df[ALL_FEATURE_COLS].values
    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)
    classes = list(model.classes_)
    proba_df = pd.DataFrame(proba, columns=classes, index=test_df.index).reset_index(drop=True)
    df = test_df.reset_index(drop=True)

    balance = starting_balance
    trades = []
    in_position = False

    for i in range(len(df) - 1):
        p_hausse = proba_df.iloc[i].get("hausse", 0.0)
        p_baisse = proba_df.iloc[i].get("baisse", 0.0)

        if in_position:
            next_row = df.iloc[i + 1]
            hit_sl = hit_tp = False
            if direction == "achat":
                hit_sl = next_row["low"] <= sl
                hit_tp = next_row["high"] >= tp
            else:
                hit_sl = next_row["high"] >= sl
                hit_tp = next_row["low"] <= tp
            if hit_sl or hit_tp:
                exit_price = sl if hit_sl else tp
                pnl_price = (exit_price - entry_price) if direction == "achat" else (entry_price - exit_price)
                risk_amount = balance * (risk_pct / 100)
                stop_dist = abs(entry_price - sl)
                lots = risk_amount / stop_dist if stop_dist > 0 else 0
                pnl = pnl_price * lots - spread * lots
                balance += pnl
                trades.append(pnl)
                in_position = False

        if not in_position:
            row = df.iloc[i]
            decision = None
            if p_hausse >= min_proba and p_hausse > p_baisse:
                decision = "achat"
            elif p_baisse >= min_proba and p_baisse > p_hausse:
                decision = "vente"
            if decision:
                atr = row["atr_14"]
                entry_price = row["close"]
                if decision == "achat":
                    sl = entry_price - atr * atr_mult_sl
                    tp = entry_price + atr * atr_mult_sl * rr_ratio
                else:
                    sl = entry_price + atr * atr_mult_sl
                    tp = entry_price - atr * atr_mult_sl * rr_ratio
                direction = decision
                in_position = True

    return trades, balance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--atr-multiple", type=float, default=1.0)
    parser.add_argument("--out", default="models_kasp")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Chargement {args.csv} ...")
    raw = load_mt5_csv(args.csv)
    print(f"{len(raw)} bougies chargées.")

    feats = build_features(raw)
    print(f"{len(feats)} bougies après feature engineering de base.")

    print("Calcul des indicateurs KASP (Heikin Ashi MTF, SuperTrend, SWDL, TrendTrader)...")
    feats_kasp = build_kasp_indicator_features(feats)
    print(f"{len(feats_kasp)} bougies après ajout des indicateurs KASP.")

    train_df, val_df, test_df = chronological_split(feats_kasp)
    print(f"\nTrain: {len(train_df)} | Validation: {len(val_df)} | Test (réservé): {len(test_df)}")
    print(f"Période test : {test_df['time'].iloc[0]} -> {test_df['time'].iloc[-1]}")
    duration_days = (test_df['time'].iloc[-1] - test_df['time'].iloc[0]).days
    print(f"Durée testée : {duration_days} jours (~{duration_days/365:.1f} ans)")

    print("\nEntraînement du détecteur de régime...")
    regime_model = RegimeDetector(n_regimes=5).fit(train_df)
    regime_model.save(os.path.join(args.out, "regime_model"))

    print("\nEntraînement de l'estimateur d'impulsion AVEC indicateurs KASP...")
    scaler, model = fit_impulse_kasp(train_df, horizon=args.horizon, atr_multiple=args.atr_multiple)
    joblib.dump({"scaler": scaler, "model": model, "classes_": list(model.classes_),
                 "feature_cols": ALL_FEATURE_COLS},
                os.path.join(args.out, "impulse_model_kasp.joblib"))

    print("\nEntraînement du détecteur d'anomalie...")
    anomaly_model = AnomalyDetector(contamination=0.02).fit(train_df)
    anomaly_model.save(os.path.join(args.out, "anomaly_model"))

    print("\n=== Évaluation ACCURACY sur le TEST SET réservé (jamais vu) ===")
    test_labels = make_impulse_labels(test_df, args.horizon, args.atr_multiple)
    valid = test_labels.notna() & test_df[ALL_FEATURE_COLS].notna().all(axis=1)
    Xt = scaler.transform(test_df.loc[valid, ALL_FEATURE_COLS].values)
    pred = model.predict(Xt)
    acc = (pred == test_labels.loc[valid].values).mean()
    print(f"Accuracy test (indicateurs génériques + KASP) : {acc:.3f}")
    print(f"(Rappel : indicateurs génériques seuls donnaient 0.404 sur le même type de période)")

    print("\n=== Évaluation EN ARGENT RÉEL (backtest) ===")
    print(f"Période testée : {test_df['time'].iloc[0]} -> {test_df['time'].iloc[-1]} "
          f"(~{duration_days/365:.1f} ans)")
    trades, final_balance = backtest_dollars(test_df, scaler, model, min_proba=0.65)
    if trades:
        wins = [t for t in trades if t > 0]
        print(f"Nombre de trades : {len(trades)}")
        print(f"Trades gagnants  : {len(wins)} ({100*len(wins)/len(trades):.1f}%)")
        print(f"Solde final      : {final_balance:.2f} $ (départ: 10000.00 $)")
        print(f"Rendement total  : {100*(final_balance-10000)/10000:+.2f} %")
        print(f"(Rappel : indicateurs génériques seuls avaient donné +5.75%, 61.9% de reussite, 21 trades)")
    else:
        print("Aucun trade déclenché sur cette période avec ce seuil.")

    print(f"\nModèles sauvegardés dans {args.out}/")


if __name__ == "__main__":
    main()
