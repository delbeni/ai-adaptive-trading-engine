"""
backtest.py
-----------
Simule les décisions du modèle (regime_model + impulse_model) sur
l'historique, avec les mêmes règles que l'EA MT5 : entrée si P(hausse)
ou P(baisse) >= seuil, SL/TP basés sur l'ATR. Ne contacte PAS internet,
tourne entièrement en local, sur le test set réservé (jamais vu
pendant l'entraînement).

Usage :
    python backtest.py --csv XAUUSD_H1.csv --symbol XAUUSD
"""

import argparse
import os
import numpy as np
import pandas as pd

from regime_engine.features import load_mt5_csv, build_features
from regime_engine.regime_detector import RegimeDetector, ImpulseEstimator


def chronological_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def run_backtest(df_test, regime_model, impulse_model,
                  min_proba=0.65, atr_multiple_sl=1.5, rr_ratio=1.5,
                  risk_per_trade_pct=0.5, starting_balance=10000.0,
                  spread_points=0.30):
    balance = starting_balance
    equity_curve = [balance]
    trades = []

    proba = impulse_model.predict_proba(df_test)
    df = df_test.reset_index(drop=True)
    proba = proba.reset_index(drop=True)

    in_position = False
    entry_idx = None
    entry_price = None
    sl = None
    tp = None
    direction = None

    for i in range(len(df) - 1):
        row = df.iloc[i]
        p_hausse = proba.iloc[i].get("hausse", 0.0)
        p_baisse = proba.iloc[i].get("baisse", 0.0)

        if in_position:
            next_row = df.iloc[i + 1]
            hit_tp = hit_sl = False
            if direction == "achat":
                hit_sl = next_row["low"] <= sl
                hit_tp = next_row["high"] >= tp
            else:
                hit_sl = next_row["high"] >= sl
                hit_tp = next_row["low"] <= tp

            if hit_sl or hit_tp:
                exit_price = sl if hit_sl else tp

                pnl_price = (exit_price - entry_price) if direction == "achat" else (entry_price - exit_price)
                risk_amount = balance * (risk_per_trade_pct / 100)
                stop_distance = abs(entry_price - sl)
                lots_equivalent = risk_amount / stop_distance if stop_distance > 0 else 0
                pnl_dollars = pnl_price * lots_equivalent - spread_points * lots_equivalent

                balance += pnl_dollars
                trades.append({
                    "entry_time": df.iloc[entry_idx]["time"], "exit_time": next_row["time"],
                    "direction": direction, "entry": entry_price, "exit": exit_price,
                    "pnl": pnl_dollars, "balance_after": balance,
                })
                in_position = False

        if not in_position:
            decision = None
            if p_hausse >= min_proba and p_hausse > p_baisse:
                decision = "achat"
            elif p_baisse >= min_proba and p_baisse > p_hausse:
                decision = "vente"

            if decision is not None:
                atr = row["atr_14"]
                entry_price = row["close"]
                if decision == "achat":
                    sl = entry_price - atr * atr_multiple_sl
                    tp = entry_price + atr * atr_multiple_sl * rr_ratio
                else:
                    sl = entry_price + atr * atr_multiple_sl
                    tp = entry_price - atr * atr_multiple_sl * rr_ratio
                direction = decision
                entry_idx = i
                in_position = True

        equity_curve.append(balance)

    return pd.DataFrame(trades), pd.Series(equity_curve)


def print_stats(trades, equity, starting_balance):
    if trades.empty:
        print("Aucun trade déclenché sur la période testée (seuil de probabilité peut-être trop strict).")
        return

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    total_return_pct = 100 * (equity.iloc[-1] - starting_balance) / starting_balance

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd_pct = -100 * drawdown.min()

    print(f"\n=== Résultats du backtest ===")
    print(f"Nombre de trades       : {len(trades)}")
    print(f"Trades gagnants        : {len(wins)} ({100*len(wins)/len(trades):.1f}%)")
    print(f"Trades perdants        : {len(losses)} ({100*len(losses)/len(trades):.1f}%)")
    print(f"Gain moyen (trades +)  : {wins['pnl'].mean() if len(wins) else 0:.2f} $")
    print(f"Perte moyenne (trades -): {losses['pnl'].mean() if len(losses) else 0:.2f} $")
    print(f"Solde final             : {equity.iloc[-1]:.2f} $ (départ: {starting_balance:.2f} $)")
    print(f"Rendement total         : {total_return_pct:+.2f} %")
    print(f"Drawdown maximum        : {max_dd_pct:.2f} %")
    print(f"\nRappel : ceci simule sur le TEST SET réservé (jamais vu pendant l'entraînement).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--models", default="models")
    parser.add_argument("--min-proba", type=float, default=0.65)
    parser.add_argument("--balance", type=float, default=10000.0)
    args = parser.parse_args()

    print(f"Chargement {args.csv} ...")
    raw = load_mt5_csv(args.csv)
    feats = build_features(raw)
    print(f"{len(feats)} bougies après feature engineering.")

    _, _, test_df = chronological_split(feats)
    print(f"Backtest sur le test set réservé : {len(test_df)} bougies "
          f"({test_df['time'].iloc[0]} -> {test_df['time'].iloc[-1]})")

    print("Chargement des modèles entraînés...")
    regime_model = RegimeDetector.load(os.path.join(args.models, "regime_model"))
    impulse_model = ImpulseEstimator.load(os.path.join(args.models, "impulse_model"))

    trades, equity = run_backtest(
        test_df, regime_model, impulse_model,
        min_proba=args.min_proba, starting_balance=args.balance,
    )

    print_stats(trades, equity, args.balance)

    if not trades.empty:
        out_path = "backtest_trades.csv"
        trades.to_csv(out_path, index=False)
        print(f"\nDétail des trades sauvegardé dans : {out_path}")


if __name__ == "__main__":
    main()
