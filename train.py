"""
train.py
--------
Entraîne les modèles à partir d'un export MT5 (CSV) et sauvegarde
dans models/. Split chronologique strict train/validation/test réservé.

Usage :
    python train.py --csv data/XAUUSD_H1.csv --symbol XAUUSD
"""

import argparse
import os
import pandas as pd

from regime_engine.features import load_mt5_csv, build_features
from regime_engine.regime_detector import RegimeDetector, ImpulseEstimator, make_impulse_labels
from regime_engine.anomaly_detector import AnomalyDetector


def chronological_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Chemin vers l'export MT5 CSV")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--atr-multiple", type=float, default=1.0)
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Chargement {args.csv} ...")
    raw = load_mt5_csv(args.csv)
    print(f"{len(raw)} bougies chargées.")

    feats = build_features(raw)
    print(f"{len(feats)} bougies après feature engineering (NaN initiaux retirés).")

    train_df, val_df, test_df = chronological_split(feats)
    print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test (réservé): {len(test_df)}")

    print("\nEntraînement du détecteur de régime...")
    regime_model = RegimeDetector(n_regimes=5).fit(train_df)
    regime_model.save(os.path.join(args.out, "regime_model"))
    train_df = train_df.copy()
    train_df["regime"] = regime_model.predict(train_df)
    print(train_df["regime"].value_counts())

    print("\nEntraînement de l'estimateur d'impulsion...")
    impulse_model = ImpulseEstimator().fit(train_df, horizon=args.horizon, atr_multiple=args.atr_multiple)
    impulse_model.save(os.path.join(args.out, "impulse_model"))

    print("\nEntraînement du détecteur d'anomalie...")
    anomaly_model = AnomalyDetector(contamination=0.02).fit(train_df)
    anomaly_model.save(os.path.join(args.out, "anomaly_model"))
    print("Détecteur d'anomalie entraîné et sauvegardé.")

    print("\n=== Évaluation sur le TEST SET réservé (jamais vu pendant l'entraînement) ===")
    test_labels = make_impulse_labels(test_df, args.horizon, args.atr_multiple)
    valid = test_labels.notna()
    proba = impulse_model.predict_proba(test_df.loc[valid])
    pred = proba.idxmax(axis=1)
    acc = (pred.values == test_labels.loc[valid].values).mean()
    print(f"Accuracy test (hors échantillon, non touché pendant le tuning) : {acc:.3f}")
    print("\nRappel : si cette accuracy est nettement inférieure à celle de validation,")
    print("c'est un signal d'overfitting -> réduire la complexité du modèle ou ajouter des données.")

    print(f"\nModèles sauvegardés dans {args.out}/")


if __name__ == "__main__":
    main()
