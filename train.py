"""
train.py
--------
Entraîne les deux modèles à partir d'un export MT5 (CSV) et sauvegarde
dans models/. Respecte le split anti-overfitting discuté :

    10 ans -> entraînement (train)
    2 ans  -> validation (utilisé pour choisir les hyperparamètres)
    2-3 ans -> test totalement inconnu (à N'UTILISER QU'UNE FOIS, à la fin)

Usage :
    python train.py --csv data/XAUUSD_H1.csv --symbol XAUUSD

Export MT5 : clic droit sur le graphique -> "Exporter les données" ->
             ou via Terminal -> Historique des cotations -> Exporter.
"""

import argparse
import os
import pandas as pd

from regime_engine.features import load_mt5_csv, build_features
from regime_engine.regime_detector import RegimeDetector, ImpulseEstimator, make_impulse_labels


def chronological_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Chemin vers l'export MT5 CSV")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--horizon", type=int, default=10, help="Bougies dans le futur pour définir l'impulsion")
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

    # --- Régime (non-supervisé, entraîné sur train uniquement) ---
    print("\nEntraînement du détecteur de régime...")
    regime_model = RegimeDetector(n_regimes=5).fit(train_df)
    regime_model.save(os.path.join(args.out, "regime_model"))
    train_df = train_df.copy()
    train_df["regime"] = regime_model.predict(train_df)
    print(train_df["regime"].value_counts())

    # --- Impulsion (supervisé, split interne train/val déjà géré dans fit()) ---
    print("\nEntraînement de l'estimateur d'impulsion...")
    impulse_model = ImpulseEstimator().fit(train_df, horizon=args.horizon, atr_multiple=args.atr_multiple)
    impulse_model.save(os.path.join(args.out, "impulse_model"))

    # --- Évaluation FINALE sur le test set jamais vu (une seule fois) ---
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
