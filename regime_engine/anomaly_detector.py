"""
anomaly_detector.py
--------------------
Surveille si les conditions de marché actuelles ressemblent à ce que
le modèle a appris pendant l'entraînement, ou si elles sont "jamais vues"
(volatilité extrême inédite, comportement hors norme -- souvent lors
d'une actu majeure, d'un crash, ou d'un événement rare).

Utilise Isolation Forest (scikit-learn) : un algorithme qui apprend la
forme "normale" des données d'entraînement, puis donne un score
d'anomalie à toute nouvelle observation. Pas de coût, tourne sur les
mêmes données XAUUSD déjà utilisées pour regime_model/impulse_model.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

from .features import REGIME_FEATURE_COLS


@dataclass
class AnomalyDetector:
    contamination: float = 0.02
    scaler: StandardScaler = None
    model: object = None

    def fit(self, df_features: pd.DataFrame):
        X = df_features[REGIME_FEATURE_COLS].values
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=200,
            random_state=42,
        )
        self.model.fit(Xs)
        return self

    def check(self, df_features_row: pd.DataFrame) -> tuple[float, bool]:
        X = df_features_row[REGIME_FEATURE_COLS].values
        Xs = self.scaler.transform(X)
        score = float(self.model.score_samples(Xs)[-1])
        prediction = self.model.predict(Xs)[-1]
        is_anomaly = bool(prediction == -1)
        return score, is_anomaly

    def save(self, path_prefix: str):
        joblib.dump({"scaler": self.scaler, "model": self.model,
                     "contamination": self.contamination}, f"{path_prefix}.joblib")

    @classmethod
    def load(cls, path_prefix: str):
        data = joblib.load(f"{path_prefix}.joblib")
        obj = cls(contamination=data["contamination"])
        obj.scaler = data["scaler"]
        obj.model = data["model"]
        return obj
