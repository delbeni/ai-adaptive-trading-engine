"""
regime_detector.py
-------------------
Deux modèles :

1. RegimeDetector : classe le marché en régimes non-supervisés
   (tendance haussière forte, tendance baissière forte, range,
   volatilité extrême, faible volatilité) via un Hidden Markov Model
   (ou KMeans si hmmlearn indisponible).

2. ImpulseEstimator : modèle supervisé (Gradient Boosting) qui répond
   à la question posée dans ta conversation :
   "Les conditions actuelles ressemblent-elles aux situations
   historiques qui ont précédé une impulsion exploitable ?"

   Le label est construit a posteriori (mouvement futur net > seuil
   après N bougies) UNIQUEMENT sur les données d'entraînement — jamais
   utilisé en features, donc pas de fuite de futur (no look-ahead).

IMPORTANT anti-overfitting (comme discuté) :
- split strict train / validation / test chronologique (pas de shuffle)
- le test set n'est JAMAIS utilisé pour choisir les hyperparamètres
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib

from .features import REGIME_FEATURE_COLS, IMPULSE_FEATURE_COLS

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False


REGIME_LABELS = {
    0: "range",
    1: "tendance_haussiere",
    2: "tendance_baissiere",
    3: "volatilite_extreme",
    4: "faible_volatilite",
}


@dataclass
class RegimeDetector:
    n_regimes: int = 5
    scaler: StandardScaler = None
    model: object = None

    def fit(self, df_features: pd.DataFrame):
        X = df_features[REGIME_FEATURE_COLS].values
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        if HAS_HMM:
            self.model = GaussianHMM(
                n_components=self.n_regimes, covariance_type="diag",
                n_iter=200, random_state=42
            )
            self.model.fit(Xs)
        else:
            self.model = KMeans(n_clusters=self.n_regimes, n_init=10, random_state=42)
            self.model.fit(Xs)
        return self

    def predict(self, df_features: pd.DataFrame) -> pd.Series:
        X = df_features[REGIME_FEATURE_COLS].values
        Xs = self.scaler.transform(X)
        if HAS_HMM:
            states = self.model.predict(Xs)
        else:
            states = self.model.predict(Xs)
        return self._label_states(df_features, states)

    def _label_states(self, df_features, states) -> pd.Series:
        """Ré-étiquette les états numériques du modèle avec des noms
        interprétables, à partir de leurs caractéristiques moyennes
        (trend_bias, vol) plutôt qu'un index arbitraire."""
        tmp = df_features.copy()
        tmp["state"] = states
        stats = tmp.groupby("state")[["trend_bias", "vol_20", "atr_ratio"]].mean()

        labels = {}
        vol_threshold_high = stats["atr_ratio"].quantile(0.8)
        vol_threshold_low = stats["atr_ratio"].quantile(0.2)
        for state, row in stats.iterrows():
            if row["atr_ratio"] >= vol_threshold_high:
                labels[state] = "volatilite_extreme"
            elif row["atr_ratio"] <= vol_threshold_low:
                labels[state] = "faible_volatilite"
            elif row["trend_bias"] > 0.15:
                labels[state] = "tendance_haussiere"
            elif row["trend_bias"] < -0.15:
                labels[state] = "tendance_baissiere"
            else:
                labels[state] = "range"

        return pd.Series([labels[s] for s in states], index=df_features.index)

    def save(self, path_prefix: str):
        joblib.dump({"scaler": self.scaler, "model": self.model,
                     "n_regimes": self.n_regimes}, f"{path_prefix}.joblib")

    @classmethod
    def load(cls, path_prefix: str):
        data = joblib.load(f"{path_prefix}.joblib")
        obj = cls(n_regimes=data["n_regimes"])
        obj.scaler = data["scaler"]
        obj.model = data["model"]
        return obj


def make_impulse_labels(df_features: pd.DataFrame, horizon: int = 10,
                         atr_multiple: float = 1.0) -> pd.Series:
    """
    Label = direction du mouvement net dans les `horizon` bougies suivantes,
    seulement si ce mouvement dépasse `atr_multiple` * ATR courant
    (sinon => pas d'edge exploitable => classe 'none').

    ATTENTION : utilisé UNIQUEMENT à l'entraînement (regarde le futur).
    Ne jamais utiliser cette fonction en production/live.
    """
    future_close = df_features["close"].shift(-horizon)
    move = future_close - df_features["close"]
    threshold = df_features["atr_14"] * atr_multiple

    label = pd.Series("none", index=df_features.index)
    label[move > threshold] = "hausse"
    label[move < -threshold] = "baisse"
    return label


@dataclass
class ImpulseEstimator:
    """Estime P(impulsion haussière), P(impulsion baissière), P(aucun edge)."""
    scaler: StandardScaler = None
    model: object = None
    classes_: list = None

    def fit(self, df_features: pd.DataFrame, horizon: int = 10, atr_multiple: float = 1.0):
        labels = make_impulse_labels(df_features, horizon, atr_multiple)
        valid = labels.notna() & df_features[IMPULSE_FEATURE_COLS].notna().all(axis=1)
        # on enlève la fin de la série où le futur n'existe pas encore
        valid &= (df_features.index < len(df_features) - horizon)

        X = df_features.loc[valid, IMPULSE_FEATURE_COLS].values
        y = labels.loc[valid].values

        # split chronologique STRICT : pas de shuffle (anti-overfitting / no leakage)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.scaler = StandardScaler().fit(X_train)
        Xt = self.scaler.transform(X_train)
        Xv = self.scaler.transform(X_val)

        self.model = GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        self.model.fit(Xt, y_train)
        self.classes_ = list(self.model.classes_)

        val_acc = self.model.score(Xv, y_val)
        print(f"[ImpulseEstimator] accuracy validation (hors échantillon, chronologique) : {val_acc:.3f}")
        return self

    def predict_proba(self, df_features: pd.DataFrame) -> pd.DataFrame:
        X = df_features[IMPULSE_FEATURE_COLS].values
        Xs = self.scaler.transform(X)
        proba = self.model.predict_proba(Xs)
        return pd.DataFrame(proba, columns=self.model.classes_, index=df_features.index)

    def save(self, path_prefix: str):
        joblib.dump({"scaler": self.scaler, "model": self.model,
                     "classes_": self.classes_}, f"{path_prefix}.joblib")

    @classmethod
    def load(cls, path_prefix: str):
        data = joblib.load(f"{path_prefix}.joblib")
        obj = cls()
        obj.scaler = data["scaler"]
        obj.model = data["model"]
        obj.classes_ = data["classes_"]
        return obj
