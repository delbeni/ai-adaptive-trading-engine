"""
regime_detector.py
-------------------
RegimeDetector : classe le marché en régimes non-supervisés via HMM.
ImpulseEstimator : modèle supervisé (Gradient Boosting) qui estime
la probabilité d'impulsion haussière/baissière.
Split chronologique strict train/validation (anti-overfitting).
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
    future_close = df_features["close"].shift(-horizon)
    move = future_close - df_features["close"]
    threshold = df_features["atr_14"] * atr_multiple

    label = pd.Series("none", index=df_features.index)
    label[move > threshold] = "hausse"
    label[move < -threshold] = "baisse"
    return label


@dataclass
class ImpulseEstimator:
    scaler: StandardScaler = None
    model: object = None
    classes_: list = None

    def fit(self, df_features: pd.DataFrame, horizon: int = 10, atr_multiple: float = 1.0):
        labels = make_impulse_labels(df_features, horizon, atr_multiple)
        valid = labels.notna() & df_features[IMPULSE_FEATURE_COLS].notna().all(axis=1)
        valid &= (df_features.index < len(df_features) - horizon)

        X = df_features.loc[valid, IMPULSE_FEATURE_COLS].values
        y = labels.loc[valid].values

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
        print(f"[ImpulseEstimator] accuracy validation : {val_acc:.3f}")
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
