"""
api.py
------
Sert les modèles entraînés via une API HTTP que l'EA MT5 interroge.
Déploiement : identique à tes backends FastAPI existants sur Render.
"""

import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from .features import build_features
from .regime_detector import RegimeDetector, ImpulseEstimator
from .risk_engine import RiskLimits

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
MIN_IMPULSE_PROBA = float(os.environ.get("MIN_IMPULSE_PROBA", "0.65"))

app = FastAPI(title="AI Adaptive Trading Engine")

_regime_model = None
_impulse_model = None


class Candle(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    tick_volume: float = 0.0
    spread: float = 0.0


class SignalRequest(BaseModel):
    symbol: str
    candles: list[Candle]


@app.on_event("startup")
def load_models():
    global _regime_model, _impulse_model
    try:
        _regime_model = RegimeDetector.load(os.path.join(MODEL_DIR, "regime_model"))
        _impulse_model = ImpulseEstimator.load(os.path.join(MODEL_DIR, "impulse_model"))
        print("Modèles chargés.")
    except FileNotFoundError:
        print("ATTENTION : modèles non trouvés. Entraîne-les d'abord avec train.py.")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": _regime_model is not None}


@app.post("/signal")
def get_signal(req: SignalRequest):
    if _regime_model is None or _impulse_model is None:
        return {"error": "Modèles non chargés côté serveur. Entraîne-les d'abord."}

    df = pd.DataFrame([c.dict() for c in req.candles])
    df["time"] = pd.to_datetime(df["time"])
    feats = build_features(df)

    if feats.empty:
        return {"error": "Pas assez de bougies pour calculer les features (min ~150 requis)."}

    last = feats.iloc[[-1]]
    regime = _regime_model.predict(feats).iloc[-1]
    proba = _impulse_model.predict_proba(last).iloc[-1]

    proba_hausse = float(proba.get("hausse", 0.0))
    proba_baisse = float(proba.get("baisse", 0.0))
    proba_none = float(proba.get("none", 0.0))

    decision = "aucun_trade"
    confidence_ok = False
    if proba_hausse >= MIN_IMPULSE_PROBA and proba_hausse > proba_baisse:
        decision = "achat"
        confidence_ok = True
    elif proba_baisse >= MIN_IMPULSE_PROBA and proba_baisse > proba_hausse:
        decision = "vente"
        confidence_ok = True

    return {
        "symbol": req.symbol,
        "regime": regime,
        "proba_hausse": round(proba_hausse, 4),
        "proba_baisse": round(proba_baisse, 4),
        "proba_none": round(proba_none, 4),
        "decision": decision,
        "confidence_ok": confidence_ok,
    }
