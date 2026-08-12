"""
api.py
------
Sert les modèles entraînés (RegimeDetector + ImpulseEstimator) via une
API HTTP que l'EA MT5 interroge avec WebRequest().

Déploiement : identique à tes backends Flask/FastAPI existants sur Render.

Lancement local :
    uvicorn regime_engine.api:app --host 0.0.0.0 --port 8000

Endpoint principal : POST /signal
Payload attendu (dernières bougies, la plus récente en dernier) :
{
  "symbol": "XAUUSD",
  "candles": [
    {"time": "...", "open": 0, "high": 0, "low": 0, "close": 0,
     "tick_volume": 0, "spread": 0},
    ...  (au moins 150 bougies pour que les features rolling soient valides)
  ]
}

Réponse :
{
  "regime": "tendance_haussiere",
  "proba_hausse": 0.78,
  "proba_baisse": 0.14,
  "proba_none": 0.08,
  "decision": "achat",          # achat / vente / aucun_trade
  "confidence_ok": true          # >= seuil configuré côté risk engine
}
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
        print("ATTENTION : modèles non trouvés. Entraîne-les d'abord avec train.py "
              "et place-les dans le dossier 'models/'.")


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
