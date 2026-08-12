# AI Adaptive Trading Engine

Implémentation de l'architecture décrite dans ta conversation :

```
Détection de régime  ->  Sélection de stratégie / estimation d'edge  ->  Filtre de risque  ->  Exécution MT5
```

## Structure

```
ai_adaptive_bot/
├── regime_engine/
│   ├── features.py         # feature engineering (momentum, volatilité, ATR, ADX, sweeps, sessions...)
│   ├── regime_detector.py  # RegimeDetector (HMM) + ImpulseEstimator (Gradient Boosting)
│   ├── risk_engine.py      # Moteur de sécurité (Python, pour tests/backtest)
│   └── api.py               # API FastAPI servie à l'EA MT5
├── mql5/
│   └── AI_Adaptive_EA.mq5  # EA d'exécution. Ré-implémente le moteur de risque EN DUR
├── train.py                  # Entraînement avec split chronologique strict
└── requirements.txt
```

## Pourquoi cette architecture (et pas tout coder en MQL5)

Comme discuté : MT5/MQL5 est excellent pour l'exécution mais mauvais pour l'entraînement
de modèles (pas de scikit-learn, pas de vrai ML). Donc :

- **Python = intelligence** (régime + probabilité d'impulsion), entraîné hors-ligne sur
  tes années de données MT5 exportées, servi via une API (comme ton FastAPI existant sur Render).
- **MQL5 = exécution uniquement**, avec son propre moteur de risque codé en dur dans
  `AI_Adaptive_EA.mq5`. Défense en profondeur : même si l'API tombe, répond n'importe
  quoi, ou est compromise, l'EA n'exécute JAMAIS un trade qui dépasse tes limites.

## Anti-overfitting (le piège que tu avais identifié)

`train.py` applique un split **chronologique strict** (jamais de shuffle aléatoire) :
- ~70% entraînement
- ~15% validation (choix des hyperparamètres)
- ~15% test **réservé**, évalué une seule fois à la fin

Si l'accuracy du test est nettement inférieure à celle de validation → le modèle a
mémorisé le passé plutôt qu'appris un vrai edge. Il faut alors simplifier le modèle
(moins d'arbres/profondeur) ou augmenter la donnée, pas l'inverse.

## Mise en route

### 1. Exporter tes données MT5

Terminal MT5 → Affichage → Boîte à outils → Historique des cotations → exporter en CSV
(ou clic droit sur le graphique → Enregistrer sous). Idéal : XAUUSD, H1 ou M15, le plus
de données possible (10+ ans si dispo chez ton broker/data vendor).

### 2. Entraîner

```bash
pip install -r requirements.txt --break-system-packages
python train.py --csv data/XAUUSD_H1.csv --symbol XAUUSD --horizon 10 --atr-multiple 1.0
```

Ça crée `models/regime_model.joblib` et `models/impulse_model.joblib`.

### 3. Lancer l'API en local pour tester

```bash
uvicorn regime_engine.api:app --reload --port 8000
curl http://localhost:8000/health
```

### 4. Déployer sur Render

Même procédé que tes backends Flask/FastAPI existants (trade copier, KaspTerminal) :
- `Start Command`: `uvicorn regime_engine.api:app --host 0.0.0.0 --port $PORT`
- Variables d'env : `MODEL_DIR=models`, `MIN_IMPULSE_PROBA=0.65`
- Committer le dossier `models/` (ou le régénérer au build si tu préfères ne pas
  versionner de binaires — à voir selon la taille).

### 5. Brancher l'EA MT5

- Ouvrir `mql5/AI_Adaptive_EA.mq5` dans MetaEditor, compiler.
- Dans MT5 : Outils → Options → Expert Advisors → cocher "Autoriser WebRequest" et
  ajouter l'URL de ton API Render.
- Attacher l'EA au graphique XAUUSD, régler les inputs (risque max, seuils...).

## Ce qui manque encore avant du réel (à ne pas sauter)

1. **Forward test en démo** plusieurs semaines/mois avant tout compte réel — exactement
   comme tu le disais : la validation historique ne garantit rien sur le futur.
2. **Mesurer le coût réel d'exécution** (spread/slippage/latence de ton broker) et
   vérifier que l'edge survit après ces coûts — c'est actuellement pris en compte
   uniquement via les seuils de risque, pas encore mesuré empiriquement.
3. **Tick data** : la version actuelle tourne sur bougies (H1/M15). Le détecteur
   d'impulsion "tick-level" que tu décrivais en fin de conversation (accélération des
   ticks, absorption, micro-structure) est une V2 — la structure du code (`features.py`,
   `ImpulseEstimator`) est faite pour être étendue dans ce sens sans tout réécrire.
4. **Ré-entraînement contrôlé** : ne jamais laisser le modèle se ré-entraîner seul en
   continu sur les résultats récents sans supervision (le piège que tu avais identifié).
   Ré-entraînement = étape manuelle, avec revue du test set avant remplacement du modèle
   en prod.
