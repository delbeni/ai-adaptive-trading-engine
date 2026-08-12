"""
risk_engine.py
--------------
Moteur de sécurité indépendant du modèle IA. C'est LUI qui a le dernier mot,
jamais l'IA. Reprend exactement les garde-fous listés dans ta conversation :

- risque max par trade
- perte max quotidienne
- drawdown max
- nombre max de positions simultanées
- spread max autorisé
- slippage max toléré
- coupure automatique si comportement anormal (ex: trop de trades / minute,
  trop de pertes consécutives)

Ce module est pur Python, sans dépendance MT5, pour pouvoir être testé
unitairement et réutilisé côté API. Le mapping 1:1 vers MQL5 est fourni
dans mql5/AI_Adaptive_EA.mq5 (mêmes règles, appliquées côté EA aussi —
défense en profondeur : même si l'API répond n'importe quoi, l'EA
ne dépasse jamais ces limites).
"""

from dataclasses import dataclass, field
from datetime import datetime, date


@dataclass
class RiskLimits:
    max_risk_per_trade_pct: float = 0.5       # % du capital
    max_daily_loss_pct: float = 2.0            # % du capital
    max_drawdown_pct: float = 8.0               # % du capital (depuis le sommet)
    max_open_positions: int = 3
    max_spread_points: float = 30.0
    max_slippage_points: float = 15.0
    max_consecutive_losses: int = 4              # coupure si atteint
    min_impulse_probability: float = 0.65        # seuil de décision de l'IA
    max_trades_per_hour: int = 6                 # anti-comportement anormal


@dataclass
class AccountState:
    balance: float
    equity: float
    day_start_equity: float
    equity_peak: float
    open_positions: int = 0
    consecutive_losses: int = 0
    trades_this_hour: int = 0
    current_date: date = field(default_factory=date.today)


class RiskEngine:
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.halted = False
        self.halt_reason = ""

    def check_new_trade(self, account: AccountState, spread_points: float,
                         estimated_slippage_points: float,
                         impulse_probability: float) -> tuple[bool, str]:
        """Retourne (autorisé, raison_si_refus). Doit être appelé avant
        CHAQUE exécution, sans exception."""

        if self.halted:
            return False, f"Système à l'arrêt : {self.halt_reason}"

        if impulse_probability < self.limits.min_impulse_probability:
            return False, f"Probabilité d'edge insuffisante ({impulse_probability:.2f} < {self.limits.min_impulse_probability})"

        if spread_points > self.limits.max_spread_points:
            return False, f"Spread trop élevé ({spread_points} > {self.limits.max_spread_points})"

        if estimated_slippage_points > self.limits.max_slippage_points:
            return False, f"Slippage estimé trop élevé ({estimated_slippage_points} > {self.limits.max_slippage_points})"

        if account.open_positions >= self.limits.max_open_positions:
            return False, "Nombre max de positions ouvertes atteint"

        daily_loss_pct = 100 * (account.day_start_equity - account.equity) / account.day_start_equity
        if daily_loss_pct >= self.limits.max_daily_loss_pct:
            self._halt(f"Perte quotidienne max atteinte ({daily_loss_pct:.2f}%)")
            return False, self.halt_reason

        drawdown_pct = 100 * (account.equity_peak - account.equity) / account.equity_peak
        if drawdown_pct >= self.limits.max_drawdown_pct:
            self._halt(f"Drawdown max atteint ({drawdown_pct:.2f}%)")
            return False, self.halt_reason

        if account.consecutive_losses >= self.limits.max_consecutive_losses:
            self._halt(f"{account.consecutive_losses} pertes consécutives (comportement anormal potentiel)")
            return False, self.halt_reason

        if account.trades_this_hour >= self.limits.max_trades_per_hour:
            return False, "Nombre max de trades/heure atteint (protection sur-trading)"

        return True, "OK"

    def position_size(self, account: AccountState, entry_price: float,
                       stop_loss_price: float, point_value: float) -> float:
        """Calcule la taille de position (lots) pour respecter le risque max/trade.
        point_value = valeur d'1 point de prix pour 1 lot (dépend du symbole)."""
        risk_amount = account.balance * (self.limits.max_risk_per_trade_pct / 100)
        stop_distance_points = abs(entry_price - stop_loss_price)
        if stop_distance_points <= 0 or point_value <= 0:
            return 0.0
        lots = risk_amount / (stop_distance_points * point_value)
        return max(round(lots, 2), 0.0)

    def _halt(self, reason: str):
        self.halted = True
        self.halt_reason = reason

    def reset_daily(self, account: AccountState):
        """À appeler une fois par jour (nouvelle session)."""
        account.day_start_equity = account.equity
        account.trades_this_hour = 0
        # NB: on ne reset PAS `halted` automatiquement -> nécessite une
        # validation manuelle si le système s'est arrêté pour drawdown/pertes.
