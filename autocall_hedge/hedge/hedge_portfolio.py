"""
hedge_portfolio.py
==================
Moteur de réplication dynamique delta-gamma-vega d'un Athena.

Stratégie de couverture :
  1. Vega hedge   → Straddle ATM (1M ou T_remaining/2 de maturité)
                    Quantité q_straddle = -Vega_autocall / Vega_straddle
  2. Delta hedge  → Sous-jacent (actions)
                    Quantité q_delta = Delta_autocall_résiduel
                    (après neutralisation du delta résiduel du straddle)
  3. Gamma        → Le straddle contribue aussi au gamma hedge
                    Gamma résiduel = Gamma_autocall - q_straddle * Gamma_straddle

À chaque rebalancement :
  - On recalcule les Greeks
  - On ajuste les quantités
  - On calcule le P&L du portefeuille et l'erreur de réplication

Convention comptable :
  - Le portefeuille est initialisé à V0 (financement par emprunt/prêt)
  - P&L = Valeur portefeuille - Valeur autocall (erreur de réplication cumulée)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List

from ..core.athena_pricer import AthenaProduct, MCConfig, price_athena
from ..core.greeks import Greeks, compute_greeks
from ..hedge.straddle import price_straddle_atm, StraddleGreeks


@dataclass
class HedgeState:
    """État du portefeuille de couverture à une date donnée."""
    date: pd.Timestamp
    spot: float
    sigma: float
    T_remaining: float

    # Greeks de l'autocall
    greeks: Greeks

    # Instruments de couverture
    q_underlying: float          # Quantité de sous-jacent (delta hedge)
    q_straddle: float            # Quantité de straddle (vega hedge)
    straddle: StraddleGreeks     # Greeks du straddle unitaire

    # Valeurs du portefeuille
    autocall_mtm: float          # MtM de l'autocall
    hedge_value: float           # Valeur mark-to-market du portefeuille de couverture
    cash: float                  # Compte de financement (bond)

    # Erreurs
    replication_error: float     # hedge_value - autocall_mtm
    gamma_residual: float        # Gamma non hedgé


@dataclass
class HedgePortfolio:
    """
    Portefeuille de réplication dynamique.
    Stocke l'historique complet des états de couverture.
    """
    product: AthenaProduct
    mc_config: MCConfig
    r: float                                  # Taux sans risque
    straddle_maturity: float = 3.0 / 12      # Maturité du straddle hedge (3M — plus stable qu'1M)
    rebalance_freq: int = 5                   # Rebalancement tous les N jours ouvrés

    history: List[HedgeState] = field(default_factory=list)

    def run(
        self,
        prices: pd.Series,
        vol_rolling: pd.Series,
        S0_product: float,
        pricing_start: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Lance la simulation de réplication dynamique sur la time series.

        Parameters
        ----------
        prices         : Série de prix historiques (index DatetimeIndex)
        vol_rolling    : Série de vol rolling calibrée
        S0_product     : Spot au moment de l'émission du produit (= strike si strike=1.0)
                         Ce niveau est FIXE pendant toute la vie du produit.
        pricing_start  : Date de début du backtest de couverture

        Returns
        -------
        DataFrame avec l'historique complet
        """
        # Niveau de strike absolu fixé à l'émission — NE CHANGE PAS
        strike_abs = self.product.strike * S0_product
        # Filtrage sur la période de pricing
        prices = prices[prices.index >= pricing_start]
        vol_rolling = vol_rolling[vol_rolling.index >= pricing_start].ffill()

        # Dates de rebalancement
        all_dates = prices.index
        rebal_dates = all_dates[::self.rebalance_freq]

        # Initialisation
        t0 = rebal_dates[0]
        S_init = float(prices.iloc[0])
        sigma_init = float(vol_rolling.asof(t0))
        T_init = float(self.product.maturity_years)

        # Prix initial de l'autocall
        g0 = compute_greeks(
            self.product, S_init, self.r, sigma_init, T_init, self.mc_config,
            strike_level_abs=strike_abs,
        )
        V0 = g0.price

        # Straddle initial
        straddle_T = min(self.straddle_maturity, T_init * 0.9)
        straddle0 = price_straddle_atm(S_init, self.r, sigma_init, straddle_T)

        # Quantités initiales
        q_straddle = -g0.vega / straddle0.vega if abs(straddle0.vega) > 1e-10 else 0.0
        q_underlying = -(g0.delta + q_straddle * straddle0.delta)

        # ── Portefeuille AUTO-FINANCÉ ─────────────────────────────────────────
        # Convention : le bureau émet l'autocall et reçoit V0 du client.
        # Il utilise ce V0 pour financer les positions de couverture initiales.
        # Bond (compte de financement) = V0 - coût des positions achetées.
        # Si bond > 0 → on a du cash résiduel placé au taux r.
        # Si bond < 0 → on emprunte au taux r.
        # À chaque rebalancement, le trade marginal (achat/vente) est financé
        # par emprunt/prêt sans injection de capital externe.
        hedge_cost = q_underlying * S_init + q_straddle * straddle0.price
        bond = V0 - hedge_cost   # Compte obligataire (peut être négatif = emprunt)

        gamma_residual = g0.gamma + q_straddle * straddle0.gamma

        state0 = HedgeState(
            date=t0,
            spot=S_init,
            sigma=sigma_init,
            T_remaining=T_init,
            greeks=g0,
            q_underlying=q_underlying,
            q_straddle=q_straddle,
            straddle=straddle0,
            autocall_mtm=V0,
            hedge_value=q_underlying * S_init + q_straddle * straddle0.price + bond,
            cash=bond,
            replication_error=0.0,
            gamma_residual=gamma_residual,
        )
        self.history.append(state0)

        print(f"[Hedge] Init | S={S_init:.2f} | σ={sigma_init:.1%} | V0={V0:.2f}")
        print(f"         Δ={g0.delta:.4f} | Γ={g0.gamma:.6f} | ν={g0.vega:.4f}")
        print(f"         q_underlying={q_underlying:.4f} | q_straddle={q_straddle:.4f} | bond={bond:.2f}")

        # ── Dates d'observation réelles (dates anniversaires exactes) ────────
        # On calcule les N dates d'observation à partir de pricing_start.
        # À chaque date, si spot >= autocall_barrier * strike_abs → le produit est callé.
        all_obs_times = self.product.observation_times()   # en années depuis émission
        obs_dates_real: list[pd.Timestamp] = [
            pricing_start + pd.Timedelta(days=int(t_obs * 365.25))
            for t_obs in all_obs_times
        ]

        # État du cycle de vie du produit
        product_alive = True          # False dès qu'un call event est déclenché
        call_event_date: pd.Timestamp | None = None
        call_redemption_value: float = 0.0   # Valeur de remboursement au call
        obs_idx_triggered: int = -1          # Index de l'observation qui a callé

        # ── Boucle de rebalancement ───────────────────────────────────────────
        for i, t in enumerate(rebal_dates[1:], 1):
            S = float(prices.asof(t))
            sigma = float(vol_rolling.asof(t))

            if np.isnan(S) or np.isnan(sigma) or sigma < 0.01:
                continue

            # Temps restant (en années depuis aujourd'hui)
            T_elapsed = (t - pricing_start).days / 365.25
            T_remaining = max(self.product.maturity_years - T_elapsed, 0.0)

            if T_remaining < 1e-4:
                break

            prev = self.history[-1]

            # ── Vérification des dates d'observation passées depuis le dernier step ──
            # On teste toutes les dates d'observation entre prev.date et t (inclus).
            # Si l'une d'elles est franchie ET spot >= barrière → call event.
            if product_alive:
                for obs_idx, obs_date in enumerate(obs_dates_real):
                    if prev.date < obs_date <= t:
                        S_at_obs = float(prices.asof(obs_date))
                        if S_at_obs >= self.product.autocall_barrier * strike_abs:
                            product_alive = False
                            call_event_date = obs_date
                            obs_idx_triggered = obs_idx
                            coupon_at_call = self.product.cumulative_coupon(obs_idx)
                            call_redemption_value = self.product.nominal * (1.0 + coupon_at_call)
                            t_to_call = (obs_date - pricing_start).days / 365.25
                            df_call = np.exp(-self.r * t_to_call)
                            print(
                                f"[Hedge] *** AUTOCALL EVENT *** | Date={obs_date.date()} | "
                                f"S={S_at_obs:.2f} >= {self.product.autocall_barrier * strike_abs:.2f} | "
                                f"Coupon={coupon_at_call:.1%} | Redemption={call_redemption_value:.2f}"
                            )
                            break  # Un seul call event possible

            # ── MtM de la position AVANT rebalancement ────────────────────────
            prev_straddle_mtm = price_straddle_atm(
                S, self.r, sigma,
                max(min(self.straddle_maturity, prev.T_remaining * 0.95), 5 / 365),
                K=prev.straddle.K,
            )
            dt_days = (t - rebal_dates[i - 1]).days
            cash_accrued = prev.cash * np.exp(self.r * dt_days / 365.25)
            hedge_value_pre = (
                prev.q_underlying * S
                + prev.q_straddle * prev_straddle_mtm.price
                + cash_accrued
            )

            # ── Cas 1 : Produit callé → enregistrement de la liquidation puis arrêt ──
            if not product_alive:
                # On n'enregistre qu'UN SEUL état post-call (le step où t >= call_event_date).
                # Après ça, on sort de la boucle — il n'y a plus rien à tracker.

                # hedge_value_pre = valeur du portefeuille au moment de la liquidation
                # C'est ce qu'on encaisse pour rembourser l'investisseur.
                # P&L final = hedge encaissé - remboursement dû
                final_pnl = hedge_value_pre - call_redemption_value

                from ..core.greeks import Greeks
                greeks_zero = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, price=0.0)
                straddle_zero = price_straddle_atm(S, self.r, sigma,
                                                   max(self.straddle_maturity, 5 / 365))

                state = HedgeState(
                    date=call_event_date,      # On date l'état à la date exacte du call
                    spot=float(prices.asof(call_event_date)),
                    sigma=float(vol_rolling.asof(call_event_date)),
                    T_remaining=T_remaining,
                    greeks=greeks_zero,
                    q_underlying=0.0,
                    q_straddle=0.0,
                    straddle=straddle_zero,
                    autocall_mtm=0.0,          # Produit remboursé → n'existe plus
                    hedge_value=0.0,           # Portefeuille liquidé → remboursement payé
                    cash=0.0,
                    replication_error=final_pnl,   # P&L de liquidation (seule métrique finale)
                    gamma_residual=0.0,
                )
                self.history.append(state)

                print(
                    f"[Hedge] Liquidation | hedge={hedge_value_pre:.2f} | "
                    f"remboursement={call_redemption_value:.2f} | "
                    f"P&L final={final_pnl:+.2f} ({final_pnl/call_redemption_value:+.2%})"
                )
                break   # ← On sort de la boucle, le produit est mort

            # ── Cas 2 : Produit vivant → pricing MC + rebalancement ───────────
            straddle_T = min(self.straddle_maturity, T_remaining * 0.95)
            straddle_T = max(straddle_T, 5 / 365)
            straddle_hedge = price_straddle_atm(S, self.r, sigma, straddle_T)

            greeks = compute_greeks(
                self.product, S, self.r, sigma, T_remaining, self.mc_config,
                strike_level_abs=strike_abs,
            )
            V = greeks.price
            replication_error = hedge_value_pre - V

            # Clip q_straddle (garde-fou en vol-stress)
            q_straddle_raw = -greeks.vega / straddle_hedge.vega if abs(straddle_hedge.vega) > 1e-10 else 0.0
            q_straddle_new = np.clip(q_straddle_raw, -10.0, 10.0)
            q_underlying_new = -(greeks.delta + q_straddle_new * straddle_hedge.delta)

            # ── AUTO-FINANCEMENT : le bond est mis à jour par le coût marginal du trade ──
            # bond(t) = bond(t-1) * exp(r*dt)                   [capitalisation]
            #         - (dq_underlying * S + dq_straddle * straddle_price)  [coût du rebal]
            # Toute vente de positions abonde le bond ; tout achat le débite.
            # On ne réinjecte JAMAIS de capital externe.
            dt_years = dt_days / 365.25
            bond_accrued = prev.cash * np.exp(self.r * dt_years)
            rebal_cost = (
                (q_underlying_new - prev.q_underlying) * S
                + (q_straddle_new - prev.q_straddle) * straddle_hedge.price
            )
            bond_new = bond_accrued - rebal_cost

            hedge_value_post = q_underlying_new * S + q_straddle_new * straddle_hedge.price + bond_new
            gamma_residual = greeks.gamma + q_straddle_new * straddle_hedge.gamma

            state = HedgeState(
                date=t,
                spot=S,
                sigma=sigma,
                T_remaining=T_remaining,
                greeks=greeks,
                q_underlying=q_underlying_new,
                q_straddle=q_straddle_new,
                straddle=straddle_hedge,
                autocall_mtm=V,
                hedge_value=hedge_value_post,
                cash=bond_new,
                replication_error=replication_error,
                gamma_residual=gamma_residual,
            )
            self.history.append(state)

        # ── Résumé du cycle de vie ────────────────────────────────────────────
        if call_event_date is not None:
            print(f"[Hedge] Produit callé à l'observation {obs_idx_triggered + 1} "
                  f"({call_event_date.date()}) | Remboursement={call_redemption_value:.2f}")
        else:
            print("[Hedge] Produit arrivé à maturité sans autocall.")

        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        """Convertit l'historique en DataFrame structuré."""
        rows = []
        for s in self.history:
            rows.append({
                "date":               s.date,
                "spot":               s.spot,
                "sigma":              s.sigma,
                "T_remaining":        s.T_remaining,
                "autocall_mtm":       s.autocall_mtm,
                "hedge_value":        s.hedge_value,
                "cash":               s.cash,
                "replication_error":  s.replication_error,
                "delta":              s.greeks.delta,
                "gamma":              s.greeks.gamma,
                "vega":               s.greeks.vega,
                "theta":              s.greeks.theta,
                "gamma_residual":     s.gamma_residual,
                "q_underlying":       s.q_underlying,
                "q_straddle":         s.q_straddle,
                "straddle_price":     s.straddle.price,
                "straddle_delta":     s.straddle.delta,
                "straddle_vega":      s.straddle.vega,
            })
        return pd.DataFrame(rows).set_index("date")