"""
greeks.py
=========
Calcul des Greeks de l'Athena par différences finies (bump & reprice).
Delta, Gamma, Vega calculés sur le pricer MC avec seed fixé pour
minimiser le bruit de simulation.

Convention :
  - Delta  : dV/dS          (sensibilité au spot)
  - Gamma  : d²V/dS²        (convexité en spot)
  - Vega   : dV/d(sigma)    (sensibilité à la vol, en pts de vol, /100)
  - Theta  : dV/dt          (décroissance temporelle, par jour)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .athena_pricer import AthenaProduct, MCConfig, price_athena


@dataclass
class Greeks:
    """Conteneur pour les Greeks d'une position."""
    delta: float      # Adimensionnel (hedge ratio en unités de sous-jacent)
    gamma: float      # 1/S  (convexité)
    vega: float       # En % de vol (dV / d(sigma) * 0.01)
    theta: float      # Par jour calendaire
    price: float      # MtM du produit


def compute_greeks(
    product: AthenaProduct,
    S0: float,
    r: float,
    sigma: float,
    T_remaining: float,
    mc_config: MCConfig,
    dS_pct: float = 0.005,          # Bump spot : 0.5%
    d_sigma: float = 0.001,         # Bump vol  : 10 bps
    dT: float = 1 / 365,            # Bump temps : 1 jour
    strike_level_abs: float | None = None,  # Strike fixe à l'émission
) -> Greeks:
    """
    Calcule les Greeks par différences finies centrées.

    Le seed MC est fixé pour que les bumps soient cohérents (même paths).
    Le strike_level_abs est le niveau de référence FIXE des barrières — il ne
    doit PAS changer quand on bumpe S0 pour calculer delta/gamma.
    """
    # Si pas fourni, on l'infère du spot courant (cas initial d'émission)
    if strike_level_abs is None:
        strike_level_abs = product.strike * S0

    dS = S0 * dS_pct

    # ── Pricer de base ────────────────────────────────────────────────────────
    base = price_athena(product, S0, r, sigma, T_remaining, mc_config,
                        strike_level_abs=strike_level_abs)
    V0 = base["price"]

    if V0 == 0.0 and T_remaining <= 0:
        return Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, price=0.0)

    # ── Delta & Gamma (différences centrées en S) ─────────────────────────────
    # strike_level_abs reste FIXE — seul S0 est bumpé
    V_up   = price_athena(product, S0 + dS, r, sigma, T_remaining, mc_config,
                          strike_level_abs=strike_level_abs)["price"]
    V_down = price_athena(product, S0 - dS, r, sigma, T_remaining, mc_config,
                          strike_level_abs=strike_level_abs)["price"]

    delta = (V_up - V_down) / (2 * dS)
    gamma = (V_up - 2 * V0 + V_down) / (dS**2)

    # ── Vega (différences centrées en sigma) ──────────────────────────────────
    V_vol_up   = price_athena(product, S0, r, sigma + d_sigma, T_remaining, mc_config,
                              strike_level_abs=strike_level_abs)["price"]
    V_vol_down = price_athena(product, S0, r, sigma - d_sigma, T_remaining, mc_config,
                              strike_level_abs=strike_level_abs)["price"]

    vega = (V_vol_up - V_vol_down) / (2 * d_sigma) * 0.01

    # ── Theta (différence en avant, 1 jour) ───────────────────────────────────
    T_shifted = max(T_remaining - dT, 1e-4)
    V_theta = price_athena(product, S0, r, sigma, T_shifted, mc_config,
                           strike_level_abs=strike_level_abs)["price"]
    theta = (V_theta - V0) / dT * dT

    return Greeks(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        price=V0,
    )
