"""
straddle.py
===========
Pricer Black-Scholes pour le straddle ATM utilisé comme instrument
de couverture vega.

Un straddle ATM = Call ATM + Put ATM (même strike = spot courant).
Il est delta-neutre par construction (Delta_call + Delta_put ≈ 0 à ATM).

On utilise BS analytique pour le straddle (rapide, pas de bruit MC).
La quantité de straddle dans le portefeuille est ajustée pour neutraliser
le vega résiduel de l'autocall.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.stats import norm


@dataclass
class StraddleGreeks:
    """Greeks du straddle unitaire (1 call + 1 put, même strike K = S)."""
    price: float       # Valeur totale du straddle
    delta: float       # ≈ 0 par construction ATM
    gamma: float       # Gamma total (additive)
    vega: float        # Vega total (en % de vol)
    K: float           # Strike effectif


def bs_call(S: float, K: float, r: float, sigma: float, T: float) -> tuple[float, float, float, float]:
    """
    Retourne (price, delta, gamma, vega) d'un call vanille BS.
    Vega retourné en convention marché (/100).
    """
    if T <= 1e-6:
        intrinsic = max(S - K, 0.0)
        return intrinsic, float(S > K), 0.0, 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = float(norm.cdf(d1))
    gamma = float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
    vega  = float(S * norm.pdf(d1) * np.sqrt(T) * 0.01)  # dV/d(sigma) * 1%

    return price, delta, gamma, vega


def bs_put(S: float, K: float, r: float, sigma: float, T: float) -> tuple[float, float, float, float]:
    """
    Retourne (price, delta, gamma, vega) d'un put vanille BS.
    """
    if T <= 1e-6:
        intrinsic = max(K - S, 0.0)
        return intrinsic, float(S < K) - 1.0, 0.0, 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    delta = float(norm.cdf(d1) - 1.0)
    gamma = float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
    vega  = float(S * norm.pdf(d1) * np.sqrt(T) * 0.01)

    return price, delta, gamma, vega


def price_straddle_atm(
    S: float,
    r: float,
    sigma: float,
    T: float,
    K: float | None = None,
) -> StraddleGreeks:
    """
    Price un straddle ATM (ou au strike K si fourni).

    Parameters
    ----------
    S     : Spot courant
    r     : Taux sans risque
    sigma : Vol implicite / historique
    T     : Maturité en années
    K     : Strike (défaut : S, i.e. ATM forward ≈ ATM spot)

    Returns
    -------
    StraddleGreeks
    """
    if K is None:
        K = S  # ATM spot strict

    c_price, c_delta, c_gamma, c_vega = bs_call(S, K, r, sigma, T)
    p_price, p_delta, p_gamma, p_vega = bs_put(S, K, r, sigma, T)

    return StraddleGreeks(
        price=c_price + p_price,
        delta=c_delta + p_delta,   # ≈ 0 ATM
        gamma=c_gamma + p_gamma,   # 2 * gamma_call
        vega=c_vega + p_vega,      # 2 * vega_call
        K=K,
    )
