"""
athena_pricer.py
================
Définition du produit Athena et pricer Monte Carlo avec variance reduction.

Structure Athena classique :
- Observations annuelles
- Remboursement anticipé si S(t_i) >= Autocall_Barrier * S(0)
  → Nominal + Coupon * (i/n_years)  [coupons cumulés]
- À maturité :
    Si S(T) >= Protection_Barrier * S(0) → 100% du nominal
    Sinon                                 → Nominal * S(T)/S(0)  (PDI)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AthenaProduct:
    """
    Paramètres du produit structuré Athena.

    Parameters
    ----------
    nominal         : Nominal en unités monétaires
    maturity_years  : Maturité en années
    obs_frequency   : Fréquence d'observation ("annual", "semi-annual", "quarterly")
    autocall_barrier: Niveau autocall (ex. 1.00 = 100% du strike)
    protection_barrier : Barrière de protection PDI à maturité (ex. 0.60)
    coupon_rate     : Coupon annuel conditionnel (ex. 0.07 = 7%)
    strike          : Niveau de strike relatif (ex. 1.00 = ATM)
    """
    nominal: float = 1_000.0
    maturity_years: int = 5
    obs_frequency: str = "annual"          # "annual" | "semi-annual" | "quarterly"
    autocall_barrier: float = 1.00        # S(t) >= autocall_barrier * S0
    protection_barrier: float = 0.60      # PDI si S(T) < protection_barrier * S0
    coupon_rate: float = 0.07             # Coupon annuel conditionnel
    strike: float = 1.00                  # Strike relatif au spot initial

    def observation_times(self) -> np.ndarray:
        """Retourne les temps d'observation en années."""
        freq_map = {"annual": 1, "semi-annual": 2, "quarterly": 4}
        steps_per_year = freq_map[self.obs_frequency]
        n = self.maturity_years * steps_per_year
        return np.linspace(1.0 / steps_per_year, self.maturity_years, n)

    def cumulative_coupon(self, obs_index: int) -> float:
        """
        Coupon total payé lors du remboursement anticipé à l'observation i.
        Athena classique : coupons cumulés depuis le début.
        """
        freq_map = {"annual": 1, "semi-annual": 2, "quarterly": 4}
        steps_per_year = freq_map[self.obs_frequency]
        years_elapsed = (obs_index + 1) / steps_per_year
        return self.coupon_rate * years_elapsed

    def total_coupon_at_maturity(self) -> float:
        freq_map = {"annual": 1, "semi-annual": 2, "quarterly": 4}
        n = self.maturity_years * freq_map[self.obs_frequency]
        return self.cumulative_coupon(n - 1)


@dataclass
class MCConfig:
    """Configuration du moteur Monte Carlo."""
    n_paths: int = 10_000
    n_steps_per_year: int = 52          # Pas hebdomadaires
    seed: Optional[int] = 42
    use_antithetic: bool = True
    use_moment_matching: bool = True


def _simulate_gbm(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator,
    use_antithetic: bool,
    use_moment_matching: bool,
) -> np.ndarray:
    """
    Simule des trajectoires GBM sous la mesure risque-neutre.

    Returns
    -------
    paths : np.ndarray shape (n_paths, n_steps+1)
    """
    dt = T / n_steps
    half = n_paths // 2 if use_antithetic else n_paths

    Z = rng.standard_normal((half, n_steps))

    if use_moment_matching:
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)

    if use_antithetic:
        Z = np.vstack([Z, -Z])

    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z

    log_increments = drift + diffusion
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)


def price_athena(
    product: AthenaProduct,
    S0: float,
    r: float,
    sigma: float,
    T_remaining: float,
    mc_config: MCConfig,
    already_called: bool = False,
    strike_level_abs: float | None = None,
) -> dict:
    """
    Priceur Monte Carlo pour un Athena.

    Parameters
    ----------
    product       : AthenaProduct
    S0            : Spot courant (niveau d'entrée ou niveau actuel)
    r             : Taux sans risque annualisé
    sigma         : Volatilité annualisée
    T_remaining   : Temps restant jusqu'à maturité en années
    mc_config     : Configuration MC
    already_called: Si True, le produit est déjà mort (valeur = 0)

    Returns
    -------
    dict avec "price", "std_error", "paths" (optionnel)
    """
    if already_called or T_remaining <= 1e-6:
        return {"price": 0.0, "std_error": 0.0}

    rng = np.random.default_rng(mc_config.seed)

    # Temps d'observation restants (en années depuis maintenant)
    all_obs = product.observation_times()
    obs_remaining = all_obs[all_obs <= product.maturity_years] - (
        product.maturity_years - T_remaining
    )
    obs_remaining = obs_remaining[obs_remaining > 1e-6]

    T_sim = T_remaining
    n_steps = max(int(T_sim * mc_config.n_steps_per_year), 1)

    paths = _simulate_gbm(
        S0=S0,
        r=r,
        sigma=sigma,
        T=T_sim,
        n_paths=mc_config.n_paths,
        n_steps=n_steps,
        rng=rng,
        use_antithetic=mc_config.use_antithetic,
        use_moment_matching=mc_config.use_moment_matching,
    )  # shape (n_paths, n_steps+1)

    t_grid = np.linspace(0, T_sim, n_steps + 1)

    payoffs = np.zeros(mc_config.n_paths)
    discount_factors = np.zeros(mc_config.n_paths)
    alive = np.ones(mc_config.n_paths, dtype=bool)

    # strike_level_abs est le niveau de strike FIXE à l'émission.
    # Quand on bumpe S0 pour calculer les Greeks, il ne faut PAS que le strike bouge.
    strike_level = strike_level_abs if strike_level_abs is not None else product.strike * S0

    for i, t_obs in enumerate(obs_remaining):
        idx = int(np.argmin(np.abs(t_grid - t_obs)))
        S_obs = paths[:, idx]

        triggered = alive & (S_obs >= product.autocall_barrier * strike_level)

        # Coupon cumulé = coupons jusqu'à cet index d'observation global
        obs_global_idx = np.searchsorted(all_obs, t_obs + (product.maturity_years - T_remaining) - 1e-9)
        coupon = product.cumulative_coupon(obs_global_idx)

        payoffs[triggered] = product.nominal * (1.0 + coupon)
        discount_factors[triggered] = np.exp(-r * t_obs)
        alive[triggered] = False

    # Payoff à maturité pour les paths encore vivants
    S_T = paths[:, -1]
    S_T_alive = S_T[alive]

    payoff_mat = np.where(
        S_T_alive >= product.protection_barrier * strike_level,
        product.nominal * (1.0 + product.total_coupon_at_maturity()),
        product.nominal * S_T_alive / strike_level,
    )

    payoffs[alive] = payoff_mat
    discount_factors[alive] = np.exp(-r * T_sim)

    pv = payoffs * discount_factors
    price = float(np.mean(pv))
    std_error = float(np.std(pv) / np.sqrt(mc_config.n_paths))

    return {"price": price, "std_error": std_error}
