"""
market_data.py
==============
Téléchargement de la time series de prix via yfinance et calibration
de la volatilité historique rolling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass


@dataclass
class MarketData:
    """Conteneur pour la time series de marché et la vol calibrée."""
    ticker: str
    prices: pd.Series          # Prix de clôture ajustés
    log_returns: pd.Series     # Log-rendements journaliers
    vol_rolling: pd.Series     # Vol annualisée rolling (fenêtre configurable)
    vol_window: int            # Fenêtre en jours ouvrés

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.prices.index

    def vol_at(self, date: pd.Timestamp) -> float:
        """Retourne la vol rolling à une date donnée (forward-fill si NaN)."""
        val = self.vol_rolling.asof(date)
        if np.isnan(val):
            val = self.vol_rolling.dropna().iloc[0]
        return float(val)

    def spot_at(self, date: pd.Timestamp) -> float:
        """Retourne le prix spot à une date donnée (asof)."""
        return float(self.prices.asof(date))


def download_market_data(
    ticker: str,
    start: str,
    end: str,
    vol_window: int = 30,
    risk_free_rate: float = 0.03,
) -> MarketData:
    """
    Télécharge la time series de prix via yfinance et calcule la vol
    historique rolling annualisée.

    Parameters
    ----------
    ticker      : Ticker Yahoo Finance (ex. "^STOXX50E", "SPY", "^FCHI")
    start       : Date de début ISO "YYYY-MM-DD"
    end         : Date de fin ISO "YYYY-MM-DD"
    vol_window  : Fenêtre rolling en jours ouvrés (défaut : 30)
    risk_free_rate : Taux sans risque (non utilisé ici, passé au pricer)

    Returns
    -------
    MarketData
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"Aucune donnée téléchargée pour {ticker}.")

    prices: pd.Series = raw["Close"].squeeze().dropna()
    prices.name = ticker

    log_returns: pd.Series = np.log(prices / prices.shift(1)).dropna()

    # Vol historique rolling annualisée : σ * sqrt(252)
    vol_rolling: pd.Series = (
        log_returns.rolling(window=vol_window).std() * np.sqrt(252)
    )
    vol_rolling.name = f"vol_{vol_window}d"

    print(
        f"[MarketData] {ticker} | {len(prices)} jours | "
        f"Vol moyenne : {vol_rolling.mean():.1%} | "
        f"Spot final : {prices.iloc[-1]:.2f}"
    )

    return MarketData(
        ticker=ticker,
        prices=prices,
        log_returns=log_returns,
        vol_rolling=vol_rolling,
        vol_window=vol_window,
    )
