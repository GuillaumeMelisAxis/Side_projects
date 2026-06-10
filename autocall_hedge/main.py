"""
main.py
=======
Point d'entrée principal pour le backtest de réplication dynamique
d'un Athena autocall.

Usage :
    python main.py

Configuration :
    Modifier les paramètres dans la section CONFIG ci-dessous.
"""

from __future__ import annotations

import os
import time
import pandas as pd

from autocall_hedge.core import (
    download_market_data,
    AthenaProduct,
    MCConfig,
)
from autocall_hedge.hedge import HedgePortfolio
from autocall_hedge.viz import run_all_plots


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — Modifier ici
# ═══════════════════════════════════════════════════════════════════════════════

TICKER          = "^STOXX50E"      # Yahoo Finance ticker
DATA_START      = "2018-01-01"     # Début des données historiques
DATA_END        = "2023-12-31"     # Fin des données historiques
PRICING_START   = "2019-01-02"     # Début du backtest de couverture
VOL_WINDOW      = 30               # Fenêtre vol historique en jours ouvrés

# Paramètres du produit Athena
PRODUCT = AthenaProduct(
    nominal          = 1_000.0,
    maturity_years   = 5,
    obs_frequency    = "annual",
    autocall_barrier = 1.30,    # Remboursement si S >= 130% du strike
    protection_barrier = 0.60,  # PDI si S(T) < 60% du strike
    coupon_rate      = 0.08,    # 8% annuel conditionnel
    strike           = 1.00,    # ATM à l'émission
)

# Monte Carlo
MC_CONFIG = MCConfig(
    n_paths              = 10_000,   # Augmenter pour plus de précision (ralentit)
    n_steps_per_year     = 52,      # Pas hebdomadaire
    seed                 = 42,
    use_antithetic       = True,
    use_moment_matching  = True,
)

# Taux sans risque
RISK_FREE_RATE = 0.045

# Portefeuille de couverture
REBALANCE_FREQ   = 5    # Rebalancement hebdomadaire (tous les 5 jours ouvrés)
STRADDLE_MAT     = 3.0 / 12   # Maturité du straddle : 3 mois (plus stable qu'1M, vega plus grand)

# Output
OUTPUT_DIR = "./output_plots"   # None pour ne pas sauvegarder

# ═══════════════════════════════════════════════════════════════════════════════


def main() -> pd.DataFrame:
    t0 = time.time()

    print("=" * 60)
    print("  ATHENA AUTOCALL — DYNAMIC DELTA-GAMMA-VEGA REPLICATION")
    print("=" * 60)

    # ── 1. Données de marché ──────────────────────────────────────────────────
    print("\n[1/3] Téléchargement des données de marché...")
    market = download_market_data(
        ticker=TICKER,
        start=DATA_START,
        end=DATA_END,
        vol_window=VOL_WINDOW,
    )

    # ── 2. Spot initial du produit (date d'émission) ──────────────────────────
    pricing_start_ts = pd.Timestamp(PRICING_START)
    S0_product = market.spot_at(pricing_start_ts)
    print(f"\n[2/3] Initialisation du produit")
    print(f"      Strike (S0 produit) = {S0_product:.2f}  ({PRICING_START})")
    print(f"      Maturité            = {PRODUCT.maturity_years} ans")
    print(f"      Coupon annuel       = {PRODUCT.coupon_rate:.0%}")
    print(f"      Autocall barrier    = {PRODUCT.autocall_barrier:.0%}")
    print(f"      Protection barrier  = {PRODUCT.protection_barrier:.0%}")

    # ── 3. Réplication dynamique ──────────────────────────────────────────────
    print(f"\n[3/3] Lancement du backtest de réplication...")
    print(f"      N_paths MC = {MC_CONFIG.n_paths} | Rebal. freq = {REBALANCE_FREQ}j\n")

    portfolio = HedgePortfolio(
        product           = PRODUCT,
        mc_config         = MC_CONFIG,
        r                 = RISK_FREE_RATE,
        straddle_maturity = STRADDLE_MAT,
        rebalance_freq    = REBALANCE_FREQ,
    )

    df = portfolio.run(
        prices         = market.prices,
        vol_rolling    = market.vol_rolling,
        S0_product     = S0_product,
        pricing_start  = pricing_start_ts,
    )

    # ── Statistiques de synthèse ──────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  RÉSULTATS ({len(df)} dates de rebalancement | {elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"  Replication Error | Mean  : {df['replication_error'].mean():+.4f}")
    print(f"                    | Std   : {df['replication_error'].std():.4f}")
    print(f"                    | Max   : {df['replication_error'].max():+.4f}")
    print(f"                    | Min   : {df['replication_error'].min():+.4f}")
    err_pct = df['replication_error'] / df['autocall_mtm'].replace(0, float('nan')) * 100
    print(f"  Error / MtM       | Mean  : {err_pct.mean():+.2f}%")
    print(f"                    | Std   : {err_pct.std():.2f}%")
    print(f"\n  Greeks (moyenne sur la période)")
    print(f"  Delta : {df['delta'].mean():.4f}")
    print(f"  Gamma : {df['gamma'].mean():.6f}")
    print(f"  Vega  : {df['vega'].mean():.4f}")

    # ── Sauvegarde CSV ────────────────────────────────────────────────────────
    if OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        csv_path = os.path.join(OUTPUT_DIR, "replication_results.csv")
        df.to_csv(csv_path)
        print(f"\n  Résultats sauvegardés → {csv_path}")

    # ── Visualisations ────────────────────────────────────────────────────────
    print(f"\n[Viz] Génération des 5 figures...")
    run_all_plots(df, ticker=TICKER, output_dir=OUTPUT_DIR)

    return df


if __name__ == "__main__":
    results = main()
