"""
visualization.py
================
Visualisations production-quality de la réplication dynamique d'un Athena.

Figures produites :
  1. MtM Autocall vs Portefeuille de couverture + Spot normalisé
  2. Erreur de réplication cumulée
  3. Greeks over time (Delta, Gamma, Vega, Theta)
  4. Composition du portefeuille de couverture (q_underlying, q_straddle, cash)
  5. P&L decomposition : gamma residual + sigma rolling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

# ── Style global ─────────────────────────────────────────────────────────────
STYLE = {
    "bg":        "#0D1117",
    "panel":     "#161B22",
    "border":    "#30363D",
    "text":      "#E6EDF3",
    "subtext":   "#8B949E",
    "autocall":  "#58A6FF",
    "hedge":     "#3FB950",
    "error":     "#F85149",
    "spot":      "#D2A8FF",
    "delta":     "#79C0FF",
    "gamma":     "#56D364",
    "vega":      "#FFA657",
    "theta":     "#FF7B72",
    "cash":      "#A5D6FF",
    "q_u":       "#7EE787",
    "q_s":       "#FFA657",
    "grid":      "#21262D",
}

def _apply_dark_style(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(STYLE["panel"])
    ax.tick_params(colors=STYLE["subtext"], labelsize=9)
    ax.xaxis.label.set_color(STYLE["subtext"])
    ax.yaxis.label.set_color(STYLE["subtext"])
    ax.title.set_color(STYLE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["border"])
    ax.grid(True, color=STYLE["grid"], linewidth=0.5, alpha=0.8)
    if title:
        ax.set_title(title, color=STYLE["text"], fontsize=11, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=STYLE["subtext"], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=STYLE["subtext"], fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


def plot_replication_overview(df: pd.DataFrame, ticker: str = "", save_path: str | None = None) -> None:
    """
    Figure 1 : Vue d'ensemble MtM + Spot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), facecolor=STYLE["bg"],
                                    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    # Panel haut : MtM autocall vs hedge portfolio
    ax1.plot(df.index, df["autocall_mtm"], color=STYLE["autocall"], lw=2.0,
             label="Autocall MtM (MC)", zorder=3)
    ax1.plot(df.index, df["hedge_value"], color=STYLE["hedge"], lw=1.8, linestyle="--",
             label="Hedge Portfolio Value", zorder=3)
    ax1.fill_between(df.index, df["autocall_mtm"], df["hedge_value"],
                     alpha=0.12, color=STYLE["error"], label="Replication gap")
    _apply_dark_style(ax1, title=f"Dynamic Replication — Athena Autocall  |  {ticker}",
                      ylabel="Value (nominal units)")
    ax1.legend(loc="upper right", framealpha=0.2, facecolor=STYLE["panel"],
               edgecolor=STYLE["border"], labelcolor=STYLE["text"], fontsize=9)

    # Panel bas : Spot normalisé
    spot_norm = df["spot"] / df["spot"].iloc[0] * 100
    ax2.plot(df.index, spot_norm, color=STYLE["spot"], lw=1.5, alpha=0.9)
    ax2.fill_between(df.index, 100, spot_norm,
                     where=(spot_norm < 100), color=STYLE["error"], alpha=0.15)
    ax2.fill_between(df.index, 100, spot_norm,
                     where=(spot_norm >= 100), color=STYLE["gamma"], alpha=0.10)
    ax2.axhline(100, color=STYLE["subtext"], lw=0.8, linestyle=":")
    _apply_dark_style(ax2, ylabel="Spot (base 100)")

    fig.patch.set_facecolor(STYLE["bg"])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.show()


def plot_replication_error(df: pd.DataFrame, save_path: str | None = None) -> None:
    """
    Figure 2 : Erreur de réplication (absolue + % du MtM)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), facecolor=STYLE["bg"],
                                    gridspec_kw={"hspace": 0.15})

    err = df["replication_error"]
    err_pct = err / df["autocall_mtm"].replace(0, np.nan) * 100

    # Erreur absolue
    ax1.bar(df.index, err, width=3, color=np.where(err >= 0, STYLE["gamma"], STYLE["error"]),
            alpha=0.8)
    ax1.axhline(0, color=STYLE["subtext"], lw=0.8)
    ax1.axhline(err.std(), color=STYLE["hedge"], lw=1.0, linestyle=":", alpha=0.7,
                label=f"+1σ = {err.std():.3f}")
    ax1.axhline(-err.std(), color=STYLE["hedge"], lw=1.0, linestyle=":", alpha=0.7,
                label=f"−1σ = {-err.std():.3f}")
    _apply_dark_style(ax1, title="Replication Error — Absolute (Hedge PV − Autocall MtM)",
                      ylabel="Abs. Error (nominal)")
    ax1.legend(framealpha=0.2, facecolor=STYLE["panel"], edgecolor=STYLE["border"],
               labelcolor=STYLE["text"], fontsize=9)

    # Erreur en %
    ax2.plot(df.index, err_pct, color=STYLE["error"], lw=1.5, alpha=0.9)
    ax2.fill_between(df.index, 0, err_pct, where=(err_pct >= 0),
                     color=STYLE["gamma"], alpha=0.15)
    ax2.fill_between(df.index, 0, err_pct, where=(err_pct < 0),
                     color=STYLE["error"], alpha=0.15)
    ax2.axhline(0, color=STYLE["subtext"], lw=0.8)
    _apply_dark_style(ax2, title="Replication Error — % of Autocall MtM",
                      ylabel="Error (%)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))

    fig.patch.set_facecolor(STYLE["bg"])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.show()


def plot_greeks(df: pd.DataFrame, save_path: str | None = None) -> None:
    """
    Figure 3 : Évolution des Greeks (Delta, Gamma, Vega, Theta)
    """
    fig = plt.figure(figsize=(16, 9), facecolor=STYLE["bg"])
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    greeks_config = [
        ("delta",  "Delta  (dV/dS)",                STYLE["delta"],  gs[0, 0]),
        ("gamma",  "Gamma  (d²V/dS²)",              STYLE["gamma"],  gs[0, 1]),
        ("vega",   "Vega  (dV/dσ × 1%)",            STYLE["vega"],   gs[1, 0]),
        ("theta",  "Theta  (dV/dt, per day)",        STYLE["theta"],  gs[1, 1]),
    ]

    for col, title, color, pos in greeks_config:
        ax = fig.add_subplot(pos)
        series = df[col]
        ax.plot(df.index, series, color=color, lw=1.8)
        ax.fill_between(df.index, 0, series, color=color, alpha=0.10)
        ax.axhline(0, color=STYLE["subtext"], lw=0.7)
        _apply_dark_style(ax, title=title)

    fig.suptitle("Greeks Evolution — Athena Autocall", color=STYLE["text"],
                 fontsize=13, fontweight="bold", y=1.01)
    fig.patch.set_facecolor(STYLE["bg"])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.show()


def plot_hedge_composition(df: pd.DataFrame, save_path: str | None = None) -> None:
    """
    Figure 4 : Composition du portefeuille de couverture
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), facecolor=STYLE["bg"],
                              gridspec_kw={"hspace": 0.25})

    # q_underlying
    axes[0].plot(df.index, df["q_underlying"], color=STYLE["q_u"], lw=1.8)
    axes[0].fill_between(df.index, 0, df["q_underlying"], color=STYLE["q_u"], alpha=0.15)
    axes[0].axhline(0, color=STYLE["subtext"], lw=0.7)
    _apply_dark_style(axes[0], title="Underlying Position  (q_underlying)", ylabel="Quantity")

    # q_straddle
    axes[1].plot(df.index, df["q_straddle"], color=STYLE["q_s"], lw=1.8)
    axes[1].fill_between(df.index, 0, df["q_straddle"], color=STYLE["q_s"], alpha=0.15)
    axes[1].axhline(0, color=STYLE["subtext"], lw=0.7)
    _apply_dark_style(axes[1], title="Straddle Position  (q_straddle — Vega Hedge)", ylabel="Quantity")

    # Cash / Bond
    axes[2].plot(df.index, df["cash"], color=STYLE["cash"], lw=1.8)
    axes[2].fill_between(df.index, 0, df["cash"], color=STYLE["cash"], alpha=0.12)
    axes[2].axhline(0, color=STYLE["subtext"], lw=0.7)
    _apply_dark_style(axes[2], title="Cash / Financing Account", ylabel="Nominal")

    fig.suptitle("Hedge Portfolio Composition", color=STYLE["text"],
                 fontsize=13, fontweight="bold", y=1.01)
    fig.patch.set_facecolor(STYLE["bg"])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.show()


def plot_gamma_vol_analysis(df: pd.DataFrame, save_path: str | None = None) -> None:
    """
    Figure 5 : Gamma résiduel + vol rolling + analyse P&L source
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), facecolor=STYLE["bg"],
                                          gridspec_kw={"hspace": 0.3})

    # Gamma résiduel
    ax1.bar(df.index, df["gamma_residual"], width=3,
            color=np.where(df["gamma_residual"] >= 0, STYLE["gamma"], STYLE["error"]),
            alpha=0.85)
    ax1.axhline(0, color=STYLE["subtext"], lw=0.7)
    _apply_dark_style(ax1, title="Residual Gamma (Gamma_autocall + q_straddle × Gamma_straddle)",
                      ylabel="Gamma (residual)")

    # Vol rolling
    ax2.plot(df.index, df["sigma"] * 100, color=STYLE["vega"], lw=1.8)
    ax2.fill_between(df.index, df["sigma"].mean() * 100, df["sigma"] * 100,
                     where=(df["sigma"] > df["sigma"].mean()),
                     color=STYLE["error"], alpha=0.15, label="Above avg vol")
    ax2.fill_between(df.index, df["sigma"].mean() * 100, df["sigma"] * 100,
                     where=(df["sigma"] <= df["sigma"].mean()),
                     color=STYLE["gamma"], alpha=0.12, label="Below avg vol")
    ax2.axhline(df["sigma"].mean() * 100, color=STYLE["subtext"],
                lw=1.0, linestyle="--", label=f"Mean: {df['sigma'].mean():.1%}")
    _apply_dark_style(ax2, title="Rolling Historical Volatility (annualized)", ylabel="Vol (%)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.legend(framealpha=0.2, facecolor=STYLE["panel"], edgecolor=STYLE["border"],
               labelcolor=STYLE["text"], fontsize=9)

    # Scatter : gamma résiduel vs variation de spot
    spot_ret = df["spot"].pct_change().fillna(0) * 100
    sc = ax3.scatter(spot_ret, df["replication_error"], c=df["sigma"],
                     cmap="plasma", s=20, alpha=0.7, edgecolors="none")
    ax3.axhline(0, color=STYLE["subtext"], lw=0.7)
    ax3.axvline(0, color=STYLE["subtext"], lw=0.7)
    cbar = plt.colorbar(sc, ax=ax3)
    cbar.set_label("Sigma", color=STYLE["subtext"], fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=STYLE["subtext"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=STYLE["subtext"])
    _apply_dark_style(ax3, title="Replication Error vs Spot Return (colored by vol)",
                      xlabel="Spot daily return (%)", ylabel="Replication Error")
    ax3.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))

    fig.patch.set_facecolor(STYLE["bg"])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.show()


def run_all_plots(df: pd.DataFrame, ticker: str = "", output_dir: str | None = None) -> None:
    """Lance toutes les visualisations en séquence."""
    def _path(name):
        return f"{output_dir}/{name}.png" if output_dir else None

    plot_replication_overview(df, ticker=ticker, save_path=_path("01_overview"))
    plot_replication_error(df, save_path=_path("02_replication_error"))
    plot_greeks(df, save_path=_path("03_greeks"))
    plot_hedge_composition(df, save_path=_path("04_hedge_composition"))
    plot_gamma_vol_analysis(df, save_path=_path("05_gamma_vol"))
