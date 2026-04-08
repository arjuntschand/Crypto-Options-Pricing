"""Stress testing across crash regimes."""

from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.black_scholes import black_scholes_price
from src.monte_carlo import monte_carlo_price
from src.heston import heston_mc_price


def _select_representative_options(options_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Select a diverse mix of options for stress testing."""
    df = options_df.copy()
    if len(df) <= n:
        return df

    spot = df["spot_price"].iloc[0]
    df["moneyness"] = df["strike"] / spot

    # Categorize
    df["money_cat"] = pd.cut(
        df["moneyness"],
        bins=[0, 0.95, 1.05, 100],
        labels=["ITM", "ATM", "OTM"],
    )
    df["expiry_cat"] = pd.cut(
        df["time_to_expiry_years"] * 365.25,
        bins=[0, 30, 90, 10000],
        labels=["short", "medium", "long"],
    )

    # Sample from each bucket
    selected = []
    groups = df.groupby(["option_type", "money_cat", "expiry_cat"], observed=True)
    per_group = max(1, n // max(len(groups), 1))
    for _, group in groups:
        selected.append(group.sample(n=min(per_group, len(group)), random_state=42))

    result = pd.concat(selected, ignore_index=True)
    if len(result) > n:
        result = result.sample(n=n, random_state=42).reset_index(drop=True)
    return result


def run_stress_test(
    options_df: pd.DataFrame,
    crash_vols: Dict[str, Dict[str, float]],
    heston_params: Dict[str, Dict[str, float]],
    r: float = 0.05,
    results_dir: str = "results",
) -> pd.DataFrame:
    """
    Run stress tests across normal, COVID crash, and crypto crash regimes.
    """
    regimes = ["normal", "covid_crash", "crypto_crash"]
    models = ["BS", "MC", "Heston"]

    all_results: List[Dict] = []
    regime_model_errors: Dict[str, Dict[str, List[float]]] = {
        reg: {m: [] for m in models} for reg in regimes
    }

    for currency in options_df["currency"].unique():
        cdf = options_df[options_df["currency"] == currency].copy()
        if cdf.empty:
            continue

        subset = _select_representative_options(cdf, n=20)
        h_params = heston_params.get(currency, {})

        for regime in regimes:
            regime_vol = crash_vols.get(regime, {}).get(currency, 0.8)
            regime_var = regime_vol ** 2

            for _, row in subset.iterrows():
                S = row["spot_price"]
                K = row["strike"]
                T = row["time_to_expiry_years"]
                opt_type = row["option_type"]
                market = row["mark_price"]

                # BS price with regime vol
                bs_price = black_scholes_price(S, K, T, r, regime_vol, opt_type)

                # MC price with regime vol
                mc_res = monte_carlo_price(S, K, T, r, regime_vol, opt_type, n_paths=5000, n_steps=100, seed=42)
                mc_price = mc_res["price"]

                # Heston with regime variance
                if h_params:
                    h_res = heston_mc_price(
                        S, K, T, r,
                        v0=regime_var,
                        kappa=h_params["kappa"], theta=h_params["theta"],
                        sigma_v=h_params["sigma_v"], rho=h_params["rho"],
                        option_type=opt_type, n_paths=5000, n_steps=100, seed=42,
                    )
                    heston_price = h_res["price"]
                else:
                    heston_price = np.nan

                if market > 0:
                    for model_name, model_price in [("BS", bs_price), ("MC", mc_price), ("Heston", heston_price)]:
                        abs_err = abs(model_price - market)
                        pct_err = abs_err / market * 100.0
                        regime_model_errors[regime][model_name].append(pct_err)

                        all_results.append({
                            "currency": currency,
                            "instrument": row["instrument_name"],
                            "regime": regime,
                            "model": model_name,
                            "model_price": model_price,
                            "market_price": market,
                            "abs_error": abs_err,
                            "pct_error": pct_err,
                        })

    detail_df = pd.DataFrame(all_results)

    # Summary table
    summary_rows: List[Dict] = []
    for regime in regimes:
        for model in models:
            errs = regime_model_errors[regime][model]
            if errs:
                summary_rows.append({
                    "regime": regime,
                    "model": model,
                    "mean_abs_error": np.mean([r["abs_error"] for r in all_results
                                               if r["regime"] == regime and r["model"] == model]),
                    "mean_pct_error": np.mean(errs),
                    "max_error": np.max(errs),
                })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{results_dir}/stress_test_results.csv", index=False)

    # Bar chart
    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(regimes))
        width = 0.25
        for i, model in enumerate(models):
            vals = []
            for regime in regimes:
                row = summary_df[(summary_df["regime"] == regime) & (summary_df["model"] == model)]
                vals.append(row["mean_pct_error"].values[0] if len(row) > 0 else 0)
            ax.bar(x + i * width, vals, width, label=model)

        ax.set_xlabel("Regime", fontsize=12)
        ax.set_ylabel("Mean % Error vs Market", fontsize=12)
        ax.set_title("Model Pricing Errors Across Market Regimes", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(["Normal", "COVID Crash", "Crypto Crash"])
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/stress_test_chart.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved stress test chart: {results_dir}/stress_test_chart.png")

    return summary_df
