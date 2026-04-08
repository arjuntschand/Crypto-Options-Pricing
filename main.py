"""
Crypto Options Pricing and Risk Engine - Full Pipeline

Run: python main.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import fetch_all_data, get_crash_volatilities
from src.black_scholes import black_scholes_price
from src.monte_carlo import monte_carlo_price
from src.heston import heston_mc_price, calibrate_heston
from src.greeks import compute_greeks_for_options
from src.vol_surface import build_vol_surface
from src.stress_test import run_stress_test
from src.validation import validate_models


def main() -> None:
    np.random.seed(42)
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    r = 0.05
    pipeline_start = time.time()

    # ─── Step 1: Data Collection ────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1: Fetching Deribit option chain data...")
    print("=" * 70)

    # Use cached data if available to skip slow API calls on re-runs
    cache_file = f"{results_dir}/deribit_options_raw.csv"
    btc_cache = f"{results_dir}/btc_historical.csv"
    eth_cache = f"{results_dir}/eth_historical.csv"
    if os.path.exists(cache_file) and os.path.exists(btc_cache) and os.path.exists(eth_cache):
        print("  Using cached data from previous run...")
        all_options = pd.read_csv(cache_file)
        all_options["expiry_date"] = pd.to_datetime(all_options["expiry_date"])
        btc_hist = pd.read_csv(btc_cache, index_col=0, parse_dates=True)
        eth_hist = pd.read_csv(eth_cache, index_col=0, parse_dates=True)
        crash_vols = get_crash_volatilities(btc_hist, eth_hist)
        print(f"  Crash volatilities computed:")
        for regime, vols in crash_vols.items():
            print(f"    {regime}: BTC={vols['BTC']:.4f}, ETH={vols['ETH']:.4f}")
    else:
        all_options, btc_hist, eth_hist, crash_vols = fetch_all_data(results_dir)

    btc_options = all_options[all_options["currency"] == "BTC"] if not all_options.empty else pd.DataFrame()
    eth_options = all_options[all_options["currency"] == "ETH"] if not all_options.empty else pd.DataFrame()
    n_btc = len(btc_options)
    n_eth = len(eth_options)
    print(f"\n  BTC options fetched: {n_btc}")
    print(f"  ETH options fetched: {n_eth}")
    print(f"  Total options: {len(all_options)}")

    if all_options.empty:
        print("\nERROR: No options data collected. Exiting.")
        sys.exit(1)

    # ─── Step 2: Black-Scholes Pricing ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Running Black-Scholes pricing...")
    print("=" * 70)
    bs_start = time.time()
    all_options["bs_price"] = all_options.apply(
        lambda row: black_scholes_price(
            row["spot_price"], row["strike"], row["time_to_expiry_years"],
            r, row["mark_iv"], row["option_type"],
        ), axis=1
    )
    print(f"  BS pricing completed in {time.time() - bs_start:.2f}s")
    print(f"  Sample BS prices: {all_options['bs_price'].head(5).tolist()}")

    # ─── Step 3: Monte Carlo Pricing ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Running Monte Carlo pricing (10,000 paths)...")
    print("=" * 70)
    mc_start = time.time()
    mc_prices = []
    for i, (_, row) in enumerate(all_options.iterrows()):
        if (i + 1) % 50 == 0:
            print(f"  MC priced {i + 1}/{len(all_options)} options...")
        res = monte_carlo_price(
            row["spot_price"], row["strike"], row["time_to_expiry_years"],
            r, row["mark_iv"], row["option_type"],
            n_paths=10000, n_steps=252, seed=42,
        )
        mc_prices.append(res["price"])
    all_options["mc_price"] = mc_prices
    print(f"  MC pricing completed in {time.time() - mc_start:.2f}s")

    # ─── Step 4: Heston Calibration ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: Calibrating Heston model...")
    print("=" * 70)
    heston_params = {}
    for currency in ["BTC", "ETH"]:
        cdf = all_options[all_options["currency"] == currency].copy()
        if cdf.empty:
            print(f"  Skipping {currency} Heston calibration (no data)")
            continue
        spot = cdf["spot_price"].iloc[0]
        print(f"\n  Calibrating Heston for {currency} (spot={spot:.2f})...")
        cal_start = time.time()
        params, rmse = calibrate_heston(cdf, spot, r, max_options=50, timeout_minutes=5.0)
        cal_time = time.time() - cal_start
        heston_params[currency] = params
        print(f"  {currency} Heston calibration completed in {cal_time:.1f}s")
        print(f"  {currency} Heston params: {params}")
        print(f"  {currency} Heston RMSE: {rmse:.4f}")

    # ─── Step 5: Heston Pricing ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: Running Heston pricing...")
    print("=" * 70)
    heston_start = time.time()
    heston_prices = []
    for i, (_, row) in enumerate(all_options.iterrows()):
        if (i + 1) % 50 == 0:
            print(f"  Heston priced {i + 1}/{len(all_options)} options...")
        currency = row["currency"]
        hp = heston_params.get(currency, None)
        if hp:
            res = heston_mc_price(
                row["spot_price"], row["strike"], row["time_to_expiry_years"],
                r, hp["v0"], hp["kappa"], hp["theta"], hp["sigma_v"], hp["rho"],
                option_type=row["option_type"],
                n_paths=10000, n_steps=252, seed=42,
            )
            heston_prices.append(res["price"])
        else:
            heston_prices.append(np.nan)
    all_options["heston_price"] = heston_prices
    print(f"  Heston pricing completed in {time.time() - heston_start:.2f}s")

    # ─── Step 6: Validation ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6: Validating models against market...")
    print("=" * 70)
    full_val, summary_val, key_metrics = validate_models(all_options, heston_params, r, results_dir)

    # ─── Step 7: Greeks ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 7: Computing Greeks...")
    print("=" * 70)
    greeks_df, total_greek_pairs = compute_greeks_for_options(all_options, heston_params, r, n_per_currency=20)
    greeks_df.to_csv(f"{results_dir}/greeks_comparison.csv", index=False)
    print(f"\n  Sample Greeks (first 5 rows):")
    if not greeks_df.empty:
        sample = greeks_df[["instrument", "option_type", "bs_delta", "bs_gamma", "bs_vega", "bs_theta"]].head(5)
        print(sample.to_string(index=False))

    # ─── Step 8: Volatility Surfaces ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 8: Building volatility surfaces...")
    print("=" * 70)
    for currency in ["BTC", "ETH"]:
        build_vol_surface(all_options, currency, results_dir)

    # ─── Step 9: Stress Tests ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 9: Running stress tests...")
    print("=" * 70)
    stress_df = run_stress_test(all_options, crash_vols, heston_params, r, results_dir)
    print("\n  Stress test results:")
    if not stress_df.empty:
        print(stress_df.to_string(index=False))

    # Compute Heston improvement under crash
    heston_crash_improvement = np.nan
    if not stress_df.empty:
        bs_crash = stress_df[
            (stress_df["regime"].isin(["covid_crash", "crypto_crash"])) &
            (stress_df["model"] == "BS")
        ]["mean_pct_error"].mean()
        heston_crash = stress_df[
            (stress_df["regime"].isin(["covid_crash", "crypto_crash"])) &
            (stress_df["model"] == "Heston")
        ]["mean_pct_error"].mean()
        heston_crash_improvement = bs_crash - heston_crash

    # ─── Step 10: Final Summary ─────────────────────────────────────────────────
    pipeline_time = time.time() - pipeline_start
    print("\n" + "=" * 70)
    print("=== FINAL SUMMARY ===")
    print("=" * 70)
    print(f"  Total BTC options priced: {n_btc}")
    print(f"  Total ETH options priced: {n_eth}")
    print(f"  Total contracts validated: {key_metrics['total_contracts']}")
    print(f"  BS mean pricing error: {key_metrics['bs_mean_pct_error']:.2f}%")
    print(f"  MC mean pricing error: {key_metrics['mc_mean_pct_error']:.2f}%")
    print(f"  Heston mean pricing error: {key_metrics['heston_mean_pct_error']:.2f}%")
    print(f"  Heston improvement over BS: {key_metrics['heston_improvement_over_bs']:.2f} percentage points")
    print(f"  Heston improvement over BS under crash volatility: {heston_crash_improvement:.2f} percentage points")
    print(f"  Total strike-maturity pairs for Greeks: {total_greek_pairs}")
    for currency in ["BTC", "ETH"]:
        hp = heston_params.get(currency, {})
        if hp:
            print(f"  Calibrated Heston params for {currency}: "
                  f"v0={hp['v0']:.4f}, kappa={hp['kappa']:.4f}, "
                  f"theta={hp['theta']:.4f}, sigma_v={hp['sigma_v']:.4f}, rho={hp['rho']:.4f}")
    print(f"\n  Pipeline completed in {pipeline_time:.1f}s")
    print(f"  Results saved in: {results_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
