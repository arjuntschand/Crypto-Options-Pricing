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

from src.data_fetcher import fetch_all_data, get_crash_volatilities, CURRENCIES
from src.black_scholes import black_scholes_price
from src.monte_carlo import monte_carlo_price
from src.heston import (
    heston_mc_price,
    calibrate_heston_by_bucket,
    get_heston_params_for_option,
)
from src.greeks import compute_greeks_for_options
from src.vol_surface import build_vol_surface
from src.stress_test import run_stress_test
from src.validation import validate_models, detect_mispricings


def main() -> None:
    np.random.seed(42)
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    r = 0.05
    pipeline_start = time.time()

    # =========================================================================
    # Step 1: Data Collection
    # =========================================================================
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
        hist_prices = {}
        for cur in CURRENCIES:
            cache_path = f"{results_dir}/{cur.lower()}_historical.csv"
            if os.path.exists(cache_path):
                hist_prices[cur] = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        crash_vols = get_crash_volatilities(hist_prices)
        print(f"  Crash volatilities computed:")
        for regime, vols in crash_vols.items():
            parts = [f"{c}={v:.4f}" for c, v in vols.items()]
            print(f"    {regime}: {', '.join(parts)}")
    else:
        all_options, hist_prices, crash_vols = fetch_all_data(results_dir)

    # Count per currency
    counts = {}
    for cur in CURRENCIES:
        counts[cur] = len(all_options[all_options["currency"] == cur]) if not all_options.empty else 0
        print(f"  {cur} options fetched: {counts[cur]}")
    print(f"  Total options: {len(all_options)}")

    if all_options.empty:
        print("\nERROR: No options data collected. Exiting.")
        sys.exit(1)

    # =========================================================================
    # Step 2: Black-Scholes Pricing
    # =========================================================================
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

    # =========================================================================
    # Step 3: Monte Carlo Pricing
    # =========================================================================
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

    # =========================================================================
    # Step 4: Heston Calibration (per expiry bucket)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Calibrating Heston model (per expiry bucket)...")
    print("=" * 70)
    # heston_bucket_params: {currency: {bucket: {params}}}
    heston_bucket_params: dict = {}
    for currency in CURRENCIES:
        cdf = all_options[all_options["currency"] == currency]
        if cdf.empty:
            print(f"  Skipping {currency} Heston calibration (no data)")
            continue
        print(f"\n  === {currency} Heston Calibration ===")
        cal_start = time.time()
        bucket_params = calibrate_heston_by_bucket(
            all_options, currency, r,
            max_options_per_bucket=30,
            timeout_per_bucket=3.0,
        )
        heston_bucket_params[currency] = bucket_params
        print(f"  {currency} calibration completed in {time.time() - cal_start:.1f}s")

    # =========================================================================
    # Step 5: Heston Pricing (using per-bucket params)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Running Heston pricing (per-bucket params)...")
    print("=" * 70)
    heston_start = time.time()
    heston_prices = []
    for i, (_, row) in enumerate(all_options.iterrows()):
        if (i + 1) % 50 == 0:
            print(f"  Heston priced {i + 1}/{len(all_options)} options...")
        hp = get_heston_params_for_option(
            heston_bucket_params, row["currency"], row["time_to_expiry_years"]
        )
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

    # =========================================================================
    # Step 6: Validation
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Validating models against market...")
    print("=" * 70)
    full_val, summary_val, key_metrics = validate_models(
        all_options, heston_bucket_params, r, results_dir
    )

    # =========================================================================
    # Step 7: Mispricing Detection
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Detecting mispricings (>15% divergence)...")
    print("=" * 70)
    mispricing_df = detect_mispricings(full_val, threshold_pct=15.0, results_dir=results_dir)
    n_mispricings = len(mispricing_df)
    print(f"  Mispricings flagged (>15% divergence): {n_mispricings}")
    print(f"  Saved to {results_dir}/mispricings.csv")

    # =========================================================================
    # Step 8: Greeks
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: Computing Greeks...")
    print("=" * 70)
    # Build a flat heston_params dict for Greeks (use long-term params as representative)
    heston_params_flat: dict = {}
    for currency, buckets in heston_bucket_params.items():
        for fallback in ["long", "medium", "short"]:
            if fallback in buckets:
                heston_params_flat[currency] = buckets[fallback]
                break
    greeks_df, total_greek_pairs = compute_greeks_for_options(
        all_options, heston_params_flat, r, n_per_currency=20
    )
    greeks_df.to_csv(f"{results_dir}/greeks_comparison.csv", index=False)
    print(f"\n  Sample Greeks (first 5 rows):")
    if not greeks_df.empty:
        sample = greeks_df[["instrument", "option_type", "bs_delta", "bs_gamma", "bs_vega", "bs_theta"]].head(5)
        print(sample.to_string(index=False))

    # =========================================================================
    # Step 9: Volatility Surfaces
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 9: Building volatility surfaces...")
    print("=" * 70)
    for currency in CURRENCIES:
        if len(all_options[all_options["currency"] == currency]) > 0:
            build_vol_surface(all_options, currency, results_dir)

    # =========================================================================
    # Step 10: Stress Tests
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 10: Running stress tests...")
    print("=" * 70)
    stress_df = run_stress_test(
        all_options, crash_vols, heston_bucket_params, r, results_dir
    )
    print("\n  Stress test results:")
    if not stress_df.empty:
        print(stress_df.to_string(index=False))

    # Compute Heston improvement under crash
    heston_crash_improvement = np.nan
    bs_crash_mape = np.nan
    heston_crash_mape = np.nan
    if not stress_df.empty:
        bs_crash_mape = stress_df[
            (stress_df["regime"].isin(["covid_crash", "crypto_crash"])) &
            (stress_df["model"] == "BS")
        ]["mean_pct_error"].mean()
        heston_crash_mape = stress_df[
            (stress_df["regime"].isin(["covid_crash", "crypto_crash"])) &
            (stress_df["model"] == "Heston")
        ]["mean_pct_error"].mean()
        heston_crash_improvement = bs_crash_mape - heston_crash_mape

    # =========================================================================
    # Final Summary
    # =========================================================================
    pipeline_time = time.time() - pipeline_start
    print("\n" + "=" * 70)
    print("=== FINAL SUMMARY ===")
    print("=" * 70)

    print(f"\n  --- Contract Counts ---")
    for cur in CURRENCIES:
        print(f"  {cur} options priced: {counts.get(cur, 0)}")
    print(f"  Total contracts validated: {key_metrics['total_contracts']}")

    print(f"\n  --- Overall MAPE ---")
    print(f"  BS mean pricing error:     {key_metrics['bs_mean_pct_error']:.2f}%")
    print(f"  MC mean pricing error:     {key_metrics['mc_mean_pct_error']:.2f}%")
    print(f"  Heston mean pricing error: {key_metrics['heston_mean_pct_error']:.2f}%")

    print(f"\n  --- MAPE by Expiry Bucket ---")
    expiry_mape = key_metrics.get("expiry_mape", {})
    print(f"  {'Bucket':<10} {'BS':>10} {'MC':>10} {'Heston':>10}")
    print(f"  {'-'*42}")
    for eb in ["short", "medium", "long"]:
        bs_e = expiry_mape.get(eb, {}).get("BS", np.nan)
        mc_e = expiry_mape.get(eb, {}).get("MC", np.nan)
        he_e = expiry_mape.get(eb, {}).get("HESTON", np.nan)
        print(f"  {eb:<10} {bs_e:>9.2f}% {mc_e:>9.2f}% {he_e:>9.2f}%")

    print(f"\n  --- Heston Improvement ---")
    print(f"  Heston improvement over BS (normal):  {key_metrics['heston_improvement_over_bs']:.2f} pp")
    print(f"  Heston improvement over BS (crash):   {heston_crash_improvement:.2f} pp")

    print(f"\n  --- Mispricing Detection ---")
    print(f"  Mispricings flagged (>15% threshold): {n_mispricings}")

    print(f"\n  --- Greeks ---")
    print(f"  Total strike-maturity pairs for Greeks: {total_greek_pairs}")

    print(f"\n  --- Calibrated Heston Params (per bucket) ---")
    for currency in CURRENCIES:
        bucket_params = heston_bucket_params.get(currency, {})
        for bucket, hp in bucket_params.items():
            print(f"  {currency}-{bucket}: v0={hp['v0']:.4f}, kappa={hp['kappa']:.4f}, "
                  f"theta={hp['theta']:.4f}, sigma_v={hp['sigma_v']:.4f}, rho={hp['rho']:.4f}")

    print(f"\n  Pipeline completed in {pipeline_time:.1f}s")
    print(f"  Results saved in: {results_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
