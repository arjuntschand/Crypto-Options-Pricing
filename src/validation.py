"""Model validation against market prices."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.black_scholes import black_scholes_price
from src.monte_carlo import monte_carlo_price
from src.heston import heston_mc_price


def validate_models(
    options_df: pd.DataFrame,
    heston_params: Dict[str, Dict[str, float]],
    r: float = 0.05,
    results_dir: str = "results",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Validate all three models against market mark prices for every option.
    Uses pre-computed prices from DataFrame columns if available (bs_price, mc_price, heston_price).
    Returns (full_results_df, summary_df, key_metrics).
    """
    has_bs = "bs_price" in options_df.columns
    has_mc = "mc_price" in options_df.columns
    has_heston = "heston_price" in options_df.columns

    rows: List[Dict] = []
    total = len(options_df)

    for i, (_, row) in enumerate(options_df.iterrows()):
        if (i + 1) % 200 == 0:
            print(f"    Validating option {i + 1}/{total}...")

        S = row["spot_price"]
        K = row["strike"]
        T = row["time_to_expiry_years"]
        sigma = row["mark_iv"]
        opt_type = row["option_type"]
        market = row["mark_price"]
        currency = row["currency"]

        # Use pre-computed prices if available, otherwise compute
        bs_price = row["bs_price"] if has_bs else black_scholes_price(S, K, T, r, sigma, opt_type)

        if has_mc:
            mc_price = row["mc_price"]
        else:
            mc_res = monte_carlo_price(S, K, T, r, sigma, opt_type, n_paths=10000, n_steps=252, seed=42)
            mc_price = mc_res["price"]

        if has_heston and not np.isnan(row.get("heston_price", np.nan)):
            heston_price = row["heston_price"]
        else:
            h_params = heston_params.get(currency, {})
            if h_params:
                h_res = heston_mc_price(
                    S, K, T, r,
                    h_params["v0"], h_params["kappa"], h_params["theta"],
                    h_params["sigma_v"], h_params["rho"],
                    option_type=opt_type, n_paths=10000, n_steps=252, seed=42,
                )
                heston_price = h_res["price"]
            else:
                heston_price = np.nan

        moneyness = K / S
        if moneyness < 0.95:
            money_bucket = "ITM" if opt_type == "C" else "OTM"
        elif moneyness > 1.05:
            money_bucket = "OTM" if opt_type == "C" else "ITM"
        else:
            money_bucket = "ATM"

        tte_days = T * 365.25
        if tte_days < 30:
            expiry_bucket = "short"
        elif tte_days < 90:
            expiry_bucket = "medium"
        else:
            expiry_bucket = "long"

        result = {
            "currency": currency,
            "instrument": row["instrument_name"],
            "strike": K,
            "spot": S,
            "time_to_expiry": T,
            "option_type": opt_type,
            "moneyness": moneyness,
            "moneyness_bucket": money_bucket,
            "expiry_bucket": expiry_bucket,
            "market_price": market,
            "bs_price": bs_price,
            "mc_price": mc_price,
            "heston_price": heston_price,
            "bs_abs_error": abs(bs_price - market),
            "mc_abs_error": abs(mc_price - market),
            "heston_abs_error": abs(heston_price - market) if not np.isnan(heston_price) else np.nan,
        }

        if market > 0:
            result["bs_pct_error"] = abs(bs_price - market) / market * 100
            result["mc_pct_error"] = abs(mc_price - market) / market * 100
            result["heston_pct_error"] = abs(heston_price - market) / market * 100 if not np.isnan(heston_price) else np.nan
        else:
            result["bs_pct_error"] = np.nan
            result["mc_pct_error"] = np.nan
            result["heston_pct_error"] = np.nan

        rows.append(result)

    full_df = pd.DataFrame(rows)
    full_df.to_csv(f"{results_dir}/validation_full.csv", index=False)

    # Summary statistics
    summary_rows: List[Dict] = []
    for model in ["bs", "mc", "heston"]:
        pct_col = f"{model}_pct_error"
        abs_col = f"{model}_abs_error"
        valid = full_df[full_df[pct_col].notna()]

        overall = {
            "model": model.upper(),
            "category": "overall",
            "bucket": "all",
            "count": len(valid),
            "mean_abs_error": valid[abs_col].mean(),
            "mean_pct_error": valid[pct_col].mean(),
            "median_pct_error": valid[pct_col].median(),
            "max_abs_error": valid[abs_col].max(),
        }
        summary_rows.append(overall)

        # By option type
        for ot in ["C", "P"]:
            sub = valid[valid["option_type"] == ot]
            if len(sub) > 0:
                summary_rows.append({
                    "model": model.upper(),
                    "category": "option_type",
                    "bucket": ot,
                    "count": len(sub),
                    "mean_abs_error": sub[abs_col].mean(),
                    "mean_pct_error": sub[pct_col].mean(),
                    "median_pct_error": sub[pct_col].median(),
                    "max_abs_error": sub[abs_col].max(),
                })

        # By moneyness
        for mb in ["ITM", "ATM", "OTM"]:
            sub = valid[valid["moneyness_bucket"] == mb]
            if len(sub) > 0:
                summary_rows.append({
                    "model": model.upper(),
                    "category": "moneyness",
                    "bucket": mb,
                    "count": len(sub),
                    "mean_abs_error": sub[abs_col].mean(),
                    "mean_pct_error": sub[pct_col].mean(),
                    "median_pct_error": sub[pct_col].median(),
                    "max_abs_error": sub[abs_col].max(),
                })

        # By expiry
        for eb in ["short", "medium", "long"]:
            sub = valid[valid["expiry_bucket"] == eb]
            if len(sub) > 0:
                summary_rows.append({
                    "model": model.upper(),
                    "category": "expiry",
                    "bucket": eb,
                    "count": len(sub),
                    "mean_abs_error": sub[abs_col].mean(),
                    "mean_pct_error": sub[pct_col].mean(),
                    "median_pct_error": sub[pct_col].median(),
                    "max_abs_error": sub[abs_col].max(),
                })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{results_dir}/validation_summary.csv", index=False)

    # Key metrics
    bs_mean_pct = full_df["bs_pct_error"].mean()
    mc_mean_pct = full_df["mc_pct_error"].mean()
    heston_mean_pct = full_df["heston_pct_error"].mean()

    best_model = "Heston" if heston_mean_pct < bs_mean_pct and heston_mean_pct < mc_mean_pct else (
        "MC" if mc_mean_pct < bs_mean_pct else "BS"
    )
    best_error = min(bs_mean_pct, mc_mean_pct, heston_mean_pct)

    key_metrics = {
        "total_contracts": len(full_df),
        "bs_mean_pct_error": bs_mean_pct,
        "mc_mean_pct_error": mc_mean_pct,
        "heston_mean_pct_error": heston_mean_pct,
        "best_model": best_model,
        "best_accuracy": 100.0 - best_error,
        "heston_improvement_over_bs": bs_mean_pct - heston_mean_pct,
    }

    # Print summary
    print("\n  === VALIDATION SUMMARY ===")
    print(f"  Total contracts validated: {len(full_df)}")
    print(f"  {'Model':<10} {'Mean % Error':>14} {'Median % Error':>16} {'Max Abs Error':>14} {'Count':>6}")
    print(f"  {'-'*62}")
    for model in ["BS", "MC", "HESTON"]:
        row = summary_df[(summary_df["model"] == model) & (summary_df["bucket"] == "all")]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"  {model:<10} {r['mean_pct_error']:>13.2f}% {r['median_pct_error']:>15.2f}% {r['max_abs_error']:>14.2f} {int(r['count']):>6}")
    print(f"\n  Best model: {best_model} (accuracy: {key_metrics['best_accuracy']:.2f}%)")
    print(f"  Heston improvement over BS: {key_metrics['heston_improvement_over_bs']:.2f} percentage points")

    return full_df, summary_df, key_metrics
