"""Greeks computation via finite differences."""

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.black_scholes import black_scholes_price
from src.monte_carlo import monte_carlo_price
from src.heston import heston_mc_price


def compute_greeks(
    pricer_func: Callable,
    base_args: Dict,
    S: float,
    option_type: str = "C",
) -> Dict[str, float]:
    """
    Compute Greeks via central finite differences.

    pricer_func: function that takes **kwargs and returns a price (float or dict with 'price')
    base_args: dict of arguments to pricer_func
    S: spot price (used to scale delta/gamma bumps)
    """
    def get_price(args: Dict) -> float:
        result = pricer_func(**args)
        if isinstance(result, dict):
            return result["price"]
        return result

    base_price = get_price(base_args)
    dS = S * 0.01  # 1% bump

    # Delta: dPrice/dS
    args_up = {**base_args, "S": S + dS}
    args_down = {**base_args, "S": S - dS}
    price_up = get_price(args_up)
    price_down = get_price(args_down)
    delta = (price_up - price_down) / (2.0 * dS)

    # Gamma: d2Price/dS2
    gamma = (price_up - 2.0 * base_price + price_down) / (dS ** 2)

    # Vega: dPrice/d_sigma (or dv0 for Heston)
    d_sigma = 0.01
    if "sigma" in base_args:
        args_vup = {**base_args, "sigma": base_args["sigma"] + d_sigma}
        args_vdn = {**base_args, "sigma": max(base_args["sigma"] - d_sigma, 0.001)}
    elif "v0" in base_args:
        args_vup = {**base_args, "v0": base_args["v0"] + d_sigma}
        args_vdn = {**base_args, "v0": max(base_args["v0"] - d_sigma, 0.001)}
    else:
        args_vup = base_args
        args_vdn = base_args
    vega = (get_price(args_vup) - get_price(args_vdn)) / (2.0 * d_sigma)

    # Theta: dPrice/dT (negative of time decay)
    dT = 1.0 / 365.0
    T = base_args.get("T", 0)
    if T > dT:
        args_tdn = {**base_args, "T": T - dT}
        theta = (get_price(args_tdn) - base_price) / dT  # price change per day
    else:
        theta = 0.0

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
    }


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "C"
) -> Dict[str, float]:
    """Compute Greeks for Black-Scholes model."""
    base_args = {"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "option_type": option_type}
    return compute_greeks(black_scholes_price, base_args, S, option_type)


def mc_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "C",
    n_paths: int = 10000, n_steps: int = 252,
) -> Dict[str, float]:
    """Compute Greeks for Monte Carlo GBM model."""
    base_args = {
        "S": S, "K": K, "T": T, "r": r, "sigma": sigma, "option_type": option_type,
        "n_paths": n_paths, "n_steps": n_steps, "seed": 42,
    }
    return compute_greeks(monte_carlo_price, base_args, S, option_type)


def heston_greeks(
    S: float, K: float, T: float, r: float,
    v0: float, kappa: float, theta: float, sigma_v: float, rho: float,
    option_type: str = "C",
    n_paths: int = 10000, n_steps: int = 252,
) -> Dict[str, float]:
    """Compute Greeks for Heston model."""
    base_args = {
        "S": S, "K": K, "T": T, "r": r,
        "v0": v0, "kappa": kappa, "theta": theta, "sigma_v": sigma_v, "rho": rho,
        "option_type": option_type, "n_paths": n_paths, "n_steps": n_steps, "seed": 42,
    }
    return compute_greeks(heston_mc_price, base_args, S, option_type)


def compute_greeks_for_options(
    options_df: pd.DataFrame,
    heston_params: Dict[str, Dict[str, float]],
    r: float = 0.05,
    n_per_currency: int = 20,
) -> pd.DataFrame:
    """
    Compute Greeks for a selection of options across all three models.
    Returns DataFrame with Greeks for each model.
    """
    results: List[Dict] = []
    total_pairs = 0

    for currency in options_df["currency"].unique():
        cdf = options_df[options_df["currency"] == currency].copy()
        if cdf.empty:
            continue

        # Select diverse subset
        n_select = min(n_per_currency, len(cdf))
        selected = cdf.sample(n=n_select, random_state=42).reset_index(drop=True)
        h_params = heston_params.get(currency, {})

        for idx, row in selected.iterrows():
            S = row["spot_price"]
            K = row["strike"]
            T = row["time_to_expiry_years"]
            sigma = row["mark_iv"]
            opt_type = row["option_type"]

            total_pairs += 1

            # BS Greeks
            bs_g = bs_greeks(S, K, T, r, sigma, opt_type)

            # MC Greeks (use fewer paths for speed)
            mc_g = mc_greeks(S, K, T, r, sigma, opt_type, n_paths=5000, n_steps=100)

            # Heston Greeks
            if h_params:
                h_g = heston_greeks(
                    S, K, T, r,
                    h_params["v0"], h_params["kappa"], h_params["theta"],
                    h_params["sigma_v"], h_params["rho"],
                    opt_type, n_paths=5000, n_steps=100,
                )
            else:
                h_g = {"delta": np.nan, "gamma": np.nan, "vega": np.nan, "theta": np.nan}

            results.append({
                "currency": currency,
                "instrument": row["instrument_name"],
                "strike": K,
                "expiry": row["expiry_date"],
                "time_to_expiry": T,
                "option_type": opt_type,
                "spot": S,
                "mark_iv": sigma,
                "bs_delta": bs_g["delta"],
                "bs_gamma": bs_g["gamma"],
                "bs_vega": bs_g["vega"],
                "bs_theta": bs_g["theta"],
                "mc_delta": mc_g["delta"],
                "mc_gamma": mc_g["gamma"],
                "mc_vega": mc_g["vega"],
                "mc_theta": mc_g["theta"],
                "heston_delta": h_g["delta"],
                "heston_gamma": h_g["gamma"],
                "heston_vega": h_g["vega"],
                "heston_theta": h_g["theta"],
            })

            if len(results) % 5 == 0:
                print(f"    Computed Greeks for {len(results)} options...")

    df = pd.DataFrame(results)
    print(f"  Total unique strike-maturity pairs for Greeks: {total_pairs}")
    return df, total_pairs
