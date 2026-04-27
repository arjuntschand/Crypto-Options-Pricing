"""Heston stochastic volatility model."""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from src.black_scholes import black_scholes_price


def heston_mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    option_type: str = "C",
    n_paths: int = 10000,
    n_steps: int = 252,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Heston model Monte Carlo pricing with full truncation scheme.

    Returns dict with: price, standard_error, ci_lower, ci_upper
    """
    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "C" else max(K - S, 0.0)
        return {"price": intrinsic, "standard_error": 0.0, "ci_lower": intrinsic, "ci_upper": intrinsic}

    rng = np.random.RandomState(seed)
    dt = T / n_steps

    # Generate correlated Brownian motions via Cholesky decomposition
    Z1 = rng.standard_normal((n_paths, n_steps))
    Z_indep = rng.standard_normal((n_paths, n_steps))
    rho_complement = np.sqrt(1.0 - rho ** 2)
    Z2 = rho * Z1 + rho_complement * Z_indep

    # Initialize variance and log-price arrays
    sqrt_dt = np.sqrt(dt)
    v = np.full(n_paths, v0)
    log_S = np.full(n_paths, np.log(S))

    for t in range(n_steps):
        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)

        # Update log-price
        log_S += (r - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * Z1[:, t]

        # Update variance with full truncation scheme
        v = v + kappa * (theta - v_pos) * dt + sigma_v * sqrt_v * sqrt_dt * Z2[:, t]
        v = np.maximum(v, 0.0)

    S_T = np.exp(log_S)

    if option_type == "C":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    discount = np.exp(-r * T)
    price = float(discount * np.mean(payoffs))
    se = float(discount * np.std(payoffs) / np.sqrt(n_paths))

    return {
        "price": price,
        "standard_error": se,
        "ci_lower": price - 1.96 * se,
        "ci_upper": price + 1.96 * se,
    }


def calibrate_heston(
    options_df: pd.DataFrame,
    S: float,
    r: float,
    max_options: int = 50,
    timeout_minutes: float = 5.0,
    label: str = "",
) -> Tuple[Dict[str, float], float]:
    """
    Calibrate Heston parameters to market data using differential evolution.

    Returns (params_dict, rmse).
    """
    df = options_df.copy()
    if len(df) > max_options:
        df = df.sample(n=max_options, random_state=42).reset_index(drop=True)

    if len(df) < 3:
        print(f"    Too few options to calibrate ({len(df)}), using defaults")
        return {"v0": 0.3, "kappa": 1.5, "theta": 0.3, "sigma_v": 0.8, "rho": -0.3}, 999.0

    market_prices = df["mark_price"].values
    strikes = df["strike"].values
    ttes = df["time_to_expiry_years"].values
    opt_types = df["option_type"].values
    spots = df["spot_price"].values

    call_count = 0
    start_time = time.time()

    def objective(params: np.ndarray) -> float:
        nonlocal call_count
        v0, kappa, theta, sigma_v, rho = params
        call_count += 1

        if call_count % 50 == 0:
            elapsed = time.time() - start_time
            print(f"    [{label}] Calibration iteration {call_count}, elapsed {elapsed:.1f}s")

        # Check timeout
        if (time.time() - start_time) > timeout_minutes * 60:
            return 1e10

        errors = []
        for i in range(len(df)):
            try:
                res = heston_mc_price(
                    S=spots[i], K=strikes[i], T=ttes[i], r=r,
                    v0=v0, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho,
                    option_type=opt_types[i],
                    n_paths=5000, n_steps=100, seed=42,
                )
                errors.append((res["price"] - market_prices[i]) ** 2)
            except Exception:
                errors.append(0.0)

        return float(np.mean(errors))

    bounds = [
        (0.01, 5.0),    # v0
        (0.1, 10.0),    # kappa
        (0.01, 5.0),    # theta
        (0.01, 5.0),    # sigma_v
        (-0.99, 0.0),   # rho
    ]

    print(f"    [{label}] Calibrating on {len(df)} options (S={S:.2f})...")
    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=30, tol=1e-4, popsize=10,
        mutation=(0.5, 1.0), recombination=0.7,
    )

    params = {
        "v0": result.x[0],
        "kappa": result.x[1],
        "theta": result.x[2],
        "sigma_v": result.x[3],
        "rho": result.x[4],
    }
    rmse = float(np.sqrt(result.fun))

    return params, rmse


def calibrate_heston_by_bucket(
    options_df: pd.DataFrame,
    currency: str,
    r: float = 0.05,
    max_options_per_bucket: int = 30,
    timeout_per_bucket: float = 3.0,
) -> Dict[str, Dict[str, float]]:
    """
    Calibrate Heston separately for short (<30d), medium (30-90d), and long (>90d) expiry buckets.

    Returns dict keyed by bucket name: {"short": {...params}, "medium": {...}, "long": {...}}
    """
    cdf = options_df[options_df["currency"] == currency].copy()
    if cdf.empty:
        return {}

    spot = cdf["spot_price"].iloc[0]
    tte_days = cdf["time_to_expiry_years"] * 365.25

    buckets = {
        "short": cdf[tte_days < 30],
        "medium": cdf[(tte_days >= 30) & (tte_days < 90)],
        "long": cdf[tte_days >= 90],
    }

    result: Dict[str, Dict[str, float]] = {}
    for bucket_name, bucket_df in buckets.items():
        if bucket_df.empty:
            print(f"    No {currency} options in {bucket_name} bucket, skipping")
            continue
        label = f"{currency}-{bucket_name}"
        print(f"\n  Calibrating Heston for {label} ({len(bucket_df)} options)...")
        params, rmse = calibrate_heston(
            bucket_df, spot, r,
            max_options=max_options_per_bucket,
            timeout_minutes=timeout_per_bucket,
            label=label,
        )
        result[bucket_name] = params
        print(f"  {label} RMSE: {rmse:.4f}")
        print(f"  {label} params: v0={params['v0']:.4f}, kappa={params['kappa']:.4f}, "
              f"theta={params['theta']:.4f}, sigma_v={params['sigma_v']:.4f}, rho={params['rho']:.4f}")

    return result


def get_heston_params_for_option(
    heston_bucket_params: Dict[str, Dict[str, Dict[str, float]]],
    currency: str,
    tte_years: float,
) -> Optional[Dict[str, float]]:
    """Look up the correct Heston params for a given currency and time to expiry."""
    bucket_params = heston_bucket_params.get(currency, {})
    if not bucket_params:
        return None

    tte_days = tte_years * 365.25
    if tte_days < 30:
        bucket = "short"
    elif tte_days < 90:
        bucket = "medium"
    else:
        bucket = "long"

    # Fall back to nearest available bucket
    if bucket in bucket_params:
        return bucket_params[bucket]
    for fallback in ["medium", "long", "short"]:
        if fallback in bucket_params:
            return bucket_params[fallback]
    return None
