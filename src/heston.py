"""Heston stochastic volatility model."""

import time
from typing import Dict, Optional, Tuple

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
) -> Tuple[Dict[str, float], float]:
    """
    Calibrate Heston parameters to market data using differential evolution.

    Returns (params_dict, rmse).
    """
    df = options_df.copy()
    if len(df) > max_options:
        df = df.sample(n=max_options, random_state=42).reset_index(drop=True)

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
            print(f"    Calibration iteration {call_count}, elapsed {elapsed:.1f}s")

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

    print(f"    Calibrating on {len(df)} options (S={S:.2f})...")
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
