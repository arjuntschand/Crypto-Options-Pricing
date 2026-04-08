"""Monte Carlo GBM option pricing."""

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def monte_carlo_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "C",
    n_paths: int = 10000,
    n_steps: int = 252,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Monte Carlo pricing under Geometric Brownian Motion.

    Returns dict with: price, standard_error, ci_lower, ci_upper
    """
    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "C" else max(K - S, 0.0)
        return {"price": intrinsic, "standard_error": 0.0, "ci_lower": intrinsic, "ci_upper": intrinsic}

    rng = np.random.RandomState(seed)
    dt = T / n_steps

    # Generate all random normals at once (vectorized)
    Z = rng.standard_normal((n_paths, n_steps))

    # Compute log increments
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * Z

    # Cumulative sum of log increments
    log_paths = np.cumsum(drift + diffusion, axis=1)
    S_T = S * np.exp(log_paths[:, -1])

    # Payoffs
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


def convergence_analysis(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "C",
    path_counts: List[int] = None,
) -> pd.DataFrame:
    """Run MC pricing at various path counts and measure convergence."""
    if path_counts is None:
        path_counts = [100, 500, 1000, 5000, 10000, 50000]

    results = []
    for n in path_counts:
        start = time.time()
        res = monte_carlo_price(S, K, T, r, sigma, option_type, n_paths=n)
        elapsed = time.time() - start
        results.append({
            "n_paths": n,
            "price": res["price"],
            "standard_error": res["standard_error"],
            "ci_lower": res["ci_lower"],
            "ci_upper": res["ci_upper"],
            "time_seconds": elapsed,
        })

    return pd.DataFrame(results)
