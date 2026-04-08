"""Black-Scholes option pricing model."""

from typing import Optional

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "C"
) -> float:
    """
    Standard Black-Scholes European option pricing.

    S: spot price
    K: strike price
    T: time to expiry in years
    r: risk-free rate
    sigma: volatility (annualized)
    option_type: 'C' for call, 'P' for put
    Returns: option price as float
    """
    # Edge cases
    if T <= 0:
        if option_type == "C":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    if sigma <= 0:
        if option_type == "C":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "C":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return float(price)


def black_scholes_iv(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "C",
) -> Optional[float]:
    """
    Find implied volatility using Brent's method.
    Returns IV or NaN if no solution exists.
    """
    if market_price <= 0 or T <= 0:
        return np.nan

    def objective(sigma: float) -> float:
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price

    try:
        iv = brentq(objective, 0.01, 10.0, xtol=1e-8, maxiter=200)
        return float(iv)
    except (ValueError, RuntimeError):
        return np.nan
