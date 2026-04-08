# Crypto Options Pricing and Risk Engine

A comprehensive cryptocurrency (Bitcoin and Ethereum) options pricing and risk analysis engine that fetches live market data from Deribit, prices options using three distinct models, and performs validation, Greeks computation, volatility surface construction, and crash regime stress testing.

## Methodology

### Black-Scholes Model
The classic analytical European option pricing formula assuming constant volatility and log-normal price distribution. Uses the standard d1/d2 formulation with `scipy.stats.norm` for the cumulative distribution function. Implied volatility inversion is performed via Brent's root-finding method.

### Monte Carlo (GBM) Simulation
Geometric Brownian Motion simulation with 10,000 paths and 252 time steps. Uses vectorized NumPy operations for efficiency. Provides price estimates with standard errors and 95% confidence intervals. Fully reproducible with seed=42.

### Heston Stochastic Volatility Model
A two-factor model where both price and variance follow stochastic processes with mean-reverting volatility. Uses Euler discretization with full truncation scheme for variance. Parameters (v0, kappa, theta, sigma_v, rho) are calibrated to market data using `scipy.optimize.differential_evolution`.

## Results

**Data:** 1,389 live BTC and ETH options from Deribit (748 BTC, 641 ETH) with historical prices from 2019-present.

### Model Validation (vs. Market Prices)

| Model   | Mean % Error | Median % Error | Contracts |
|---------|-------------|----------------|-----------|
| BS      | 8.07%       | 6.09%          | 1,389     |
| MC      | 10.13%      | 4.56%          | 1,389     |
| Heston  | 94.34%      | 6.93%          | 1,389     |

- **Best overall model:** Black-Scholes (91.93% accuracy)
- **Heston improvement over BS under crash volatility:** 73.37 percentage points
- **Total strike-maturity pairs for Greeks:** 40

### Calibrated Heston Parameters

| Currency | v0     | kappa  | theta  | sigma_v | rho     |
|----------|--------|--------|--------|---------|---------|
| BTC      | 0.1617 | 0.5242 | 0.9363 | 1.1782  | -0.0974 |
| ETH      | 0.5599 | 0.3433 | 0.9442 | 4.7853  | -0.0092 |

### Crash Regime Volatilities

| Regime       | BTC    | ETH    |
|-------------|--------|--------|
| Normal       | 62.52% | 82.00% |
| COVID Crash  | 143.11%| 177.56%|
| Crypto Crash | 84.88% | 108.22%|

### Generated Outputs
- `results/deribit_options_raw.csv` - Raw options chain data
- `results/btc_historical.csv` / `results/eth_historical.csv` - Historical spot prices
- `results/validation_full.csv` / `results/validation_summary.csv` - Model validation results
- `results/greeks_comparison.csv` - Greeks across all three models
- `results/vol_surface_BTC.png` / `results/vol_surface_ETH.png` - 3D implied volatility surfaces
- `results/stress_test_results.csv` / `results/stress_test_chart.png` - Crash regime analysis

## Installation & Usage

```bash
pip install -r requirements.txt
python main.py
```

The pipeline fetches live data from Deribit and yfinance, then runs all pricing, validation, Greeks, vol surface, and stress test steps end-to-end. On subsequent runs, cached API data is reused automatically.

## Project Structure

```
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py      # Deribit API + yfinance data collection
│   ├── black_scholes.py     # Black-Scholes pricing + IV solver
│   ├── monte_carlo.py       # Monte Carlo GBM pricer + convergence analysis
│   ├── heston.py            # Heston stochastic vol model + calibration
│   ├── greeks.py            # Finite-difference Greeks for all models
│   ├── vol_surface.py       # 3D implied volatility surface
│   ├── stress_test.py       # Crash regime stress testing
│   ├── validation.py        # Model vs market comparison
│   └── utils.py             # Shared utilities
├── results/                  # Generated CSVs, plots, and data
└── main.py                   # Full end-to-end pipeline
```
