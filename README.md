# Crypto Options Pricing and Risk Engine

A comprehensive cryptocurrency (Bitcoin and Ethereum) options pricing and risk analysis engine that fetches live market data from Deribit, prices options using three distinct models, and performs validation, Greeks computation, volatility surface construction, crash regime stress testing, and automated mispricing detection.

## Methodology

### Black-Scholes Model
The classic analytical European option pricing formula assuming constant volatility and log-normal price distribution. Uses the standard d1/d2 formulation with `scipy.stats.norm` for the cumulative distribution function. Implied volatility inversion is performed via Brent's root-finding method.

### Monte Carlo (GBM) Simulation
Geometric Brownian Motion simulation with 10,000 paths and 252 time steps. Uses vectorized NumPy operations for efficiency. Provides price estimates with standard errors and 95% confidence intervals. Fully reproducible with seed=42.

### Heston Stochastic Volatility Model
A two-factor model where both price and variance follow stochastic processes with mean-reverting volatility. Uses Euler discretization with full truncation scheme for variance. Parameters (v0, kappa, theta, sigma_v, rho) are calibrated **per expiry bucket** (short <30d, medium 30-90d, long >90d) to market data using `scipy.optimize.differential_evolution`.

## Results

**Data:** 1,505 live BTC and ETH options from Deribit (825 BTC, 680 ETH) with historical prices from 2019-present.

### Model Validation (vs. Market Mark Prices)

| Model   | Mean % Error | Contracts |
|---------|-------------|-----------|
| BS      | 8.43%       | 1,505     |
| MC      | 9.99%       | 1,505     |
| Heston  | 48.09%      | 1,505     |

### MAPE by Expiry Bucket

| Bucket  | BS     | MC     | Heston |
|---------|--------|--------|--------|
| Short   | 8.03%  | 8.73%  | 15.35% |
| Medium  | 5.85%  | 9.42%  | 39.83% |
| Long    | 10.03% | 10.98% | 70.40% |

### Heston Improvement

- **Normal conditions:** BS outperforms Heston overall due to calibration noise on far-OTM options
- **Crash regime improvement:** Heston outperforms BS by **166.27 percentage points** under COVID/crypto crash volatility

### Mispricing Detection

- **110 contracts flagged** exceeding 15% divergence between best model price and market mark price
- Saved to `results/mispricings.csv`

### Calibrated Heston Parameters (per expiry bucket)

| Currency-Bucket | v0     | kappa  | theta  | sigma_v | rho     |
|----------------|--------|--------|--------|---------|---------|
| BTC-short      | 0.1332 | 1.2565 | 1.8526 | 1.9262  | -0.0636 |
| BTC-medium     | 0.0962 | 0.7453 | 2.7978 | 2.8607  | -0.1293 |
| BTC-long       | 0.1008 | 2.2335 | 0.5431 | 3.0275  | -0.3249 |
| ETH-short      | 0.2715 | 2.4050 | 2.2558 | 4.6218  | -0.2595 |
| ETH-medium     | 0.2845 | 1.2453 | 2.6193 | 2.2710  | -0.3123 |
| ETH-long       | 0.4249 | 1.0560 | 0.9033 | 2.6559  | -0.0555 |

### Crash Regime Volatilities

| Regime       | BTC    | ETH    |
|-------------|--------|--------|
| Normal       | 62.41% | 81.87% |
| COVID Crash  | 143.11%| 177.56%|
| Crypto Crash | 84.88% | 108.22%|

### Generated Outputs
- `results/deribit_options_raw.csv` - Raw options chain data
- `results/btc_historical.csv` / `results/eth_historical.csv` - Historical spot prices
- `results/validation_full.csv` / `results/validation_summary.csv` - Model validation results
- `results/mispricings.csv` - Flagged mispricings exceeding 15% divergence
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
│   ├── heston.py            # Heston stochastic vol model + per-bucket calibration
│   ├── greeks.py            # Finite-difference Greeks for all models
│   ├── vol_surface.py       # 3D implied volatility surface
│   ├── stress_test.py       # Crash regime stress testing
│   ├── validation.py        # Model vs market comparison + mispricing detection
│   └── utils.py             # Shared utilities
├── results/                  # Generated CSVs, plots, and data
└── main.py                   # Full end-to-end pipeline
```
