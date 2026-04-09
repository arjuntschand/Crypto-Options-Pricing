"""Data fetcher for Deribit options and historical spot prices."""

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf


DERIBIT_BASE = "https://www.deribit.com/api/v2/public"
MAX_RETRIES = 3
RATE_LIMIT_SLEEP = 0.05


def _api_call(url: str, params: Dict, max_retries: int = MAX_RETRIES) -> Optional[Dict]:
    """Make a Deribit API call with retries."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if "result" in data:
                return data["result"]
            print(f"  Warning: no 'result' in response for {params}")
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"  Warning: API call failed after {max_retries} retries: {e}")
                return None
    return None


def fetch_deribit_options(currency: str = "BTC") -> pd.DataFrame:
    """Fetch all active option instruments and their order book data from Deribit."""
    print(f"  Fetching {currency} instruments...")
    instruments = _api_call(
        f"{DERIBIT_BASE}/get_instruments",
        {"currency": currency, "kind": "option", "expired": "false"},
    )
    if not instruments:
        print(f"  Warning: Could not fetch {currency} instruments")
        return pd.DataFrame()

    print(f"  Found {len(instruments)} {currency} option instruments")

    rows: List[Dict] = []
    for i, inst in enumerate(instruments):
        if (i + 1) % 50 == 0:
            print(f"  Fetched order book for {i + 1}/{len(instruments)} {currency} instruments...")
        time.sleep(RATE_LIMIT_SLEEP)

        name = inst["instrument_name"]
        book = _api_call(
            f"{DERIBIT_BASE}/get_order_book",
            {"instrument_name": name},
        )
        if book is None:
            continue

        # Parse instrument name: BTC-28MAR26-100000-C
        parts = name.split("-")
        if len(parts) < 4:
            continue
        opt_type = parts[-1]  # C or P
        strike = float(parts[-2])
        expiry_str = parts[1]  # e.g. 28MAR26

        try:
            expiry_date = datetime.strptime(expiry_str, "%d%b%y").replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        now = datetime.now(timezone.utc)
        tte_days = (expiry_date - now).total_seconds() / 86400.0
        tte_years = tte_days / 365.25

        spot = book.get("underlying_price", book.get("index_price", 0))
        mark_price_usd = book.get("mark_price", 0)
        # Deribit returns mark_price in BTC/ETH, convert to USD
        if mark_price_usd and spot:
            mark_price_usd = mark_price_usd * spot
        mark_iv = book.get("mark_iv", 0)
        if mark_iv:
            mark_iv = mark_iv / 100.0  # Convert from percentage to decimal

        rows.append({
            "instrument_name": name,
            "currency": currency,
            "expiry_date": expiry_date,
            "strike": strike,
            "option_type": opt_type,
            "spot_price": spot,
            "mark_price": mark_price_usd,
            "mark_iv": mark_iv,
            "time_to_expiry_years": tte_years,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Filter
    df = df[df["time_to_expiry_years"] > 1.0 / 365.25].copy()
    df = df[df["mark_price"] > 0].copy()
    df = df[df["mark_iv"] > 0].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_historical_prices() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch historical BTC and ETH daily prices from yfinance."""
    print("  Fetching BTC-USD historical prices...")
    btc = yf.download("BTC-USD", start="2019-01-01", end=datetime.now().strftime("%Y-%m-%d"), progress=False)
    print("  Fetching ETH-USD historical prices...")
    eth = yf.download("ETH-USD", start="2019-01-01", end=datetime.now().strftime("%Y-%m-%d"), progress=False)

    # Flatten multi-level columns if present
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    if isinstance(eth.columns, pd.MultiIndex):
        eth.columns = eth.columns.get_level_values(0)

    return btc, eth


def compute_realized_vol(prices_df: pd.DataFrame, window: int = 30) -> pd.Series:
    """Compute rolling realized annualized volatility."""
    log_returns = np.log(prices_df["Close"] / prices_df["Close"].shift(1))
    return log_returns.rolling(window).std() * np.sqrt(365)


def get_crash_volatilities(btc_hist: pd.DataFrame, eth_hist: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Extract realized volatility for crash windows."""
    crash_windows = {
        "covid_crash": ("2020-02-15", "2020-04-15"),
        "crypto_crash": ("2022-05-01", "2022-07-01"),
    }

    result: Dict[str, Dict[str, float]] = {}
    for regime, (start, end) in crash_windows.items():
        btc_slice = btc_hist.loc[start:end]
        eth_slice = eth_hist.loc[start:end]
        btc_ret = np.log(btc_slice["Close"] / btc_slice["Close"].shift(1)).dropna()
        eth_ret = np.log(eth_slice["Close"] / eth_slice["Close"].shift(1)).dropna()
        result[regime] = {
            "BTC": float(btc_ret.std() * np.sqrt(365)),
            "ETH": float(eth_ret.std() * np.sqrt(365)),
        }

    # Normal regime: full history
    btc_ret_all = np.log(btc_hist["Close"] / btc_hist["Close"].shift(1)).dropna()
    eth_ret_all = np.log(eth_hist["Close"] / eth_hist["Close"].shift(1)).dropna()
    result["normal"] = {
        "BTC": float(btc_ret_all.std() * np.sqrt(365)),
        "ETH": float(eth_ret_all.std() * np.sqrt(365)),
    }

    return result


def fetch_all_data(results_dir: str = "results") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Fetch all data and save to CSV."""
    # Deribit options
    btc_options = fetch_deribit_options("BTC")
    eth_options = fetch_deribit_options("ETH")
    all_options = pd.concat([btc_options, eth_options], ignore_index=True)

    if not all_options.empty:
        all_options.to_csv(f"{results_dir}/deribit_options_raw.csv", index=False)
    print(f"  Total BTC options after filtering: {len(btc_options)}")
    print(f"  Total ETH options after filtering: {len(eth_options)}")

    # Historical prices
    btc_hist, eth_hist = fetch_historical_prices()
    btc_hist.to_csv(f"{results_dir}/btc_historical.csv")
    eth_hist.to_csv(f"{results_dir}/eth_historical.csv")

    # Crash volatilities
    crash_vols = get_crash_volatilities(btc_hist, eth_hist)
    print(f"  Crash volatilities computed:")
    for regime, vols in crash_vols.items():
        print(f"    {regime}: BTC={vols['BTC']:.4f}, ETH={vols['ETH']:.4f}")

    return all_options, btc_hist, eth_hist, crash_vols
