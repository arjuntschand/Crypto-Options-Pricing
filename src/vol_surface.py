"""Implied volatility surface construction and visualization."""

from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def build_vol_surface(
    options_df: pd.DataFrame,
    currency: str = "BTC",
    results_dir: str = "results",
) -> Optional[pd.DataFrame]:
    """
    Build and plot the implied volatility surface from Deribit options data.
    """
    df = options_df[options_df["currency"] == currency].copy()
    if df.empty:
        print(f"  No {currency} options data for vol surface")
        return None

    spot = df["spot_price"].iloc[0]
    df["moneyness"] = df["strike"] / spot
    df["tte_days"] = df["time_to_expiry_years"] * 365.25
    df["iv_pct"] = df["mark_iv"] * 100.0

    # Filter reasonable moneyness range
    df = df[(df["moneyness"] > 0.3) & (df["moneyness"] < 3.0)].copy()
    df = df[df["tte_days"] > 0].copy()

    if len(df) < 10:
        print(f"  Not enough {currency} data points for vol surface ({len(df)})")
        return None

    # Create grid for interpolation
    moneyness_grid = np.linspace(df["moneyness"].min(), df["moneyness"].max(), 80)
    tte_grid = np.linspace(df["tte_days"].min(), df["tte_days"].max(), 80)
    M, TTE = np.meshgrid(moneyness_grid, tte_grid)

    # Interpolate
    points = df[["moneyness", "tte_days"]].values
    values = df["iv_pct"].values

    try:
        IV = griddata(points, values, (M, TTE), method="cubic")
    except Exception:
        IV = griddata(points, values, (M, TTE), method="linear")

    # Fill remaining NaN with nearest
    IV_nearest = griddata(points, values, (M, TTE), method="nearest")
    mask = np.isnan(IV)
    if mask.any():
        IV[mask] = IV_nearest[mask]

    # Plot 3D surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(M, TTE, IV, cmap="viridis", alpha=0.85, edgecolor="none")
    ax.set_xlabel("Moneyness (Strike/Spot)", fontsize=12)
    ax.set_ylabel("Time to Expiry (days)", fontsize=12)
    ax.set_zlabel("Implied Volatility (%)", fontsize=12)
    ax.set_title(f"{currency} Implied Volatility Surface", fontsize=14, fontweight="bold")
    ax.view_init(elev=25, azim=45)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="IV (%)")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/vol_surface_{currency}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved vol surface plot: {results_dir}/vol_surface_{currency}.png")

    # Save surface data
    surface_data = pd.DataFrame({
        "moneyness": M.ravel(),
        "tte_days": TTE.ravel(),
        "implied_vol_pct": IV.ravel(),
    })
    surface_data.to_csv(f"{results_dir}/vol_surface_data_{currency}.csv", index=False)
    print(f"  Saved vol surface data: {results_dir}/vol_surface_data_{currency}.csv")

    return surface_data
