"""Utility functions for the crypto options pricing engine."""

import time
import functools
from typing import Any, Callable


def timer(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  [{func.__name__}] completed in {elapsed:.2f}s")
        return result
    return wrapper


RISK_FREE_RATE = 0.05
SEED = 42
TRADING_DAYS_PER_YEAR = 252
