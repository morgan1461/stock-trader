"""Feature engineering for daily open-to-close return prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_feature_frame(raw_prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features and target labels from market data."""
    df = raw_prices.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df["daily_return"] = df["close"].pct_change()
    df["intraday_return"] = (df["close"] - df["open"]) / df["open"]

    grouped = df.groupby("ticker", group_keys=False)
    df["ret_1d"] = grouped["close"].pct_change(1)
    df["ret_2d"] = grouped["close"].pct_change(2)
    df["ret_5d"] = grouped["close"].pct_change(5)
    df["vol_5d"] = grouped["close"].pct_change().rolling(5).std().reset_index(level=0, drop=True)
    df["mom_10d"] = grouped["close"].pct_change(10)
    df["volume_ratio"] = df["volume"] / grouped["volume"].rolling(20).mean().reset_index(level=0, drop=True)
    df["range_ratio"] = (df["high"] - df["low"]) / df["open"]

    df["target_next_intraday"] = grouped["intraday_return"].shift(-1)

    feat = df.merge(fundamentals, on="ticker", how="left")

    feat["market_cap_log"] = np.log1p(feat["market_cap"])
    feat["shares_log"] = np.log1p(feat["shares"])

    cols_to_keep = [
        "date",
        "ticker",
        "open",
        "close",
        "intraday_return",
        "target_next_intraday",
        "ret_1d",
        "ret_2d",
        "ret_5d",
        "vol_5d",
        "mom_10d",
        "volume_ratio",
        "range_ratio",
        "market_cap_log",
        "shares_log",
        "last_price",
    ]

    feat = feat[cols_to_keep]
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna().reset_index(drop=True)
    return feat


FEATURE_COLUMNS = [
    "ret_1d",
    "ret_2d",
    "ret_5d",
    "vol_5d",
    "mom_10d",
    "volume_ratio",
    "range_ratio",
    "market_cap_log",
    "shares_log",
    "last_price",
]
