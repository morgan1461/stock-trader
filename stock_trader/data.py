"""Data loading utilities using public Yahoo Finance endpoints."""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_ohlcv(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch historical OHLCV data for multiple tickers from Yahoo Finance."""
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if data.empty:
            logger.warning("No data returned for %s", ticker)
            continue

        data = data.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        data["ticker"] = ticker
        data.index.name = "date"
        frames.append(data.reset_index())

    if not frames:
        raise ValueError("No ticker data could be downloaded.")

    full_df = pd.concat(frames, ignore_index=True)
    full_df["date"] = pd.to_datetime(full_df["date"])
    return full_df


def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """Fetch lightweight ticker-level fundamentals from Yahoo metadata."""
    rows: list[dict] = []
    for ticker in tickers:
        info = {}
        try:
            info = yf.Ticker(ticker).fast_info
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Unable to fetch fundamentals for %s: %s", ticker, exc)

        rows.append(
            {
                "ticker": ticker,
                "market_cap": _safe_float(info.get("market_cap")),
                "shares": _safe_float(info.get("shares")),
                "last_price": _safe_float(info.get("last_price")),
            }
        )

    fundamentals = pd.DataFrame(rows)
    for col in ["market_cap", "shares", "last_price"]:
        median = fundamentals[col].median()
        fundamentals[col] = fundamentals[col].fillna(median if pd.notna(median) else 0.0)
    return fundamentals


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
