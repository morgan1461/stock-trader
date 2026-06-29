"""Data loading utilities using public Yahoo Finance and SEC data."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers_exchange.json"


def resolve_tickers(
    tickers: list[str] | None,
    ticker_source: str,
    cache_path: str,
    max_tickers: int | None = None,
) -> list[str]:
    """Resolve ticker universe from either explicit tickers or a source."""
    if tickers:
        cleaned = sorted({str(t).strip().upper() for t in tickers if t is not None and str(t).strip()})
        if not cleaned:
            raise ValueError("No valid custom tickers were provided.")
        return cleaned[:max_tickers] if max_tickers else cleaned

    if ticker_source == "all_us":
        return fetch_all_us_tickers(cache_path=cache_path, max_tickers=max_tickers)

    raise ValueError(f"Unsupported ticker source: {ticker_source}")


def fetch_all_us_tickers(cache_path: str, max_tickers: int | None = None) -> list[str]:
    """Fetch a broad US stock universe from SEC exchange data with caching."""
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    dataframe: pd.DataFrame | None = None
    try:
        request = Request(
            SEC_TICKER_URL,
            headers={
                "User-Agent": "stock-trader-bot/1.0 (research@example.com)",
                "Accept": "application/json",
            },
        )
        with urlopen(request, timeout=30) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))

        fields = payload.get("fields", [])
        rows = payload.get("data", [])
        dataframe = pd.DataFrame(rows, columns=fields)
        dataframe.to_csv(cache_file, index=False)
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Failed to fetch SEC ticker universe: %s", exc)
        if cache_file.exists():
            dataframe = pd.read_csv(cache_file)
        else:
            raise ValueError("Could not fetch ticker universe and no local cache exists.") from exc

    if dataframe is None or dataframe.empty:
        raise ValueError("Ticker universe is empty.")

    if "exchange" in dataframe.columns:
        allowed = {"Nasdaq", "NYSE", "NYSE American", "Cboe", "NYSE Arca"}
        dataframe = dataframe[dataframe["exchange"].isin(allowed)]

    tickers = (
        dataframe["ticker"].dropna().astype(str).str.upper().str.strip().drop_duplicates().tolist()
    )
    if not tickers:
        raise ValueError("No tickers available after filtering.")

    return tickers[:max_tickers] if max_tickers else tickers


def fetch_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Fetch historical OHLCV data in batches to handle large universes."""
    frames: list[pd.DataFrame] = []

    for i in range(0, len(tickers), max(batch_size, 1)):
        batch = tickers[i : i + batch_size]
        if not batch:
            continue

        downloaded = yf.download(
            batch,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )

        frames.extend(_normalize_download(batch, downloaded))

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


def _normalize_download(batch: list[str], downloaded: pd.DataFrame) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []

    if downloaded.empty:
        for ticker in batch:
            logger.warning("No data returned for %s", ticker)
        return frames

    if isinstance(downloaded.columns, pd.MultiIndex):
        for ticker in batch:
            if ticker not in downloaded.columns.get_level_values(0):
                logger.warning("No data returned for %s", ticker)
                continue
            ticker_frame = downloaded[ticker].dropna(how="all")
            normalized = _rename_ohlcv_columns(ticker_frame)
            if normalized.empty:
                logger.warning("No data returned for %s", ticker)
                continue
            normalized["ticker"] = ticker
            normalized.index.name = "date"
            frames.append(normalized.reset_index())
        return frames

    ticker = batch[0]
    normalized = _rename_ohlcv_columns(downloaded.dropna(how="all"))
    if normalized.empty:
        logger.warning("No data returned for %s", ticker)
        return frames

    normalized["ticker"] = ticker
    normalized.index.name = "date"
    frames.append(normalized.reset_index())
    return frames


def _rename_ohlcv_columns(data: pd.DataFrame) -> pd.DataFrame:
    return data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
