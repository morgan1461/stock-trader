"""Configuration values for the stock trading workflow."""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    tickers: list[str] | None = None
    ticker_source: str = "all_us"
    start: str = "2018-01-01"
    end: str = "2026-01-01"
    top_k: int = 3
    random_state: int = 42
    max_tickers: int | None = None
    ticker_cache_path: str = "data/us_tickers.csv"
    download_batch_size: int = 100
