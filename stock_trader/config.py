"""Configuration values for the stock trading workflow."""

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    tickers: list[str] = field(
        default_factory=lambda: [
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOGL",
            "META",
            "TSLA",
            "JPM",
            "XOM",
            "UNH",
        ]
    )
    start: str = "2018-01-01"
    end: str = "2026-01-01"
    top_k: int = 3
    random_state: int = 42
