"""End-to-end orchestrator for stock selection workflow."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .backtest import walk_forward_backtest
from .config import PipelineConfig
from .data import fetch_fundamentals, fetch_ohlcv
from .features import build_feature_frame
from .model import predict, train_model


def run_end_to_end(config: PipelineConfig | None = None) -> dict:
    """Run data collection, feature generation, training, backtesting, and latest picks."""
    cfg = config or PipelineConfig()

    prices = fetch_ohlcv(cfg.tickers, cfg.start, cfg.end)
    fundamentals = fetch_fundamentals(cfg.tickers)
    feature_df = build_feature_frame(prices, fundamentals)

    results = walk_forward_backtest(feature_df, top_k=cfg.top_k, random_state=cfg.random_state)

    latest_date = feature_df["date"].max()
    latest_slice = feature_df[feature_df["date"] == latest_date].copy()
    artifacts = train_model(feature_df[feature_df["date"] < latest_date], random_state=cfg.random_state)
    latest_slice["predicted_return"] = predict(latest_slice, artifacts)

    picks = latest_slice.sort_values("predicted_return", ascending=False).head(cfg.top_k)

    return {
        "config": asdict(cfg),
        "metrics": {
            "mae": results["mae"],
            "r2": results["r2"],
            "total_return": results["total_return"],
            "win_rate": results["win_rate"],
        },
        "latest_date": pd.Timestamp(latest_date).strftime("%Y-%m-%d"),
        "daily_returns": results["daily_returns"],
        "trades": results["trades"],
        "predictions": results["predictions"],
        "today_picks": picks[["ticker", "predicted_return"]].reset_index(drop=True),
    }
