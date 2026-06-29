"""CLI entrypoint to execute the end-to-end stock model workflow."""

from __future__ import annotations

import argparse
import json

from stock_trader.config import PipelineConfig
from stock_trader.pipeline import run_end_to_end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stock-trader end-to-end pipeline.")
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated tickers. If omitted, uses the full US ticker source.",
    )
    parser.add_argument("--start", type=str, default="2018-01-01", help="Backtest start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default="2026-01-01", help="Backtest end date (YYYY-MM-DD).")
    parser.add_argument("--top-k", type=int, default=3, help="Number of daily stocks to select.")
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=500,
        help="Limit universe size for faster runs (set 0 for full universe).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size used while downloading price history.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    parsed_tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    limit = None if args.max_tickers == 0 else args.max_tickers

    config = PipelineConfig(
        tickers=parsed_tickers or None,
        ticker_source="custom" if parsed_tickers else "all_us",
        start=args.start,
        end=args.end,
        top_k=args.top_k,
        max_tickers=limit,
        download_batch_size=args.batch_size,
    )

    result = run_end_to_end(config)

    printable = {
        "config": result["config"],
        "resolved_ticker_count": result["resolved_ticker_count"],
        "latest_date": result["latest_date"],
        "metrics": result["metrics"],
        "today_picks": result["today_picks"].to_dict(orient="records"),
    }

    print(json.dumps(printable, indent=2))
