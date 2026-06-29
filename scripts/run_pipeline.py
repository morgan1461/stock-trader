"""CLI entrypoint to execute the end-to-end stock model workflow."""

from __future__ import annotations

import json

from stock_trader.config import PipelineConfig
from stock_trader.pipeline import run_end_to_end


if __name__ == "__main__":
    config = PipelineConfig()
    result = run_end_to_end(config)

    printable = {
        "config": result["config"],
        "latest_date": result["latest_date"],
        "metrics": result["metrics"],
        "today_picks": result["today_picks"].to_dict(orient="records"),
    }

    print(json.dumps(printable, indent=2))
