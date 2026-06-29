"""Walk-forward backtesting utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from .model import predict, train_model


def walk_forward_backtest(feature_df: pd.DataFrame, top_k: int, random_state: int = 42) -> dict[str, pd.DataFrame | float]:
    """Perform rolling walk-forward simulation for buy-open/sell-close strategy."""
    dates = sorted(feature_df["date"].unique())
    if len(dates) < 120:
        raise ValueError("Need at least 120 trading days for a meaningful backtest.")

    warmup_days = 90
    trades: list[dict] = []
    pred_rows: list[pd.DataFrame] = []

    for i in range(warmup_days, len(dates) - 1):
        trade_date = dates[i]
        train = feature_df[feature_df["date"] < trade_date]
        today_universe = feature_df[feature_df["date"] == trade_date].copy()

        if train.empty or today_universe.empty:
            continue

        artifacts = train_model(train, random_state=random_state)
        today_universe["predicted_return"] = predict(today_universe, artifacts)

        selected = today_universe.sort_values("predicted_return", ascending=False).head(top_k)
        selected = selected.copy()
        selected["portfolio_weight"] = 1.0 / len(selected)
        selected["strategy_return"] = selected["portfolio_weight"] * selected["target_next_intraday"]
        selected["trade_date"] = trade_date
        trades.extend(selected.to_dict("records"))

        pred_rows.append(today_universe[["date", "ticker", "target_next_intraday", "predicted_return"]])

    trade_df = pd.DataFrame(trades)
    pred_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()

    if trade_df.empty or pred_df.empty:
        raise ValueError("Backtest did not produce any trades.")

    daily_returns = trade_df.groupby("trade_date", as_index=False)["strategy_return"].sum()
    daily_returns = daily_returns.rename(columns={"strategy_return": "portfolio_return"})
    daily_returns["equity_curve"] = (1.0 + daily_returns["portfolio_return"]).cumprod()

    mae = mean_absolute_error(pred_df["target_next_intraday"], pred_df["predicted_return"])
    r2 = r2_score(pred_df["target_next_intraday"], pred_df["predicted_return"])

    return {
        "trades": trade_df,
        "daily_returns": daily_returns,
        "predictions": pred_df,
        "mae": float(mae),
        "r2": float(r2),
        "total_return": float(daily_returns["equity_curve"].iloc[-1] - 1.0),
        "win_rate": float((daily_returns["portfolio_return"] > 0).mean()),
    }
