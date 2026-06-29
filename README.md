# stock-trader

End-to-end Python workflow for a **buy-at-open / sell-at-close** daily stock selection strategy.

## What this project does

- Pulls public historical stock data from Yahoo Finance
- Builds a broad ticker universe from SEC exchange listings (or custom ticker lists)
- Builds predictive features from historical price/volume behavior plus company metadata
- Trains a regression model to predict each stock's **next-day intraday return**
- Runs a walk-forward backtest over historical data
- Produces a daily ranked list of stocks to buy in the morning and sell at close
- Includes a Jupyter notebook for transparent, step-by-step analysis

## Project structure

- `/stock_trader/config.py` – pipeline configuration
- `/stock_trader/data.py` – ticker-universe resolution and public data ingestion
- `/stock_trader/features.py` – feature engineering
- `/stock_trader/model.py` – model training/inference
- `/stock_trader/backtest.py` – walk-forward backtesting
- `/stock_trader/pipeline.py` – full orchestration
- `/scripts/run_pipeline.py` – CLI entrypoint
- `/notebooks/end_to_end_workflow.ipynb` – notebook walkthrough
- `/tests` – unit tests

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the workflow

### Easy default run (recommended)
Uses the SEC universe, but limits to a manageable subset so it runs quickly.

```bash
python scripts/run_pipeline.py
```

### Full-universe run
Set `--max-tickers 0` to use the entire resolved stock universe.

```bash
python scripts/run_pipeline.py --max-tickers 0
```

### Custom ticker run

```bash
python scripts/run_pipeline.py --tickers AAPL,MSFT,NVDA --top-k 2
```

Output includes:

- Resolved ticker count
- Model metrics (MAE, R²)
- Strategy metrics (total return, win rate)
- Latest ranked stocks to buy at open and sell at close

## Run tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Modeling approach (high-level)

1. Resolve the stock universe (all-US source or custom list).
2. Download OHLCV history in batches for stability and speed.
3. Compute technical predictors (lagged returns, momentum, volatility, volume ratio, intraday range).
4. Add static ticker-level metadata (market cap, shares, last price).
5. Train a Gradient Boosting regressor.
6. Evaluate in a strict walk-forward simulation:
   - Train on all prior days.
   - Predict on current day.
   - Select top-k predicted stocks.
   - Realize return using the next day's open-to-close target.

This setup is intentionally simple and interpretable, while still being fully automated and testable.
