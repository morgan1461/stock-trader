import unittest

import numpy as np
import pandas as pd

from stock_trader.backtest import walk_forward_backtest
from stock_trader.features import FEATURE_COLUMNS


class BacktestTests(unittest.TestCase):
    def test_walk_forward_backtest_generates_metrics(self):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2023-01-02", periods=170)
        rows = []
        for ticker in ["AAA", "BBB", "CCC", "DDD"]:
            for dt in dates:
                row = {
                    "date": dt,
                    "ticker": ticker,
                    "open": 100.0,
                    "close": 100.1,
                    "intraday_return": rng.normal(0.001, 0.01),
                    "target_next_intraday": rng.normal(0.001, 0.01),
                }
                for c in FEATURE_COLUMNS:
                    row[c] = rng.normal(0, 1)
                rows.append(row)

        feature_df = pd.DataFrame(rows)

        result = walk_forward_backtest(feature_df, top_k=2, random_state=42)

        self.assertIn("total_return", result)
        self.assertIn("mae", result)
        self.assertIn("win_rate", result)
        self.assertGreater(len(result["daily_returns"]), 0)


if __name__ == "__main__":
    unittest.main()
