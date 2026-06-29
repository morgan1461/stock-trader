import unittest

import pandas as pd

from stock_trader.features import FEATURE_COLUMNS, build_feature_frame


class FeatureEngineeringTests(unittest.TestCase):
    def test_build_feature_frame_outputs_required_columns(self):
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        rows = []
        for ticker in ["AAA", "BBB"]:
            for i, dt in enumerate(dates):
                base = 100 + i
                rows.append(
                    {
                        "date": dt,
                        "ticker": ticker,
                        "open": base,
                        "high": base * 1.01,
                        "low": base * 0.99,
                        "close": base * 1.002,
                        "adj_close": base * 1.002,
                        "volume": 1_000_000 + i * 10,
                    }
                )

        raw = pd.DataFrame(rows)
        fundamentals = pd.DataFrame(
            [
                {"ticker": "AAA", "market_cap": 1_000_000_000, "shares": 1_000_000, "last_price": 100},
                {"ticker": "BBB", "market_cap": 2_000_000_000, "shares": 2_000_000, "last_price": 200},
            ]
        )

        out = build_feature_frame(raw, fundamentals)

        for col in FEATURE_COLUMNS + ["target_next_intraday", "date", "ticker"]:
            self.assertIn(col, out.columns)

        self.assertGreater(len(out), 0)


if __name__ == "__main__":
    unittest.main()
