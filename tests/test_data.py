import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stock_trader.data import fetch_all_us_tickers, resolve_tickers


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


class DataUniverseTests(unittest.TestCase):
    def test_fetch_all_us_tickers_from_sec_payload(self):
        payload = {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [1, "Alpha", "AAA", "Nasdaq"],
                [2, "Beta", "BBB", "NYSE"],
                [3, "Gamma", "CCC", "OTC"],
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "us_tickers.csv"
            with patch("stock_trader.data.urlopen", return_value=_FakeResponse(payload)):
                tickers = fetch_all_us_tickers(str(cache), max_tickers=None)

            self.assertEqual(tickers, ["AAA", "BBB"])
            self.assertTrue(cache.exists())

    def test_resolve_custom_tickers(self):
        tickers = resolve_tickers(
            tickers=[" msft ", "AAPL", "MSFT"],
            ticker_source="custom",
            cache_path="unused.csv",
            max_tickers=None,
        )

        self.assertEqual(tickers, ["AAPL", "MSFT"])


if __name__ == "__main__":
    unittest.main()
