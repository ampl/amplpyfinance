#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

# from builtins import map, range, object, zip, sorted

import unittest
import tempfile
import shutil
import os
import yfinance as yf

CACHE = {}

sector_mapper = {
    "MSFT": "Tech",
    "AMZN": "Consumer Discretionary",
    "KO": "Consumer Staples",
    "MA": "Financial Services",
    "COST": "Consumer Staples",
    "LUV": "Aerospace",
    "XOM": "Energy",
    "PFE": "Healthcare",
    "JPM": "Financial Services",
    "UNH": "Healthcare",
    "ACN": "Misc",
    "DIS": "Media",
    "GILD": "Healthcare",
    "F": "Auto",
    "TSLA": "Auto",
}

sector_lower = {
    "Consumer Staples": 0.1,  # at least 10% to staples
    "Tech": 0.05  # at least 5% to tech
    # For all other sectors, it will be assumed there is no lower bound
}

sector_upper = {"Tech": 0.2, "Aerospace": 0.1, "Energy": 0.1, "Auto": 0.15}


class TestBase(unittest.TestCase):
    def setUp(self):
        print("Method:", self._testMethodName)
        self._download_prices()

    def _download_prices(self):
        global CACHE
        if CACHE == {}:
            from pypfopt import expected_returns, risk_models

            tickers = [
                "MSFT",
                "AMZN",
                "KO",
                "MA",
                "COST",
                "LUV",
                "XOM",
                "PFE",
                "JPM",
                "UNH",
                "ACN",
                "DIS",
                "GILD",
                "F",
                "TSLA",
            ]
            ohlc = yf.download(tickers, period="max")
            CACHE["prices"] = ohlc["Adj Close"].dropna(how="all")
            CACHE["mu"] = expected_returns.capm_return(CACHE["prices"])
            CACHE["S"] = risk_models.CovarianceShrinkage(CACHE["prices"]).ledoit_wolf()
        self.prices = CACHE["prices"]
        self.S = CACHE["S"]
        self.mu = CACHE["mu"]

    def assertEqualWeights(self, weights1, weights2, eps=1e-6):
        if not isinstance(weights1, list):
            weights1 = weights1.values()
        if not isinstance(weights2, list):
            weights2 = weights2.values()
        self.assertEqual(
            sum(abs(w1 - w2) > eps for (w1, w2) in zip(weights1, weights2)),
            0,
        )

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
