#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from . import TestBase
from pypfopt import DiscreteAllocation
from amplpyfinance import EfficientFrontierWithAMPL, DiscreteAllocationWithAMPL


class TestDiscreteAllocationWithAMPL(TestBase.TestBase):
    """Test EfficientFrontierWithAMPL."""

    def test_lp_portfolio(self):
        ef = EfficientFrontierWithAMPL(None, self.S, weight_bounds=(None, None))
        ef.min_volatility()
        weights = ef.clean_weights()
        latest_prices = self.prices.iloc[-1]  # prices as of the day you are allocating

        da1 = DiscreteAllocation(
            weights, latest_prices, total_portfolio_value=20000, short_ratio=0.3
        )
        alloc1, leftover1 = da1.lp_portfolio(verbose=True)
        print(f"Discrete allocation performed with ${leftover1:.2f} leftover")
        print(alloc1)

        da2 = DiscreteAllocationWithAMPL(
            weights, latest_prices, total_portfolio_value=20000, short_ratio=0.3
        )
        alloc2, leftover2 = da2.lp_portfolio(solver="cbc", verbose=True)
        print(f"Discrete allocation performed with ${leftover2:.2f} leftover")
        print(alloc2)

        self.assertAlmostEqual(leftover1, leftover2)
        self.assertEqual(alloc1, alloc2)

    def test_reinvest(self):
        ef = EfficientFrontierWithAMPL(None, self.S, weight_bounds=(None, None))
        ef.min_volatility()
        weights = ef.clean_weights()
        latest_prices = self.prices.iloc[-1]  # prices as of the day you are allocating

        da1 = DiscreteAllocation(
            weights, latest_prices, total_portfolio_value=20000, short_ratio=0.3
        )
        alloc1, leftover1 = da1.lp_portfolio(reinvest=True, verbose=True)
        print(f"Discrete allocation performed with ${leftover1:.2f} leftover")
        print(alloc1)

        da2 = DiscreteAllocationWithAMPL(
            weights, latest_prices, total_portfolio_value=20000, short_ratio=0.3
        )
        alloc2, leftover2 = da2.lp_portfolio(solver="cbc", reinvest=True, verbose=True)
        print(f"Discrete allocation performed with ${leftover2:.2f} leftover")
        print(alloc2)

        self.assertAlmostEqual(leftover1, leftover2)
        self.assertEqual(alloc1, alloc2)


if __name__ == "__main__":
    unittest.main()
