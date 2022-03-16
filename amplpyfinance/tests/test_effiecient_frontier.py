#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from . import TestBase
from .TestBase import sector_mapper, sector_lower, sector_upper
import numpy as np
from pypfopt import EfficientFrontier
from amplpyfinance import EfficientFrontierWithAMPL

EPS = 1e-4


class TestEfficientFrontierWithAMPL(TestBase.TestBase):
    """Test EfficientFrontierWithAMPL."""

    def test_min_volatility(self):
        ef1 = EfficientFrontier(None, self.S, weight_bounds=(None, None))
        ef1.min_volatility()

        ef2 = EfficientFrontierWithAMPL(
            None, self.S, weight_bounds=(None, None), solver="gurobi"
        )
        ef2.min_volatility()

        _, sigma1, _ = ef1.portfolio_performance(verbose=True)
        _, sigma2, _ = ef2.portfolio_performance(verbose=True)
        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertEqualWeights(ef1.clean_weights(), ef2.clean_weights(), EPS)

    def test_max_sharpe(self):
        ef1 = EfficientFrontier(self.mu, self.S)
        ef1.max_sharpe()

        ef2 = EfficientFrontierWithAMPL(self.mu, self.S)
        ef2.max_sharpe()

        mu1, sigma1, sharpe1 = ef1.portfolio_performance(verbose=True)
        mu2, sigma2, sharpe2 = ef2.portfolio_performance(verbose=True)
        self.assertLessEqual(abs(mu1 - mu2), EPS)
        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertLessEqual(abs(sharpe1 - sharpe2), EPS)
        self.assertEqualWeights(ef1.clean_weights(), ef2.clean_weights(), EPS)

    def test_max_sharpe_sectors(self):
        ef1 = EfficientFrontier(self.mu, self.S)
        ef1.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
        ef1.max_sharpe()

        ef2 = EfficientFrontierWithAMPL(self.mu, self.S)
        ef2.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
        ef2.max_sharpe()

        mu1, sigma1, sharpe1 = ef1.portfolio_performance(verbose=True)
        mu2, sigma2, sharpe2 = ef2.portfolio_performance(verbose=True)
        self.assertLessEqual(abs(mu1 - mu2), EPS)
        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertLessEqual(abs(sharpe1 - sharpe2), EPS)
        self.assertEqualWeights(ef1.clean_weights(), ef2.clean_weights(), EPS)

    def test_max_sharpe_sectors_and_bounds(self):
        ef1 = EfficientFrontier(self.mu, self.S)
        ef1.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
        ef1.add_constraint(lambda w: w[ef1.tickers.index("AMZN")] == 0.10)
        ef1.add_constraint(lambda w: w[ef1.tickers.index("TSLA")] <= 0.05)
        ef1.add_constraint(lambda w: w[ef1.tickers.index("ACN")] >= 0.05)
        ef1.max_sharpe()

        ef2 = EfficientFrontierWithAMPL(self.mu, self.S)
        ef2.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
        ef2.ampl.param["ticker_lower"] = {"AMZN": 0.10, "MSFT": 0.05}
        ef2.ampl.param["ticker_upper"] = {"TSLA": 0.05}
        ef2.max_sharpe()

        mu1, sigma1, sharpe1 = ef1.portfolio_performance(verbose=True)
        mu2, sigma2, sharpe2 = ef2.portfolio_performance(verbose=True)
        self.assertLessEqual(abs(mu1 - mu2), EPS)
        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertLessEqual(abs(sharpe1 - sharpe2), EPS)
        self.assertEqualWeights(ef1.clean_weights(), ef2.clean_weights(), EPS)

    def test_efficient_risk(self):
        ef1 = EfficientFrontier(self.mu, self.S)
        ef1.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
        ef1.efficient_risk(target_volatility=0.15)

        ef2 = EfficientFrontierWithAMPL(self.mu, self.S)
        ef2.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
        ef2.efficient_risk(target_volatility=0.15)

        mu1, sigma1, sharpe1 = ef1.portfolio_performance(verbose=True)
        mu2, sigma2, sharpe2 = ef2.portfolio_performance(verbose=True)
        self.assertLessEqual(abs(mu1 - mu2), EPS)
        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertLessEqual(abs(sharpe1 - sharpe2), EPS)
        # self.assertEqualWeights(ef1.clean_weights(), ef2.clean_weights(), EPS)

    def test_efficient_risk_l2reg(self):
        from pypfopt import objective_functions

        ef1 = EfficientFrontier(self.mu, self.S)
        ef1.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
        ef1.add_objective(objective_functions.L2_reg, gamma=0.1)
        ef1.efficient_risk(0.15)

        ef2 = EfficientFrontierWithAMPL(self.mu, self.S)
        ef2.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
        ef2.ampl.param["gamma"] = 0.1
        ef2.efficient_risk(target_volatility=0.15)

        mu1, sigma1, sharpe1 = ef1.portfolio_performance(verbose=True)
        mu2, sigma2, sharpe2 = ef2.portfolio_performance(verbose=True)
        self.assertLessEqual(abs(mu1 - mu2), EPS)
        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertLessEqual(abs(sharpe1 - sharpe2), EPS)
        # self.assertEqualWeights(ef1.clean_weights(), ef2.clean_weights(), EPS)

    def test_efficient_return(self):
        ef1 = EfficientFrontier(self.mu, self.S, weight_bounds=(None, None))
        ef1.efficient_return(target_return=0.07, market_neutral=True)

        ef2 = EfficientFrontierWithAMPL(self.mu, self.S, weight_bounds=(None, None))
        ef2.efficient_return(target_return=0.07, market_neutral=True)

        mu1, sigma1, sharpe1 = ef1.portfolio_performance(verbose=True)
        mu2, sigma2, sharpe2 = ef2.portfolio_performance(verbose=True)
        self.assertLessEqual(abs(mu1 - mu2), EPS)
        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertLessEqual(abs(sharpe1 - sharpe2), EPS)
        self.assertEqualWeights(ef1.clean_weights(), ef2.clean_weights(), EPS)

    def test_efficient_return_l2reg(self):
        from pypfopt import objective_functions

        ef1 = EfficientFrontier(self.mu, self.S, weight_bounds=(None, None))
        ef1.add_objective(objective_functions.L2_reg, gamma=0.1)
        ef1.efficient_return(target_return=0.07, market_neutral=True)

        ef2 = EfficientFrontierWithAMPL(self.mu, self.S, weight_bounds=(None, None))
        ef2.ampl.param["gamma"] = 0.1
        ef2.efficient_return(target_return=0.07, market_neutral=True)

        mu1, sigma1, sharpe1 = ef1.portfolio_performance(verbose=True)
        mu2, sigma2, sharpe2 = ef2.portfolio_performance(verbose=True)
        self.assertLessEqual(abs(mu1 - mu2), EPS)
        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertLessEqual(abs(sharpe1 - sharpe2), EPS)
        self.assertEqualWeights(ef1.clean_weights(), ef2.clean_weights(), EPS)

    def test_max_quadratic_utility_nobounds(self):
        ef1 = EfficientFrontier(self.mu, self.S, weight_bounds=(None, None))
        ef1.max_quadratic_utility(risk_aversion=2, market_neutral=False)

        ef2 = EfficientFrontierWithAMPL(self.mu, self.S, weight_bounds=(None, None))
        ef2.max_quadratic_utility(risk_aversion=2, market_neutral=False)

        mu1, sigma1, sharpe1 = ef1.portfolio_performance(verbose=True)
        mu2, sigma2, sharpe2 = ef2.portfolio_performance(verbose=True)
        self.assertLessEqual(abs(mu1 - mu2), EPS)
        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertLessEqual(abs(sharpe1 - sharpe2), EPS)
        self.assertEqualWeights(ef1.clean_weights(), ef2.clean_weights(), EPS)

    def test_max_quadratic_utility_withbounds(self):
        ef1 = EfficientFrontier(self.mu, self.S)
        ef1.max_quadratic_utility(risk_aversion=2, market_neutral=False)

        ef2 = EfficientFrontierWithAMPL(self.mu, self.S)
        ef2.max_quadratic_utility(risk_aversion=2, market_neutral=False)

        mu1, sigma1, sharpe1 = ef1.portfolio_performance(verbose=True)
        mu2, sigma2, sharpe2 = ef2.portfolio_performance(verbose=True)
        self.assertLessEqual(abs(mu1 - mu2), EPS)
        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertLessEqual(abs(sharpe1 - sharpe2), EPS)
        self.assertEqualWeights(ef1.clean_weights(), ef2.clean_weights(), EPS)

    def test_pypfopt_nonconvex_objective(self):
        ef1 = EfficientFrontier(self.mu, self.S, weight_bounds=(None, None))

        def utility_obj(weights, mu, cov_matrix, risk_aversion=1):
            return -weights.dot(mu) + 0.5 * risk_aversion * np.dot(
                weights.T, np.dot(cov_matrix, weights)
            )

        ef1.nonconvex_objective(
            utility_obj, objective_args=(ef1.expected_returns, ef1.cov_matrix, 2)
        )

        ef2 = EfficientFrontier(self.mu, self.S, weight_bounds=(None, None))
        ef2.max_quadratic_utility(risk_aversion=2, market_neutral=False)

        mu1, sigma1, sharpe1 = ef1.portfolio_performance(verbose=True)
        mu2, sigma2, sharpe2 = ef2.portfolio_performance(verbose=True)
        # The precision is lower with EfficientFrontier.nonconvex_objective
        self.assertLessEqual(abs(mu1 - mu2), 1e-3)
        self.assertLessEqual(abs(sigma1 - sigma2), 1e-3)
        self.assertLessEqual(abs(sharpe1 - sharpe2), 1e-3)
        # self.assertEqualWeights(ef1.clean_weights(), ef2.clean_weights(), EPS)

    def test_nonconvex_objective(self):
        ef1 = EfficientFrontierWithAMPL(self.mu, self.S)
        ef1.max_quadratic_utility(risk_aversion=2)

        ef2 = EfficientFrontierWithAMPL(self.mu, self.S)
        ef2.ampl.param["risk_aversion"] = 2
        ef2.ampl.param["market_neutral"] = 0
        ef2.ampl.eval(
            r"""
            maximize nonconvex_objective:
                sum {i in A} mu[i] * w[i]
                - 0.5 * risk_aversion * sum {i in A, j in A} w[i] * S[i, j] * w[j];
            solve nonconvex_objective;
            """
        )
        ef2.save_portfolio()

        mu1, sigma1, sharpe1 = ef1.portfolio_performance(verbose=True)
        mu2, sigma2, sharpe2 = ef2.portfolio_performance(verbose=True)
        self.assertLessEqual(abs(mu1 - mu2), EPS)
        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertLessEqual(abs(sharpe1 - sharpe2), EPS)
        self.assertEqualWeights(ef1.clean_weights(), ef2.clean_weights(), EPS)


if __name__ == "__main__":
    unittest.main()
