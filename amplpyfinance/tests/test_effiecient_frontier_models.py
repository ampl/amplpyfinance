#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from . import TestBase
from .TestBase import sector_mapper, sector_lower, sector_upper
import pandas as pd
from amplpy import AMPL
from pypfopt import EfficientFrontier
from amplpyfinance import EfficientFrontierWithAMPL

EPS = 1e-5


def save_portfolio(ampl, risk_free_rate=0.02):
    sigma = ampl.get_value("sqrt(sum {i in A, j in A} w[i] * S[i, j] * w[j])")
    try:
        mu = ampl.get_value("sum {i in A} mu[i] * w[i]")
        sharpe = (mu - risk_free_rate) / sigma
    except:
        mu, sharpe = None, None
    weights = [v for _, v in ampl.var["w"].get_values().to_list()]
    return weights, mu, sigma, sharpe


class TestEfficientFrontierModels(TestBase.TestBase):
    """Test TestEfficientFrontierModels."""

    def test_min_volatility(self):
        ef = EfficientFrontierWithAMPL(
            None, self.S, weight_bounds=(None, None), solver="gurobi"
        )
        ef.min_volatility()
        _, sigma1, _ = ef.portfolio_performance(verbose=True)

        ampl = AMPL()
        ampl.eval(
            r"""
            set A ordered;
            param S{A, A};
            var w{A} >= -1 <= 1;
            minimize portfolio_variance:
                sum {i in A, j in A} w[i] * S[i, j] * w[j];
            s.t. portfolio_weights:
                sum {i in A} w[i] = 1;
            """
        )
        ampl.set["A"] = ef.tickers
        ampl.param["S"] = pd.DataFrame(
            ef.cov_matrix, index=ef.tickers, columns=ef.tickers
        ).unstack(level=0)
        ampl.option["solver"] = "gurobi"
        ampl.solve()
        weights2, _, sigma2, _ = save_portfolio(ampl)

        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertEqualWeights(ef.clean_weights(), weights2, EPS)

    def test_max_sharpe(self):
        ef = EfficientFrontierWithAMPL(self.mu, self.S, solver="gurobi")
        ef.max_sharpe()
        mu1, sigma1, sharpe1 = ef.portfolio_performance(verbose=True)

        ampl = AMPL()
        ampl.eval(
            r"""
            set A ordered;
            param S{A, A};
            param mu{A} default 0;
            param risk_free_rate default 0.02;
            var k >= 0;
            var z{i in A} >= 0;  # scaled weights
            var w{i in A} = z[i] / k;
            minimize portfolio_sharpe:
                sum {i in A, j in A} z[i] * S[i, j] * z[j];
            s.t. ms_muz:
                sum {i in A} (mu[i] - risk_free_rate) * z[i] = 1;
            s.t. portfolio_weights:
                sum {i in A}  z[i] = k;
            """
        )
        ampl.set["A"] = ef.tickers
        ampl.param["S"] = pd.DataFrame(
            ef.cov_matrix, index=ef.tickers, columns=ef.tickers
        ).unstack(level=0)
        ampl.param["mu"] = ef.expected_returns
        ampl.option["solver"] = "gurobi"
        ampl.solve()
        weights2, mu2, sigma2, sharpe2 = save_portfolio(ampl)

        self.assertLessEqual(abs(mu1 - mu2), EPS)
        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertLessEqual(abs(sharpe1 - sharpe2), EPS)
        self.assertEqualWeights(ef.clean_weights(), weights2, EPS)

    def test_efficient_risk(self):
        ef = EfficientFrontierWithAMPL(self.mu, self.S, solver="gurobi")
        ef.efficient_risk(target_volatility=0.15)
        mu1, sigma1, sharpe1 = ef.portfolio_performance(verbose=True)

        ampl = AMPL()
        ampl.eval(
            r"""
            set A ordered;
            param S{A, A};
            param mu{A} default 0;
            param target_variance;
            var w{A} >= 0 <= 1;
            maximize portfolio_return:
                sum {i in A} mu[i] * w[i];
            s.t. portfolio_variance:
                sum {i in A, j in A} w[i] * S[i, j] * w[j] <= target_variance;
            s.t. portfolio_weights:
                sum {i in A} w[i] = 1;
            """
        )
        ampl.set["A"] = ef.tickers
        ampl.param["S"] = pd.DataFrame(
            ef.cov_matrix, index=ef.tickers, columns=ef.tickers
        ).unstack(level=0)
        ampl.param["mu"] = ef.expected_returns
        ampl.param["target_variance"] = 0.15**2
        ampl.option["solver"] = "gurobi"
        ampl.solve()
        weights2, mu2, sigma2, sharpe2 = save_portfolio(ampl)

        self.assertLessEqual(abs(mu1 - mu2), EPS)
        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertLessEqual(abs(sharpe1 - sharpe2), EPS)
        self.assertEqualWeights(ef.clean_weights(), weights2, EPS)

    def test_efficient_risk_l2reg(self):
        ef = EfficientFrontierWithAMPL(
            self.mu, self.S, solver="gurobi", solver_options="outlev=0"
        )
        ef.ampl.param["gamma"] = 0.2
        ef.efficient_risk(target_volatility=0.15)
        mu1, sigma1, sharpe1 = ef.portfolio_performance(verbose=True)

        ampl = AMPL()
        ampl.eval(
            r"""
            set A ordered;
            param S{A, A};
            param mu{A} default 0;
            param target_variance;
            var w{A} >= 0 <= 1;
            param gamma default 0;
            var l2_reg = gamma * sum{i in A} w[i] * w[i];
            maximize portfolio_return:
                -l2_reg + sum {i in A} mu[i] * w[i];
            s.t. portfolio_variance:
                sum {i in A, j in A} w[i] * S[i, j] * w[j] <= target_variance;
            s.t. portfolio_weights:
                sum {i in A} w[i] = 1;
            """
        )
        ampl.set["A"] = ef.tickers
        ampl.param["S"] = pd.DataFrame(
            ef.cov_matrix, index=ef.tickers, columns=ef.tickers
        ).unstack(level=0)
        ampl.param["mu"] = ef.expected_returns
        ampl.param["target_variance"] = 0.15**2
        ampl.param["gamma"] = 0.2
        ampl.option["solver"] = "gurobi"
        ampl.option["gurobi_options"] = "outlev=0"
        ampl.solve()
        weights2, mu2, sigma2, sharpe2 = save_portfolio(ampl)

        self.assertLessEqual(abs(mu1 - mu2), EPS)
        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertLessEqual(abs(sharpe1 - sharpe2), EPS)
        self.assertEqualWeights(ef.clean_weights(), weights2, EPS)

    def test_efficient_return(self):
        ef = EfficientFrontierWithAMPL(
            self.mu, self.S, weight_bounds=(None, None), solver="gurobi"
        )
        ef.efficient_return(target_return=0.07, market_neutral=True)
        mu1, sigma1, sharpe1 = ef.portfolio_performance(verbose=True)

        ampl = AMPL()
        ampl.eval(
            r"""
            set A ordered;
            param S{A, A};
            param mu{A} default 0;
            param target_return;
            var w{A} >= -1 <= 1;
            minimize portfolio_variance:
                sum {i in A, j in A} w[i] * S[i, j] * w[j];
            s.t. portfolio__return:
                sum {i in A} mu[i] * w[i] >= target_return;
            s.t. portfolio_weights:
                sum {i in A} w[i] = 0;
            """
        )
        ampl.set["A"] = ef.tickers
        ampl.param["S"] = pd.DataFrame(
            ef.cov_matrix, index=ef.tickers, columns=ef.tickers
        ).unstack(level=0)
        ampl.param["mu"] = ef.expected_returns
        ampl.param["target_return"] = 0.07
        ampl.option["solver"] = "gurobi"
        ampl.solve()
        weights2, mu2, sigma2, sharpe2 = save_portfolio(ampl)

        self.assertLessEqual(abs(mu1 - mu2), EPS)
        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertLessEqual(abs(sharpe1 - sharpe2), EPS)
        self.assertEqualWeights(ef.clean_weights(), weights2, EPS)

    def test_efficient_return_l2reg(self):
        ef = EfficientFrontierWithAMPL(
            self.mu, self.S, weight_bounds=(None, None), solver="gurobi"
        )
        ef.ampl.param["gamma"] = 0.2
        ef.efficient_return(target_return=0.07, market_neutral=True)
        mu1, sigma1, sharpe1 = ef.portfolio_performance(verbose=True)

        ampl = AMPL()
        ampl.eval(
            r"""
            set A ordered;
            param S{A, A};
            param mu{A} default 0;
            param target_return;
            var w{A} >= -1 <= 1;
            param gamma default 0;
            var l2_reg = gamma * sum{i in A} w[i] * w[i];
            minimize portfolio_variance:
                l2_reg + sum {i in A, j in A} w[i] * S[i, j] * w[j];
            s.t. portfolio_return:
                sum {i in A} mu[i] * w[i] >= target_return;
            s.t. portfolio_weights:
                sum {i in A} w[i] = 0;
            """
        )
        ampl.set["A"] = ef.tickers
        ampl.param["S"] = pd.DataFrame(
            ef.cov_matrix, index=ef.tickers, columns=ef.tickers
        ).unstack(level=0)
        ampl.param["mu"] = ef.expected_returns
        ampl.param["gamma"] = 0.2
        ampl.param["target_return"] = 0.07
        ampl.option["solver"] = "gurobi"
        ampl.solve()
        weights2, mu2, sigma2, sharpe2 = save_portfolio(ampl)

        self.assertLessEqual(abs(mu1 - mu2), EPS)
        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertLessEqual(abs(sharpe1 - sharpe2), EPS)
        self.assertEqualWeights(ef.clean_weights(), weights2, EPS)

    def test_max_quadratic_utility(self):
        ef = EfficientFrontierWithAMPL(
            self.mu, self.S, weight_bounds=(None, None), solver="gurobi"
        )
        ef.max_quadratic_utility(risk_aversion=2, market_neutral=False)
        mu1, sigma1, sharpe1 = ef.portfolio_performance(verbose=True)

        ampl = AMPL()
        ampl.eval(
            r"""
            set A ordered;
            param S{A, A};
            param mu{A} default 0;
            param risk_aversion default 1;
            var w{A} >= -1 <= 1;
            maximize quadratic_utility:
                sum {i in A} mu[i] * w[i] 
                - 0.5 * risk_aversion * sum {i in A, j in A} w[i] * S[i, j] * w[j];
            s.t. portfolio_weights:
                sum {i in A} w[i] = 1;
            """
        )
        ampl.set["A"] = ef.tickers
        ampl.param["S"] = pd.DataFrame(
            ef.cov_matrix, index=ef.tickers, columns=ef.tickers
        ).unstack(level=0)
        ampl.param["mu"] = ef.expected_returns
        ampl.param["risk_aversion"] = 2
        ampl.option["solver"] = "gurobi"
        ampl.solve()
        weights2, mu2, sigma2, sharpe2 = save_portfolio(ampl)

        self.assertLessEqual(abs(mu1 - mu2), EPS)
        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertLessEqual(abs(sharpe1 - sharpe2), EPS)
        self.assertEqualWeights(ef.clean_weights(), weights2, EPS)

    def test_sector_constraints(self):
        ef = EfficientFrontierWithAMPL(self.mu, self.S)
        ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
        ef.efficient_risk(target_volatility=0.15)
        mu1, sigma1, sharpe1 = ef.portfolio_performance(verbose=True)

        ampl = AMPL()
        ampl.eval(
            r"""
            set A ordered;
            param S{A, A};
            param mu{A} default 0;
            param target_variance;
            var w{A} >= 0 <= 1;
            maximize portfolio_return:
                sum {i in A} mu[i] * w[i];
            s.t. portfolio_variance:
                sum {i in A, j in A} w[i] * S[i, j] * w[j] <= target_variance;
            s.t. portfolio_weights:
                sum {i in A} w[i] = 1;
            set SECTORS default {};
            set SECTOR_MEMBERS{SECTORS};
            param sector_lower{SECTORS} default -Infinity;
            param sector_upper{SECTORS} default Infinity;
            s.t. sector_constraints_lower{s in SECTORS: sector_lower[s] != -Infinity}:
                sum {i in SECTOR_MEMBERS[s]} w[i] >= sector_lower[s];
            s.t. sector_constraints_upper{s in SECTORS: sector_upper[s] != Infinity}:
                sum {i in SECTOR_MEMBERS[s]} w[i] <= sector_upper[s];
            """
        )
        ampl.set["A"] = ef.tickers
        ampl.param["S"] = pd.DataFrame(
            ef.cov_matrix, index=ef.tickers, columns=ef.tickers
        ).unstack(level=0)
        ampl.param["mu"] = ef.expected_returns
        ampl.param["target_variance"] = 0.15**2
        sectors = set(sector_mapper.values())
        ampl.set["SECTORS"] = sectors
        for sector in sectors:
            ampl.set["SECTOR_MEMBERS"][sector] = [
                ticker for ticker, s in sector_mapper.items() if s == sector
            ]
        ampl.param["sector_lower"] = sector_lower
        ampl.param["sector_upper"] = sector_upper
        ampl.option["solver"] = "gurobi"
        ampl.solve()
        weights2, mu2, sigma2, sharpe2 = save_portfolio(ampl)

        self.assertLessEqual(abs(mu1 - mu2), EPS)
        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertLessEqual(abs(sharpe1 - sharpe2), EPS)
        self.assertEqualWeights(ef.clean_weights(), weights2, 1e-4)

    def test_card_constraints(self):
        ef = EfficientFrontierWithAMPL(self.mu, self.S)
        ef.ampl.param["card_ub"] = 3
        ef.efficient_risk(target_volatility=0.15)
        mu1, sigma1, sharpe1 = ef.portfolio_performance(verbose=True)

        ampl = AMPL()
        ampl.eval(
            r"""
            set A ordered;
            param S{A, A};
            param mu{A} default 0;
            param target_variance;
            var w{A} >= 0 <= 1;
            var y{A} binary;
            maximize portfolio_return:
                sum {i in A} mu[i] * w[i];
            s.t. portfolio_variance:
                sum {i in A, j in A} w[i] * S[i, j] * w[j] <= target_variance;
            s.t. portfolio_weights:
                sum {i in A} w[i] = 1;
            param lb default 0;
            param ub default 1;
            s.t. w_lower{i in A}:
                lb * y[i] <= w[i];
            s.t. w_upper{i in A}:
                w[i] <= ub * y[i];
            param card_ub default Infinity;
            s.t. card_limit:
                sum {i in A} y[i] <= card_ub;
            """
        )
        ampl.set["A"] = ef.tickers
        ampl.param["S"] = pd.DataFrame(
            ef.cov_matrix, index=ef.tickers, columns=ef.tickers
        ).unstack(level=0)
        ampl.param["mu"] = ef.expected_returns
        ampl.param["card_ub"] = 3
        ampl.param["target_variance"] = 0.15**2
        ampl.option["solver"] = "gurobi"
        ampl.solve()
        weights2, mu2, sigma2, sharpe2 = save_portfolio(ampl)

        self.assertLessEqual(len([w for w in weights2 if w > EPS]), 3)
        self.assertLessEqual(abs(mu1 - mu2), EPS)
        self.assertLessEqual(abs(sigma1 - sigma2), EPS)
        self.assertLessEqual(abs(sharpe1 - sharpe2), EPS)
        self.assertEqualWeights(ef.clean_weights(), weights2, EPS)


if __name__ == "__main__":
    unittest.main()
