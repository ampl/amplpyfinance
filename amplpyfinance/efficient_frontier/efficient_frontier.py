import warnings
import numpy as np
import pandas as pd
from pypfopt.base_optimizer import BaseOptimizer
from amplpy import AMPL


class EfficientFrontierWithAMPL(BaseOptimizer):
    def __init__(
        self,
        expected_returns,
        cov_matrix,
        weight_bounds=(0, 1),
        solver="gurobi",
        verbose=False,
        solver_options="",
    ):
        # Inputs
        self.cov_matrix = EfficientFrontierWithAMPL._validate_cov_matrix(cov_matrix)
        self.expected_returns = EfficientFrontierWithAMPL._validate_expected_returns(
            expected_returns
        )
        self._max_return_value = None

        if self.expected_returns is None:
            num_assets = len(cov_matrix)
        else:
            num_assets = len(expected_returns)

        # Labels
        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        elif isinstance(cov_matrix, pd.DataFrame):
            tickers = list(cov_matrix.columns)
        else:  # use integer labels
            tickers = list(range(num_assets))

        if expected_returns is not None and cov_matrix is not None:
            if cov_matrix.shape != (num_assets, num_assets):
                raise ValueError("Covariance matrix does not match expected returns")

        self.solver = solver
        self.verbose = verbose
        self.solver_options = solver_options
        self.weight_bounds = weight_bounds

        self.ampl = AMPL()
        ampl = self.ampl
        model = r"""
        set A ordered;         # assets
        param S{A, A};         # cov matrix
        param mu{A} default 0; # expected returns

        param lb default -1;
        param ub default 1;
        param market_neutral default 0;
        param w_lb := if market_neutral then -1 else lb;
        param w_ub := if market_neutral then 1 else ub;
        var w{A} >= w_lb <= w_ub;  # weights

        param risk_free_rate default 0.02;
        param risk_aversion default 1;
        param target_variance;
        param target_return;

        param total_weight := if market_neutral == 1 then 0 else 1;
        s.t. portfolio_weights:
            sum {i in A} w[i] = total_weight;

        param gamma default 0;
        var l2_reg = gamma * sum{i in A} w[i] * w[i];

        var y{A} binary;
        param ticker_lower{A} default -Infinity;
        param ticker_upper{A} default Infinity;
        s.t. w_lower{i in A}:
            max(lb, ticker_lower[i]) * y[i] <= w[i];
        s.t. w_upper{i in A}:
            w[i] <= min(ub, ticker_upper[i]) * y[i];
        
        param card_ub default Infinity;
        s.t. card_limit:
            sum {i in A} y[i] <= card_ub;

        set SECTORS default {};
        set SECTOR_MEMBERS{SECTORS};
        param sector_lower{SECTORS} default -Infinity;
        param sector_upper{SECTORS} default Infinity;
        s.t. sector_constraints_lower{s in SECTORS: sector_lower[s] != -Infinity}:
            sum {i in SECTOR_MEMBERS[s]} w[i] >= sector_lower[s];
        s.t. sector_constraints_upper{s in SECTORS: sector_upper[s] != Infinity}:
            sum {i in SECTOR_MEMBERS[s]} w[i] <= sector_upper[s];

        problem min_volatility: 
                w, w_lower, w_upper, portfolio_weights, y, card_limit, 
                sector_constraints_lower, sector_constraints_upper;
            minimize mv_portfolio_variance:
                sum {i in A, j in A} w[i] * S[i, j] * w[j];

        problem efficient_risk:
                l2_reg, w, w_lower, w_upper, portfolio_weights, y, card_limit,
                sector_constraints_lower, sector_constraints_upper;
            maximize eri_portfolio_return:
                -l2_reg + sum {i in A} mu[i] * w[i];
            s.t. eri_portfolio_variance:
                sum {i in A, j in A} w[i] * S[i, j] * w[j] <= target_variance;

        problem efficient_return:
                l2_reg, w, w_lower, w_upper, portfolio_weights, y, card_limit,
                sector_constraints_lower, sector_constraints_upper;
            minimize ert_portfolio_variance:
                l2_reg + sum {i in A, j in A} w[i] * S[i, j] * w[j];
            s.t. ert_return:
                sum {i in A} mu[i] * w[i] >= target_return;

        problem _max_return:
                l2_reg, w, w_lower, w_upper, portfolio_weights, y, card_limit,
                sector_constraints_lower, sector_constraints_upper;
            # Auxiliar problem. This should not be used to optimize a portfolio
            maximize mr_max_return:
                sum {i in A} mu[i] * w[i];

        problem max_quadratic_utility:
                l2_reg, w, w_lower, w_upper, portfolio_weights, y, card_limit,
                sector_constraints_lower, sector_constraints_upper;
            maximize mqu_quadratic_utility:
                l2_reg + sum {i in A} mu[i] * w[i] 
                - 0.5 * risk_aversion * sum {i in A, j in A} w[i] * S[i, j] * w[j];

        problem max_sharpe: y, card_limit;
            var k >= 0;
            var z{i in A} >= 0;  # scaled weights
            minimize ms_objective:
                sum {i in A, j in A} z[i] * S[i, j] * z[j];
            s.t. ms_muz:
                sum {i in A} (mu[i] - risk_free_rate) * z[i] = 1;
            s.t. ms_portfolio_weights:
                sum {i in A}  z[i] = k;
            s.t. ms_ticker_lower{i in A: max(lb, ticker_lower[i]) != -Infinity}:
                z[i] >= max(lb, ticker_lower[i]) * k;
            s.t. ms_ticker_upper{i in A: min(ub, ticker_upper[i]) != Infinity}:
                z[i] <= min(ub, ticker_upper[i]) * k;
            s.t. ms_y{i in A: card_ub >= 1}:
                z[i] <= min(ub, ticker_upper[i]) * k * y[i];
            s.t. ms_sector_constraints_lower{s in SECTORS: sector_lower[s] != -Infinity}:
                sum {i in SECTOR_MEMBERS[s]} z[i] >= sector_lower[s] * k;
            s.t. ms_sector_constraints_upper{s in SECTORS: sector_upper[s] != Infinity}:
                sum {i in SECTOR_MEMBERS[s]} z[i] <= sector_upper[s] * k;
        
        problem Initial;
        """
        ampl.eval(model)

        ampl = self.ampl
        ampl.set["A"] = tickers
        ampl.param["S"] = pd.DataFrame(
            self.cov_matrix, index=tickers, columns=tickers
        ).unstack(level=0)
        if self.expected_returns is not None:
            ampl.param["mu"] = self.expected_returns
        lb, ub = self.weight_bounds
        if lb is not None:
            ampl.param["lb"] = lb
        if ub is not None:
            ampl.param["ub"] = ub

        super().__init__(
            len(tickers),
            tickers,
        )

    @staticmethod
    def _validate_expected_returns(expected_returns):
        if expected_returns is None:
            return None
        elif isinstance(expected_returns, pd.Series):
            return expected_returns.values
        elif isinstance(expected_returns, list):
            return np.array(expected_returns)
        elif isinstance(expected_returns, np.ndarray):
            return expected_returns.ravel()
        else:
            raise TypeError("expected_returns is not a series, list or array")

    @staticmethod
    def _validate_cov_matrix(cov_matrix):
        if cov_matrix is None:
            raise ValueError("cov_matrix must be provided")
        elif isinstance(cov_matrix, pd.DataFrame):
            return cov_matrix.values
        elif isinstance(cov_matrix, np.ndarray):
            return cov_matrix
        else:
            raise TypeError("cov_matrix is not a dataframe or array")

    def _solve(self, problem):
        ampl = self.ampl
        ampl.eval(f"problem {problem};")
        ampl.option["solver"] = self.solver
        if self.solver and self.solver_options:
            ampl.option[f"{self.solver}_options"] = self.solver_options
        if ampl.param["card_ub"].value() >= len(self.tickers):
            ampl.eval("fix {i in A} y[i] := 1;")
        ampl.solve()

    def _max_return(self):
        if self.expected_returns is None:
            raise ValueError("no expected returns provided")

        self._solve("_max_return")
        self.save_portfolio()
        return self.mu

    def max_quadratic_utility(self, risk_aversion=1, market_neutral=False):
        if risk_aversion <= 0:
            raise ValueError("risk aversion coefficient must be greater than zero")

        ampl = self.ampl
        ampl.param["risk_aversion"] = risk_aversion
        ampl.param["market_neutral"] = 1 if market_neutral else 0
        self._solve("max_quadratic_utility")
        self.save_portfolio()
        return self._make_output_weights(self.weights)

    def add_sector_constraints(self, sector_mapper, sector_lower, sector_upper):
        if np.any(self.weight_bounds[0] < 0):
            warnings.warn(
                "Sector constraints may not produce reasonable results if shorts are allowed."
            )

        ampl = self.ampl
        sectors = set(sector_mapper.values())
        ampl.set["SECTORS"] = sectors
        for sector in sectors:
            ampl.set["SECTOR_MEMBERS"][sector] = [
                ticker for ticker, s in sector_mapper.items() if s == sector
            ]
        ampl.param["sector_lower"] = sector_lower
        ampl.param["sector_upper"] = sector_upper

    def min_volatility(self):
        self._solve("min_volatility")
        self.save_portfolio()
        return self._make_output_weights(self.weights)

    def efficient_risk(self, target_volatility, market_neutral=False):
        if not isinstance(target_volatility, (float, int)) or target_volatility < 0:
            raise ValueError("target_volatility should be a positive float")

        global_min_volatility = np.sqrt(1 / np.sum(np.linalg.inv(self.cov_matrix)))

        if target_volatility < global_min_volatility:
            raise ValueError(
                "The minimum volatility is {:.3f}. Please use a higher target_volatility".format(
                    global_min_volatility
                )
            )

        ampl = self.ampl
        ampl.param["market_neutral"] = 1 if market_neutral else 0
        ampl.param["target_variance"] = target_volatility**2
        self._solve("efficient_risk")
        self.save_portfolio()
        return self._make_output_weights(self.weights)

    def efficient_return(self, target_return, market_neutral=False):
        if not isinstance(target_return, float) or target_return < 0:
            raise ValueError("target_return should be a positive float")
        if not self._max_return_value:
            self._max_return_value = self._max_return()
        if target_return > self._max_return_value:
            raise ValueError(
                "target_return must be lower than the maximum possible return"
            )

        ampl = self.ampl
        ampl.param["market_neutral"] = 1 if market_neutral else 0
        ampl.param["target_return"] = target_return
        self._solve("efficient_return")
        self.save_portfolio()
        return self._make_output_weights(self.weights)

    def max_sharpe(self, risk_free_rate=0.02):
        if not isinstance(risk_free_rate, (int, float)):
            raise ValueError("risk_free_rate should be numeric")
        ampl = self.ampl
        ampl.param["risk_free_rate"] = risk_free_rate
        self._solve("max_sharpe")
        ampl.eval("let {i in A} w[i] := z[i] / k;")
        self.save_portfolio()
        return self._make_output_weights(self.weights)

    def save_portfolio(self, risk_free_rate=None):
        ampl = self.ampl
        self.sigma = ampl.get_value("sqrt(sum {i in A, j in A} w[i] * S[i, j] * w[j])")
        self.mu = ampl.get_value("sum {i in A} mu[i] * w[i]")
        if risk_free_rate is None:
            risk_free_rate = ampl.param["risk_free_rate"].value()
        self.sharpe = (self.mu - risk_free_rate) / self.sigma
        self.weights = np.array([v for _, v in ampl.var["w"].get_values().to_list()])

    def portfolio_performance(self, verbose=False, risk_free_rate=0.02):
        if self.weights is None:
            raise ValueError("Weights is None")

        if self.expected_returns is not None:
            sharpe = (self.mu - risk_free_rate) / self.sigma
            if verbose:
                print("Expected annual return: {:.1f}%".format(100 * self.mu))
                print("Annual volatility: {:.1f}%".format(100 * self.sigma))
                print("Sharpe Ratio: {:.2f}".format(sharpe))
            return self.mu, self.sigma, sharpe
        else:
            if verbose:
                print("Annual volatility: {:.1f}%".format(100 * self.sigma))
            return None, self.sigma, None
