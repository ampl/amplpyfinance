import warnings
import numpy as np
import pandas as pd
import pypfopt
from amplpy import AMPL


class EfficientFrontier(pypfopt.base_optimizer.BaseOptimizer):
    """
    An EfficientFrontier object contains multiple
    optimization methods that can be called (corresponding to different objective
    functions) with various parameters.

    AMPL version of :class:`pypfopt.EfficientFrontier` with similar interface.
    This class is also available under the alias :class:`amplpyfinance.EfficientFrontierWithAMPL`
    in order to distinguish from :class:`pypfopt.EfficientFrontier` if used together.

    Instance variables:

    - Inputs:

        - ``n_assets`` - int
        - ``tickers`` - str list
        - ``bounds`` - float tuple OR (float tuple) list
        - ``cov_matrix`` - np.ndarray
        - ``expected_returns`` - np.ndarray
        - ``solver`` - str
        - ``solver_options`` - str

    - Output: ``weights`` - np.ndarray

    Public methods:

    - :func:`min_volatility()` optimizes for minimum volatility
    - :func:`max_sharpe()` optimizes for maximal Sharpe ratio (a.k.a the tangency portfolio)
    - :func:`max_quadratic_utility()` maximizes the quadratic utility, given some risk aversion.
    - :func:`efficient_risk()` maximizes return for a given target risk
    - :func:`efficient_return()` minimizes risk for a given target returns

    - :func:`portfolio_performance()` calculates the expected return, volatility and Sharpe ratio for
      the optimized portfolio.
    - :func:`clean_weights()` rounds the weights and clips near-zeros.
    - :func:`save_weights_to_file()` saves the weights to csv, json, or txt.
    """

    def __init__(
        self,
        expected_returns,
        cov_matrix,
        weight_bounds=(0, 1),
        tickers=None,
        solver="gurobi",
        solver_options="",
        verbose=False,
    ):
        """
        Corresponding AMPL code:

        .. code-block:: ampl

            set A ordered;
            param S{A, A};
            param mu{A} default 0;
            param lb default 0;
            param ub default 1;
            var w{A} >= lb <= ub;

        .. code-block:: python

            ampl.set["A"] = tickers
            ampl.param["S"] = pd.DataFrame(
                cov_matrix, index=tickers, columns=tickers
            ).unstack(level=0)
            ampl.param["mu"] = expected_returns
            ampl.param["lb"] = weight_bounds[0]
            ampl.param["ub"] = weight_bounds[1]

        :param expected_returns: expected returns for each asset. Can be None if
                                optimizing for volatility only (but not recommended).
        :type expected_returns: pd.Series, list, np.ndarray
        :param cov_matrix: covariance of returns for each asset. This **must** be
                           positive semidefinite, otherwise optimization will fail.
        :type cov_matrix: pd.DataFrame or np.array
        :param weight_bounds: minimum and maximum weight of each asset OR single min/max pair
                              if all identical, defaults to (0, 1). Must be changed to (-1, 1)
                              for portfolios with shorting.
        :type weight_bounds: tuple OR tuple list, optional
        :param tickers: asset labels.
        :type tickers: str list, optional
        :param solver: name of the AMPL solver to use.
        :type solver: str
        :param solver_options: options for the given solver
        :type solver_options: str
        :param verbose: whether performance and debugging info should be printed, defaults to False
        :type verbose: bool, optional
        :raises TypeError: if ``expected_returns`` is not a series, list or array
        :raises TypeError: if ``cov_matrix`` is not a dataframe or array
        """
        # Inputs
        self.cov_matrix = EfficientFrontier._validate_cov_matrix(cov_matrix)
        self.expected_returns = EfficientFrontier._validate_expected_returns(
            expected_returns
        )
        self._max_return_value = None

        if self.expected_returns is None:
            num_assets = len(cov_matrix)
        else:
            num_assets = len(expected_returns)

        # Labels
        if not tickers:
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
        self._set_solver()
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

    def _set_solver(self):
        self.ampl.option["solver"] = self.solver
        if self.solver and self.solver_options:
            self.ampl.option[f"{self.solver}_options"] = self.solver_options

    def _solve(self, problem):
        ampl = self.ampl
        ampl.eval(f"problem {problem};")
        self._set_solver()
        if ampl.param["card_ub"].value() >= len(self.tickers):
            ampl.eval("fix {i in A} y[i] := 1;")
        ampl.solve()
        if ampl.get_value("solve_result") != "solved":
            raise pypfopt.exceptions.OptimizationError(
                "Failed to solve. Please check the solver log"
            )

    def _max_return(self):
        """
        Helper method to maximize return. This should not be used to optimize a portfolio.

        AMPL version of :func:`pypfopt.EfficientFrontier._max_return` with the same interface:

        :return: asset weights for the return-minimising portfolio
        :rtype: OrderedDict
        """
        if self.expected_returns is None:
            raise ValueError("no expected returns provided")

        self._solve("_max_return")
        self._save_portfolio()
        return self.mu

    def min_volatility(self):
        """
        Minimize volatility.

        Corresponding AMPL code:

        .. code-block:: ampl

            set A ordered;
            param S{A, A};

            param lb default 0;
            param ub default 1;
            var w{A} >= lb <= ub;

            minimize portfolio_variance:
                sum {i in A, j in A} w[i] * S[i, j] * w[j];
            s.t. portfolio_weights:
                sum {i in A} w[i] = 1;

        .. code-block:: python

            ampl.solve()

        AMPL version of :func:`pypfopt.EfficientFrontier.min_volatility` with the same interface:

        :return: asset weights for the volatility-minimising portfolio
        :rtype: OrderedDict
        """
        self._solve("min_volatility")
        self._save_portfolio()
        return self._make_output_weights(self.weights)

    def efficient_risk(self, target_volatility, market_neutral=False):
        """
        Maximize return for a target risk. The resulting portfolio will have a
        volatility less than the target (but not guaranteed to be equal).

        Corresponding AMPL code:

        .. code-block:: ampl

            param target_volatility;
            param market_neutral default 0;

            set A ordered;
            param S{A, A};
            param mu{A} default 0;

            param lb default 0;
            param ub default 1;
            var w{A} >= lb <= ub;

            maximize portfolio_return:
                sum {i in A} mu[i] * w[i];
            s.t. portfolio_variance:
                sum {i in A, j in A} w[i] * S[i, j] * w[j] <= target_volatility^2;
            s.t. portfolio_weights:
                sum {i in A} w[i] = if market_neutral then 0 else 1;

        .. code-block:: python

            ampl.param["target_volatility"] = target_volatility
            ampl.param["market_neutral"] = market_neutral
            ampl.solve()

        AMPL version of :func:`pypfopt.EfficientFrontier.efficient_risk` with the same interface:

        :param target_volatility: the desired maximum volatility of the resulting portfolio.
        :type target_volatility: float
        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),
                               defaults to False. Requires negative lower weight bound.
        :param market_neutral: bool, optional
        :raises ValueError: if ``target_volatility`` is not a positive float
        :raises ValueError: if no portfolio can be found with volatility equal to ``target_volatility``
        :raises ValueError: if ``risk_free_rate`` is non-numeric
        :return: asset weights for the efficient risk portfolio
        :rtype: OrderedDict
        """
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
        self._save_portfolio()
        return self._make_output_weights(self.weights)

    def efficient_return(self, target_return, market_neutral=False):
        """
        Calculate the 'Markowitz portfolio', minimising volatility for a given target return.

        Corresponding AMPL code:

        .. code-block:: ampl

            param target_return;
            param market_neutral default 0;

            set A ordered;
            param S{A, A};
            param mu{A} default 0;

            param lb default 0;
            param ub default 1;
            var w{A} >= lb <= ub;

            minimize portfolio_variance:
                sum {i in A, j in A} w[i] * S[i, j] * w[j];
            s.t. portfolio__return:
                sum {i in A} mu[i] * w[i] >= target_return;
            s.t. portfolio_weights:
                sum {i in A} w[i] = if market_neutral then 0 else 1;

        .. code-block:: python

            ampl.param["target_return"] = target_return
            ampl.param["market_neutral"] = market_neutral
            ampl.solve()

        AMPL version of :func:`pypfopt.EfficientFrontier.efficient_return` with the same interface:

        :param target_return: the desired return of the resulting portfolio.
        :type target_return: float
        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),
                               defaults to False. Requires negative lower weight bound.
        :type market_neutral: bool, optional
        :raises ValueError: if ``target_return`` is not a positive float
        :raises ValueError: if no portfolio can be found with return equal to ``target_return``
        :return: asset weights for the Markowitz portfolio
        :rtype: OrderedDict
        """

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
        self._save_portfolio()
        return self._make_output_weights(self.weights)

    def max_sharpe(self, risk_free_rate=0.02):
        """
        Maximize the Sharpe Ratio. The result is also referred to as the tangency portfolio,
        as it is the portfolio for which the capital market line is tangent to the efficient frontier.

        This is a convex optimization problem after making a certain variable substitution. See
        `Cornuejols and Tutuncu (2006) <http://web.math.ku.dk/~rolf/CT_FinOpt.pdf>`_ for more.

        Corresponding AMPL code:

        .. code-block:: ampl

            param risk_free_rate default 0.02;

            set A ordered;
            param S{A, A};
            param mu{A} default 0;

            var k >= 0;
            var z{i in A} >= 0;  # scaled weights
            var w{i in A} = z[i] / k;

            minimize portfolio_sharpe:
                sum {i in A, j in A} z[i] * S[i, j] * z[j];
            s.t. muz:
                sum {i in A} (mu[i] - risk_free_rate) * z[i] = 1;
            s.t. portfolio_weights:
                sum {i in A}  z[i] = k;

        .. code-block:: python

            ampl.param["risk_free_rate"] = risk_free_rate
            ampl.solve()

        AMPL version of :func:`pypfopt.EfficientFrontier.max_sharpe` with the same interface:

        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.
                               The period of the risk-free rate should correspond to the
                               frequency of expected returns.
        :type risk_free_rate: float, optional
        :raises ValueError: if ``risk_free_rate`` is non-numeric
        :return: asset weights for the Sharpe-maximising portfolio
        :rtype: OrderedDict
        """
        if not isinstance(risk_free_rate, (int, float)):
            raise ValueError("risk_free_rate should be numeric")
        ampl = self.ampl
        ampl.param["risk_free_rate"] = risk_free_rate
        self._solve("max_sharpe")
        ampl.eval("let {i in A} w[i] := z[i] / k;")
        self._save_portfolio()
        return self._make_output_weights(self.weights)

    def max_quadratic_utility(self, risk_aversion=1, market_neutral=False):
        r"""
        Maximize the given quadratic utility, i.e:

        .. math::

            \max_w w^T \mu - \frac \delta 2 w^T \Sigma w

        Corresponding AMPL code:

        .. code-block:: ampl

            param risk_aversion default 1;
            param market_neutral default 0;

            set A ordered;
            param S{A, A};
            param mu{A} default 0;

            param lb default 0;
            param ub default 1;
            var w{A} >= lb <= ub;

            maximize quadratic_utility:
                sum {i in A} mu[i] * w[i]
                - 0.5 * risk_aversion * sum {i in A, j in A} w[i] * S[i, j] * w[j];
            s.t. portfolio_weights:
                sum {i in A} w[i] = if market_neutral then 0 else 1;

        .. code-block:: python

            ampl.param["risk_aversion"] = risk_aversion
            ampl.param["market_neutral"] = market_neutral
            ampl.solve()

        AMPL version of :func:`pypfopt.EfficientFrontier.max_quadratic_utility` with the same interface:

        :param risk_aversion: risk aversion parameter (must be greater than 0),
                              defaults to 1
        :type risk_aversion: positive float
        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),
                               defaults to False. Requires negative lower weight bound.
        :param market_neutral: bool, optional
        :return: asset weights for the maximum-utility portfolio
        :rtype: OrderedDict
        """
        if risk_aversion <= 0:
            raise ValueError("risk aversion coefficient must be greater than zero")

        ampl = self.ampl
        ampl.param["risk_aversion"] = risk_aversion
        ampl.param["market_neutral"] = 1 if market_neutral else 0
        self._solve("max_quadratic_utility")
        self._save_portfolio()
        return self._make_output_weights(self.weights)

    def add_sector_constraints(self, sector_mapper, sector_lower, sector_upper):
        """
        Adds constraints on the sum of weights of different groups of assets.
        Most commonly, these will be sector constraints e.g portfolio's exposure to
        tech must be less than x%::

            sector_mapper = {
                "GOOG": "tech",
                "FB": "tech",,
                "XOM": "Oil/Gas",
                "RRC": "Oil/Gas",
                "MA": "Financials",
                "JPM": "Financials",
            }

            sector_lower = {"tech": 0.1}  # at least 10% to tech
            sector_upper = {
                "tech": 0.4, # less than 40% tech
                "Oil/Gas": 0.1 # less than 10% oil and gas
            }

        Corresponding AMPL code:

        .. code-block:: ampl

            param lb default 0;
            param ub default 1;
            var w{A} >= lb <= ub;

            set SECTORS default {};
            set SECTOR_MEMBERS{SECTORS};
            param sector_lower{SECTORS} default -Infinity;
            param sector_upper{SECTORS} default Infinity;

            s.t. sector_constraints_lower{s in SECTORS: sector_lower[s] != -Infinity}:
                sum {i in SECTOR_MEMBERS[s]} w[i] >= sector_lower[s];
            s.t. sector_constraints_upper{s in SECTORS: sector_upper[s] != Infinity}:
                sum {i in SECTOR_MEMBERS[s]} w[i] <= sector_upper[s];

        .. code-block:: python

            sectors = set(sector_mapper.values())
            ampl.set["SECTORS"] = sectors
            for sector in sectors:
                ampl.set["SECTOR_MEMBERS"][sector] = [
                    ticker for ticker, s in sector_mapper.items() if s == sector
                ]
            ampl.param["sector_lower"] = sector_lower
            ampl.param["sector_upper"] = sector_upper

        AMPL version of :func:`pypfopt.EfficientFrontier.add_sector_constraints` with the same interface:

        :param sector_mapper: dict that maps tickers to sectors
        :type sector_mapper: {str: str} dict
        :param sector_lower: lower bounds for each sector
        :type sector_lower: {str: float} dict
        :param sector_upper: upper bounds for each sector
        :type sector_upper: {str:float} dict
        """
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

    def _save_portfolio(self, risk_free_rate=None):
        ampl = self.ampl
        self.sigma = ampl.get_value("sqrt(sum {i in A, j in A} w[i] * S[i, j] * w[j])")
        self.mu = ampl.get_value("sum {i in A} mu[i] * w[i]")
        if risk_free_rate is None:
            risk_free_rate = ampl.param["risk_free_rate"].value()
        self.sharpe = (self.mu - risk_free_rate) / self.sigma
        self.weights = np.array([v for _, v in ampl.var["w"].get_values().to_list()])

    def portfolio_performance(self, verbose=False, risk_free_rate=0.02):
        """
        After optimising, calculate (and optionally print) the performance of the optimal
        portfolio. Currently calculates expected return, volatility, and the Sharpe ratio.

        :param verbose: whether performance should be printed, defaults to False
        :type verbose: bool, optional
        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.
                               The period of the risk-free rate should correspond to the
                               frequency of expected returns.
        :type risk_free_rate: float, optional
        :raises ValueError: if weights have not been calcualted yet
        :return: expected return, volatility, Sharpe ratio.
        :rtype: (float, float, float)
        """
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
