from pypfopt import DiscreteAllocation
from pypfopt import exceptions
from amplpy import AMPL
import collections


class DiscreteAllocationWithAMPL(DiscreteAllocation):
    """
    Generate a discrete portfolio allocation from continuous weights.

    AMPL version of :func:`pypfopt.DiscreteAllocation` with similar interface.

    Instance variables:

    - Inputs:

        - ``weights`` - dict
        - ``latest_prices`` - pd.Series or dict
        - ``total_portfolio_value`` - int/float
        - ``short_ratio``- float

    - Output: ``allocation`` - dict

    Public methods:

    - :func:`greedy_portfolio()` - uses a greedy algorithm
    - :func:`lp_portfolio()` - uses linear programming
    """

    def __init__(
        self, weights, latest_prices, total_portfolio_value=10000, short_ratio=None
    ):
        """
        :param weights: continuous weights generated from the ``efficient_frontier`` module
        :type weights: dict
        :param latest_prices: the most recent price for each asset
        :type latest_prices: pd.Series
        :param total_portfolio_value: the desired total value of the portfolio, defaults to 10000
        :type total_portfolio_value: int/float, optional
        :param short_ratio: the short ratio, e.g 0.3 corresponds to 130/30. If None,
                            defaults to the input weights.
        :type short_ratio: float, defaults to None.
        :raises TypeError: if ``weights`` is not a dict
        :raises TypeError: if ``latest_prices`` isn't a series
        :raises ValueError: if ``short_ratio < 0``
        """
        super().__init__(weights, latest_prices, total_portfolio_value, short_ratio)

    def lp_portfolio(
        self, reinvest=False, verbose=False, solver="gurobi", solver_options=None
    ):
        r"""
        Convert continuous weights into a discrete portfolio allocation
        using integer programming.

        Model from `pypfopt <https://pyportfolioopt.readthedocs.io/en/latest/Postprocessing.html?highlight=discrete#integer-programming>`_:

        - :math:`T \in \mathbb{R}` is the total dollar value to be allocated
        - :math:`p \in \mathbb{R}^n` is the array of latest prices
        - :math:`w \in \mathbb{R}^n` is the set of target weights
        - :math:`x \in \mathbb{Z}^n` is the integer allocation (i.e the result)
        - :math:`r \in \mathbb{R}` is the remaining unallocated value, i.e :math:`r = T - x \cdot p`.

        The optimization problem is then given by:

        .. math::

            \begin{equation*}
            \begin{aligned}
            & \underset{x \in \mathbb{Z}^n}{\text{minimize}} & & r + \lVert wT - x \odot p \rVert_1  \\
            & \text{subject to} & & r + x \cdot p = T\\
            \end{aligned}
            \end{equation*}


        Corresponding AMPL code:

        .. code-block:: ampl

            param n;
            param p{1..n};
            param w{1..n};
            param total_portfolio_value;

            var x{1..n} >= 0 integer;
            var u{1..n} >= 0;
            var r >= 0;

            minimize objective:
                r + sum{i in 1..n} u[i];

            s.t. norm1{i in 1..n}:
                u[i] >= w[i] * total_portfolio_value - x[i] * p[i];
            s.t. norm2{i in 1..n}:
                -u[i] <= w[i] * total_portfolio_value - x[i] * p[i];
            s.t. total_value:
                r + sum{i in 1..n} x[i] * p[i] = total_portfolio_value;

        .. code-block:: python

            ampl.param["n"] = len(latest_prices)
            ampl.param["p"] = latest_prices
            ampl.param["w"] = weights
            ampl.param["total_portfolio_value"] = total_portfolio_value
            ampl.solve()

        AMPL version of :func:`pypfopt.DiscreteAllocation.lp_portfolio` with similar interface:

        :param reinvest: whether or not to reinvest cash gained from shorting
        :type reinvest: bool, defaults to False
        :param verbose: print error analysis?
        :type verbose: bool
        :param solver: name of the AMPL solver to use.
        :type solver: str
        :param solver_options: options for the given solver
        :type solver_options: str
        :return: the number of shares of each ticker that should be purchased, along with the amount
                of funds leftover.
        :rtype: (dict, float)
        """

        if any([w < 0 for _, w in self.weights]):
            longs = {t: w for t, w in self.weights if w >= 0}
            shorts = {t: -w for t, w in self.weights if w < 0}

            # Make them sum to one
            long_total_weight = sum(longs.values())
            short_total_weight = sum(shorts.values())
            longs = {t: w / long_total_weight for t, w in longs.items()}
            shorts = {t: w / short_total_weight for t, w in shorts.items()}

            # Construct long-only discrete allocations for each
            short_val = self.total_portfolio_value * self.short_ratio
            long_val = self.total_portfolio_value
            if reinvest:
                long_val += short_val

            if verbose:
                print("\nAllocating long sub-portfolio:")
            da1 = DiscreteAllocationWithAMPL(
                longs, self.latest_prices[longs.keys()], total_portfolio_value=long_val
            )
            long_alloc, long_leftover = da1.lp_portfolio(
                solver=solver, solver_options=solver_options
            )

            if verbose:
                print("\nAllocating short sub-portfolio:")
            da2 = DiscreteAllocationWithAMPL(
                shorts,
                self.latest_prices[shorts.keys()],
                total_portfolio_value=short_val,
            )
            short_alloc, short_leftover = da2.lp_portfolio(
                verbose=verbose, solver=solver, solver_options=solver_options
            )
            short_alloc = {t: -w for t, w in short_alloc.items()}

            # Combine and return
            self.allocation = long_alloc.copy()
            self.allocation.update(short_alloc)
            self.allocation = self._remove_zero_positions(self.allocation)
            return self.allocation, long_leftover + short_leftover

        ampl = AMPL()
        ampl.eval(
            r"""
            param n;
            param p{1..n};
            param w{1..n};
            param total_portfolio_value;

            var x{1..n} >= 0 integer;
            var u{1..n} >= 0;
            var r >= 0;

            minimize objective:
                r + sum{i in 1..n} u[i];

            s.t. norm1{i in 1..n}:
                u[i] >= w[i] * total_portfolio_value - x[i] * p[i];
            s.t. norm2{i in 1..n}:
                -u[i] <= w[i] * total_portfolio_value - x[i] * p[i];
            s.t. total_value:
                r + sum{i in 1..n} x[i] * p[i] = total_portfolio_value;
            """
        )
        p = self.latest_prices.values
        ampl.param["n"] = len(p)
        ampl.param["p"] = p
        ampl.param["w"] = [i[1] for i in self.weights]
        ampl.param["total_portfolio_value"] = self.total_portfolio_value
        ampl.option["solver"] = solver
        if solver_options:
            ampl.option[f"{solver}_options"] = solver_options
        ampl.solve()
        if ampl.get_value("solve_result") != "solved":
            raise exceptions.OptimizationError(
                "Failed to solve. Please check the solver log"
            )
        vals = [int(round(v)) for _, v in ampl.var["x"].get_values().to_list()]
        self.allocation = self._remove_zero_positions(
            collections.OrderedDict(zip([i[0] for i in self.weights], vals))
        )
        r_value = ampl.var["r"].value()
        if verbose:
            print("Funds remaining: {:.2f}".format(r_value))
            self._allocation_rmse_error()
        return self.allocation, r_value
