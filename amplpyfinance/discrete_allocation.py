from pypfopt import DiscreteAllocation
from pypfopt import exceptions
from amplpy import AMPL
import collections


class DiscreteAllocationWithAMPL(DiscreteAllocation):
    def lp_portfolio(self, reinvest=False, verbose=False, solver="cbc", options=None):
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
            long_alloc, long_leftover = da1.lp_portfolio(solver=solver, options=options)

            if verbose:
                print("\nAllocating short sub-portfolio:")
            da2 = DiscreteAllocationWithAMPL(
                shorts,
                self.latest_prices[shorts.keys()],
                total_portfolio_value=short_val,
            )
            short_alloc, short_leftover = da2.lp_portfolio(
                verbose=verbose, solver=solver, options=options
            )
            short_alloc = {t: -w for t, w in short_alloc.items()}

            # Combine and return
            self.allocation = long_alloc.copy()
            self.allocation.update(short_alloc)
            self.allocation = self._remove_zero_positions(self.allocation)
            return self.allocation, long_leftover + short_leftover

        # https://pyportfolioopt.readthedocs.io/en/latest/Postprocessing.html?highlight=discrete#integer-programming
        ampl = AMPL()
        ampl.option["solver"] = solver
        if options is not None:
            for opt, value in options.items():
                ampl.option[opt] = value
        ampl.eval(
            r"""
        param n;
        set I := 1..n;
        param p{I};
        param w{I};
        var x{I} >= 0 integer;
        var u{I} >= 0;
        var r >= 0;
        param total_portfolio_value;
        minimize objective: r + sum{i in I} u[i];
        s.t. norm1{i in I}: u[i] >= w[i] * total_portfolio_value - x[i] * p[i];
        s.t. norm2{i in I}: -u[i] <= w[i] * total_portfolio_value - x[i] * p[i];
        s.t. total_value: r + sum{i in I} x[i] * p[i] = total_portfolio_value;
        """
        )
        p = self.latest_prices.values
        ampl.param["n"] = len(p)
        ampl.param["p"] = p
        ampl.param["w"] = [i[1] for i in self.weights]
        ampl.param["total_portfolio_value"] = self.total_portfolio_value
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
