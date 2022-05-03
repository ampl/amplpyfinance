import io
import pandas as pd
import numpy as np
from utils import run
from contextlib import redirect_stdout


class InputData:
    def __init__(self, data):
        self.problem_type = data.get("problem_type")
        self.solver = data.get("solver", "gurobi")
        self.solver_options = data.get("solver_options", "outlev=1")
        self.tickers = data["tickers"]
        self.mu = None
        if "mu" in data:
            self.mu = np.array(data["mu"])
        self.target_volatility = data.get("target_volatility", None)
        self.target_return = data.get("target_return", None)
        self.market_neutral = data.get("market_neutral", False)
        self.card = data.get("card", None)
        self.S = pd.DataFrame(data["S"], index=self.tickers, columns=self.tickers)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self)

    def to_json(self):
        return str(self)


def solve_ef(data):
    input_data = InputData(data)

    from amplpyfinance import EfficientFrontierWithAMPL

    with io.StringIO() as buf, redirect_stdout(buf):
        _, out, err = run(["ampl", "-vvq"])
        print(f"{out.decode()}\n{err.decode()}")

        ef = EfficientFrontierWithAMPL(
            input_data.mu,
            input_data.S,
            solver=input_data.solver,
            solver_options=input_data.solver_options,
        )
        if input_data.card is not None:
            ef.ampl.param["card_ub"] = input_data.card

        if input_data.problem_type == "min_volatility":
            ef.min_volatility()
        if input_data.problem_type == "efficient_risk":
            assert input_data.target_volatility is not None
            ef.efficient_risk(
                target_volatility=input_data.target_volatility,
                market_neutral=input_data.market_neutral,
            )
        elif input_data.problem_type == "efficient_return":
            assert input_data.target_return is not None
            ef.efficient_return(
                target_return=input_data.target_return,
                market_neutral=input_data.market_neutral,
            )
        output = buf.getvalue()

    mu, sigma, sharpe = ef.portfolio_performance(verbose=True)
    return {
        "mu": mu,
        "sigma": sigma,
        "sharpe": sharpe,
        "output": output,
    }
