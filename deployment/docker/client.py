import requests
import pandas as pd
import numpy as np
import sys
import os


def solve(method, endpoint, tickers, mu, S, problem_type):
    print(method, endpoint)
    data = {
        "problem_type": problem_type,
        "tickers": tickers,
        "mu": mu.tolist(),
        "S": S.values.tolist(),
        "solver": "gurobi",
        "solver_options": "outlev=1 timelim=10",
        "target_volatility": 0.15,
        "target_return": 0.07,
        "market_neutral": False,
        "card": 10,
    }

    username, password = "admin", "password"
    headers = {"Content-type": "application/json", "Accept": "text/plain"}

    response = requests.request(
        method, endpoint, auth=(username, password), json=data, headers=headers
    )

    print(response.status_code)
    try:
        print(response.json()["output"])
    except:
        pass
    print(response.content.decode())
    # print(response.json())


from pypfopt import expected_returns, risk_models
import yfinance as yf
import pandas as pd


def list_sp500():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = table[0]
    tickers = df.Symbol.tolist()
    return [t for t in tickers if t not in ("ABT", "BF.B", "BRK.B")]


if __name__ == "__main__":
    # tickers = [
    #     "MSFT",
    #     "AMZN",
    #     "KO",
    #     "MA",
    #     "COST",
    #     "LUV",
    #     "XOM",
    #     "PFE",
    #     "JPM",
    #     "UNH",
    #     "ACN",
    #     "DIS",
    #     "GILD",
    #     "F",
    #     "TSLA",
    # ]
    if not os.path.isfile("prices.csv"):
        tickers = list_sp500()
        ohlc = yf.download(tickers, period="max")
        prices = ohlc["Adj Close"].dropna(how="all")
        prices.to_csv("prices.csv")

    prices = pd.read_csv("prices.csv", index_col="Date")
    tickers = prices.columns.tolist()
    mu = expected_returns.capm_return(prices)
    print("NaNs:", mu[np.isnan(mu)])
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    host, port = "127.0.0.1", 80
    if len(sys.argv) >= 2:
        host = sys.argv[1]
        if ':' in host:
            host, port = host.split(':')
    if len(sys.argv) >= 3:
        port = sys.argv[2]

    print(f"Tickers: {tickers}")
    print(f"mu:\n{mu}")
    print(f"S:\n{S}")

    # solve("PUT", f"http://{host}:{port}/solve", tickers, mu, S, "min_volatility")

    for i in range(10):
        solve("PUT", f"http://{host}:{port}/solve", tickers, mu, S, "min_volatility")
        solve("PUT", f"http://{host}:{port}/solve", tickers, mu, S, "efficient_risk")
        solve("PUT", f"http://{host}:{port}/solve", tickers, mu, S, "efficient_return")
