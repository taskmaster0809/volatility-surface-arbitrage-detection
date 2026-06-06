from datetime import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm
import numpy as np


MIN_PRICE = 1 # SPX options minimum price


def get_time_to_expiry(date: str, today):
    return (dt.strptime(date, "%Y-%m-%d").date() - today).days / 365.25


class MarketData:
    def __init__(self, ticker: str):
        self.today = dt.today().date()
        self.ticker = yf.Ticker(ticker)
        expiries = self.ticker.options  # Option expiries available

        self.expiries = []
        for expiry in expiries:
            if 30 / 365.25 <= get_time_to_expiry(expiry, self.today) <= 2:
                # Only using expiries from 1 month to 2 years
                self.expiries.append(expiry)

        # Spot price of underlying: S0 in the model
        self.spot = self.ticker.history(period="1d")["Close"].iloc[-1]

        irx = yf.Ticker("^IRX")
        # US 13 Week Treasury Bill: r in the model
        self.interest_rate = irx.history(period="1d")["Close"].iloc[-1] / 100

    def black_scholes_price(self, K, T, sigma):
        d1 = (np.log(self.spot / K) + (self.interest_rate + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return self.spot * norm.cdf(d1) - K * np.exp(-self.interest_rate * T) * norm.cdf(d2)

    def calls_one_date(self, date: str):
        option_chain = self.ticker.option_chain(date)
        time_to_expiry = get_time_to_expiry(date, self.today)
        forward = self.spot * np.exp(self.interest_rate * time_to_expiry)
        calls = option_chain.calls[["strike", "lastPrice", "bid", "ask"]]
        y = np.log(calls["strike"] / forward)
        calls = calls.loc[ np.abs(y) <= 0.5 ] # Filtering out deep OTM and ITM options

        calls["marketPrice"] = np.where(
            (calls["bid"] > 0) & (calls["ask"] > 0),
            (calls["bid"] + calls["ask"])/2,
            calls["lastPrice"]
        )  # Set last price as the market price in case of illiquid options

        calls["timeToExpiry"] = time_to_expiry

        # Cleaning data
        calls = calls[calls["marketPrice"] > MIN_PRICE] # Filtering cheap deep OTM options for SPX
        calls = calls.dropna(subset=["marketPrice"])
        calls.reset_index(drop=True, inplace=True)

        # if len(calls) > 15:
        #     indices = np.linspace(0, len(calls) - 1, 15, dtype=int)
        #     calls = calls.iloc[indices]

        return calls[["strike", "timeToExpiry", "marketPrice"]]

    def implied_vol(self, row):
        def objective(imp_vol):
            return self.black_scholes_price(row.strike, row.timeToExpiry, imp_vol) - row.marketPrice

        try:
            return brentq(f=objective, a=1e-6, b=5)
        except ValueError:
            return np.nan

    def calls(self):
        # Using multiple threads to do API calls
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.calls_one_date, self.expiries))

        calls = pd.concat(results, ignore_index=True)

        calls["impliedVol"] = [self.implied_vol(row) for row in calls.itertuples(index=False)]

        calls.dropna(subset=["impliedVol"], inplace=True)
        calls.reset_index(drop=True, inplace=True)

        calls = calls[["strike", "timeToExpiry", "impliedVol"]]
        calls["totalVariance"] = (calls["impliedVol"] ** 2) * calls["timeToExpiry"]
        calls["F"] = self.spot * np.exp(self.interest_rate * calls["timeToExpiry"])
        calls["logMoneyness"] = np.log(calls["strike"] / calls["F"])

        return calls


data = MarketData("^SPX")
print(data.calls())
