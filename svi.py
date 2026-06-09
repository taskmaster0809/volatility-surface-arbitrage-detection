from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from market_data import MarketData


@dataclass
class SVIResult:
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    w: np.ndarray
    T: float = None
    y: np.ndarray = None


class SVISlice:
    def __init__(self, y, w_market, weights=None):
        self.y = np.asarray(y)
        self.w_market = np.asarray(w_market)

        if weights is None:
            self.weights = np.ones_like(self.w_market)
        else:
            self.weights = np.asarray(weights)

    @staticmethod
    def phi(y, rho, m, sigma):
        return rho * (y - m) + np.sqrt((y - m) ** 2 + sigma ** 2)

    @staticmethod
    def w(y, a, b, rho, m, sigma):
        return a + b * SVISlice.phi(y, rho, m, sigma)

    @staticmethod
    def dw(y, b, rho, m, sigma):
        return b * (rho + (y - m) / np.sqrt((y - m) ** 2 + sigma ** 2))

    @staticmethod
    def d2w(y, b, m, sigma):
        return (b * sigma ** 2) / (((y - m) ** 2 + sigma ** 2) ** 1.5)

    def fit(self, x0=(-0.5, 0, 0.1)):
        def inner(rho, m, sigma):
            phi = self.phi(self.y, rho, m, sigma)
            x = np.column_stack([np.ones_like(phi), phi])
            a, b = np.linalg.lstsq(x, self.w_market, rcond=None)[0]
            b = max(b, 0) # clip b since b >= 0
            if b == 0:
                a = np.mean(self.w_market)
            return a, b

        def residuals(params):
            rho, m, sigma = params
            a, b = inner(rho, m, sigma)
            w = self.w(self.y, a, b, rho, m, sigma)
            if a + b * sigma * np.sqrt(1 - rho ** 2) < 0:
                return np.full_like(self.w_market, 1e12)

            return np.sqrt(self.weights) * (w - self.w_market)

        min_bound = (-1, -np.inf, 1e-12)
        max_bound = (1, np.inf, np.inf)
        result = least_squares(fun=residuals, x0=x0, bounds=[min_bound, max_bound])

        rho_fit, m_fit, sigma_fit = result.x
        a_fit, b_fit = inner(rho_fit, m_fit, sigma_fit)
        w_fit = self.w(self.y, a_fit, b_fit, rho_fit, m_fit, sigma_fit)

        return SVIResult(a_fit, b_fit, rho_fit, m_fit, sigma_fit, w_fit)

    @staticmethod
    def check_butterfly(svi: SVIResult):
        def g(y, a, b, rho, m, sigma):
            w = SVISlice.w(y, a, b, rho, m, sigma)
            dw = SVISlice.dw(y, b, rho, m, sigma)
            d2w = SVISlice.d2w(y, b, m, sigma)
            return (1 - y * dw / (2 * w)) ** 2 - ((dw ** 2) / 4) * (1/w + 1/4) + d2w/2

        y_grid = np.linspace(svi.y.min() - 0.1, svi.y.max() + 0.1, 1000)

        g_vec = g(y_grid, svi.a, svi.b, svi.rho, svi.m, svi.sigma)
        return np.all(g_vec >= 0), g_vec

    @staticmethod
    def check_calendar(svi1: SVIResult, svi2: SVIResult):
        if svi1.T is None or svi2.T is None:
            return None

        if svi1.T > svi2.T:
            svi1, svi2 = svi2, svi1

        y_grid = np.linspace(min(svi1.y.min(), svi2.y.min()) - 0.1, max(svi1.y.max(), svi2.y.max()) + 0.1, 1000)

        w1 = SVISlice.w(y_grid, svi1.a, svi1.b, svi1.rho, svi1.m, svi1.sigma)
        w2 = SVISlice.w(y_grid, svi2.a, svi2.b, svi2.rho, svi2.m, svi2.sigma)

        return np.all(w1 <= w2), w2 - w1


class SVISurface:
    def __init__(self, market_data: MarketData):
        self.data = market_data.calls()
        self.results = self.fit_surface()

    def fit_surface(self):
        group_by_maturity = list(self.data.groupby("timeToExpiry"))
        result = []
        for group in group_by_maturity:
            time_to_expiry = group[0]
            y = group[1]["logMoneyness"]
            w_market = group[1]["totalVariance"]

            # Using vega weights (omitting S because it's constant across maturities)
            weights = np.sqrt(time_to_expiry) * norm.pdf(-y / np.sqrt(w_market) + np.sqrt(w_market) / 2)
            svi_slice = SVISlice(y, w_market, weights)

            svi_result = svi_slice.fit()
            svi_result.T = time_to_expiry
            svi_result.y = y.to_numpy()

            result.append(svi_result)

        return result

    def check_butterfly_all(self):
        is_butterfly_arbitrage = []
        for svi_result in self.results:
            is_butterfly_arbitrage.append( (svi_result.T, SVISlice.check_butterfly(svi_result)) )

        return is_butterfly_arbitrage

    def check_calendar_all(self):
        is_calendar_arbitrage = []
        for index in range(len(self.results) - 1):
            is_calendar_arbitrage.append( (self.results[index].T,
                                           SVISlice.check_calendar(self.results[index], self.results[index + 1])) )

        return is_calendar_arbitrage

    def check_surface_arbitrage(self):
        return \
        {
            "butterfly": self.check_butterfly_all(),
            "calendar":  self.check_calendar_all()
        }

