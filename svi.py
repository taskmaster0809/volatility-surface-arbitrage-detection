from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from market_data import MarketData


@dataclass
class SVIResult:
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    w: np.ndarray


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

    def w(self, y, a, b, rho, m, sigma):
        return a + b * self.phi(y, rho, m, sigma)

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
            b = max(b, 0)
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

    def check_butterfly(self, svi: SVIResult):
        grid_y = np.linspace(self.y.min() - 0.5, self.y.max() + 0.5, 1000)

        def g(y, a, b, rho, m, sigma):
            w = self.w(y, a, b, rho, m, sigma)
            dw = self.dw(y, b, rho, m, sigma)
            d2w = self.d2w(y, b, m, sigma)
            return (1 - y * dw / (2 * w)) ** 2 - ((dw ** 2) / 4) * (1/w + 1/4) + d2w/2

        g_vec = g(grid_y, svi.a, svi.b, svi.rho, svi.m, svi.sigma)
        return np.all(g_vec >= 0), g_vec


class SVISurface:
    def __init__(self, market_data: MarketData):
        self.data = market_data.calls()



