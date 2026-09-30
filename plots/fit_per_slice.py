from plotly.subplots import make_subplots
import numpy as np

from src.market_data import MarketData
from src.svi import SVISurface, SVISlice


market_data = MarketData("^SPX")
svi_surface = SVISurface(market_data=market_data)
calls_data = svi_surface.data
svi_results = svi_surface.results

n = len(svi_results)
cols = 3
rows = -(-n // cols)
subplot_titles = [f"T={svi.T:.3f}" for svi in svi_results]
fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

for i in range(n):
    a = svi_results[i].a
    b = svi_results[i].b
    rho = svi_results[i].rho
    m = svi_results[i].m
    sigma = svi_results[i].sigma
    T = svi_results[i].T

    slice_data = calls_data[calls_data["timeToExpiry"] == T]

    w_market = slice_data["totalVariance"]
    y_market = slice_data["logMoneyness"]

    row = i // cols + 1
    col = i % cols + 1
    grid_y = np.linspace(svi_results[i].y.min() - 0.01, svi_results[i].y.max() + 0.01, 500)
    w = SVISlice.w(grid_y, a, b, rho, m, sigma)
    fig.add_scatter(x=grid_y, y=w, row=row, col=col, showlegend=False, line=dict(color="black"))
    fig.add_scatter(x=y_market, y=w_market, mode="markers", row=row, col=col, marker=dict(size=2, color="red"),
                    showlegend=False)
    fig.update_xaxes(title_text="y")
    fig.update_yaxes(title_text="w")

fig.update_layout(height=300*rows, title_text="SVI Calibration per Slice")

fig.show()