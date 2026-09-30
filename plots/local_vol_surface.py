import numpy as np
import plotly.graph_objects as go

from src.market_data import MarketData
from src.svi import SVISurface
from src.local_vol import local_vol

def local_vol_surface(surface: SVISurface, epsilon=0.1):
    svi_results = np.array(surface.results)
    time_to_expiries = np.array([result.T for result in svi_results])
    min_ys = []
    max_ys = []
    for result in svi_results:
        y = result.y
        min_ys.append(min(y))
        max_ys.append(max(y))

    min_y = min(min_ys)
    max_y = max(max_ys)

    grid_y = np.linspace(min_y - epsilon, max_y + epsilon, 1000)
    local_var = [local_vol(surface, grid_y, T) for T in time_to_expiries]

    fig = go.Figure(go.Surface(x=grid_y, y=time_to_expiries, z=np.sqrt(local_var)))
    fig.update_layout(title="Local Volatility Surface", title_font_size=25, hoverlabel=dict(font_size=16),
                      autosize=True, height=800, scene=dict(xaxis_title="Log-Moneyness (y)",
                                                            yaxis_title="Time To Expiry (T)",
                                                            zaxis_title="Local Vol"
                                                            )
                      )
    fig.update_traces(hovertemplate="Log-Moneyness: %{x:.4f}<br>" +
                                    "Time to Expiry (years): %{y:.2f}<br>" +
                                    "Local Volatility: %{z:.4f}<extra></extra>"

                      )

    fig.show()


local_vol_surface(SVISurface(MarketData("^SPX")))

