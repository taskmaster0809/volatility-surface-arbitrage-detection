import numpy as np

from src.svi import SVISurface, SVISlice


def local_vol(surface: SVISurface, y, T):
    svi_results = np.array(surface.results)
    time_to_expiries = np.array([result.T for result in svi_results])
    assert len(time_to_expiries) >= 2, "Insufficient number of slices to construct a surface"

    if T < time_to_expiries[0] or T > time_to_expiries[-1]:
        raise ValueError(f"T={T} outside the observed range of maturities "
                         f"[{time_to_expiries[0]}, {time_to_expiries[-1]}]")

    try:
        point_greater = svi_results[time_to_expiries > T][0]
    except IndexError:
        point_greater = svi_results[time_to_expiries >= T][0]

    try:
        point_lesser = svi_results[time_to_expiries < T][-1]
    except IndexError:
        point_lesser = svi_results[time_to_expiries <= T][-1]

    t1 = point_lesser.T
    t2 = point_greater.T

    w1 = SVISlice.w(y, point_lesser.a, point_lesser.b, point_lesser.rho, point_lesser.m, point_lesser.sigma)
    w2 = SVISlice.w(y, point_greater.a, point_greater.b, point_greater.rho, point_greater.m, point_greater.sigma)

    w_t = (w2 - w1)/(t2 - t1) # Approximate dw/dt using finite differences

    g1 = SVISlice.g(y, point_lesser.a, point_lesser.b, point_lesser.rho, point_lesser.m, point_lesser.sigma)
    g2 = SVISlice.g(y, point_greater.a, point_greater.b, point_greater.rho, point_greater.m, point_greater.sigma)

    g = g1 + (T - t1) / (t2 - t1) * (g2 - g1) # Interpolating g

    local_var = np.where((g <= 0) | (w_t <= 0), np.nan, w_t/g)

    return local_var

