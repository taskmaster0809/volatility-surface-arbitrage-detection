import numpy as np
import pytest

from src.svi import SVIResult, SVISlice


def calibration(params=None):
    if params is None:
        params = {'a': 0.04, 'b': 0.2, "rho": -0.7, 'm': 0, "sigma": 0.2}
    y_grid = np.linspace(-2, 2, 4000)
    w_true = SVISlice.w(y_grid, params['a'], params['b'], params["rho"], params['m'], params["sigma"])
    rng = np.random.default_rng(8092004)
    errors = rng.normal(0, 0.001, size=len(w_true))
    w_true += errors

    svi_slice = SVISlice(y_grid, w_true)

    return svi_slice.fit(), params


def test_calibration(rel_tol=0.01, abs_tol=0.01):
    print("\nTesting errors on parameter calibration...")
    result, params = calibration()
    for param, value in vars(result).items():
        if param != 'w' and param != 'T' and param != 'y':
            if params[param] == pytest.approx(0):
                error = abs(value - params[param])
                assert error < abs_tol
            else:
                error = abs((value - params[param]) / params[param])
                assert error < rel_tol
            print(f"Error for {param}: {error*100:.4f}%")


def test_check_butterfly_fail(params=None):
    print("\nTesting existence of butterfly arbitrage for some parameters...")
    if params is None:
        params = {'a': 0.04, 'b': 10, "rho": -0.7, 'm': 0, "sigma": 0.001} # Butterfly arbitrage should exist
                                                                           # for large b and small sigma

    svi_result = SVIResult(params['a'], params['b'], params["rho"], params['m'], params["sigma"], w=None)
    is_arbitrage_free, g = SVISlice.check_butterfly(svi_result)
    print(f"g range: [{g.min():.3f}, {g.max():.3f}]")
    assert not is_arbitrage_free


def test_check_butterfly_pass(params=None):
    print("\nTesting non-existence of butterfly arbitrage for some parameters...")
    if params is None:
        params = {'a': 0.04, 'b': 0.2, "rho": -0.7, 'm': 0, "sigma": 0.2} # Butterfly arbitrage shouldn't exist
                                                                           # for these parameters

    svi_result = SVIResult(params['a'], params['b'], params["rho"], params['m'], params["sigma"], w=None)
    is_arbitrage_free, g = SVISlice.check_butterfly(svi_result)
    print(f"g range: [{g.min():.3f}, {g.max():.3f}]")
    assert is_arbitrage_free


def test_check_calendar():
    print("\nTesting non-existence of calendar arbitrage...")
    svi1 = SVIResult(a=0.04, b=0.2, rho=-0.7, m=0, sigma=0.2, w=None, T=0.5)
    svi2 = SVIResult(a=0.09, b=0.2, rho=-0.7, m=0, sigma=0.2, w=None, T=1)
    # svi1.w should be less than svi2.w for these parameters

    is_arbitrage_free, diff = SVISlice.check_calendar(svi1, svi2)

    assert is_arbitrage_free and np.all(diff >= 0)

