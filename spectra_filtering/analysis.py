"""Time-domain characterisation helpers for Assignment 1.

``summary_stats``, ``seasonal_cycle`` and ``decorrelation_timescale`` characterise a
series in the time domain. Each docstring states exactly what to return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def summary_stats(values: np.ndarray) -> dict[str, float]:
    """Basic descriptive statistics of a (possibly gappy) series.

    Parameters
    ----------
    values : numpy.ndarray, shape (N,)
        The series, which may contain ``NaN``.

    Returns
    -------
    stats : dict[str, float]
        Dictionary with keys:

        ``n``
            Total number of samples (including gaps).
        ``n_missing``
            Number of ``NaN`` values.
        ``mean``, ``std``, ``median``, ``min``, ``max``
            Computed over the finite values (NaN-aware).
        ``range``
            ``max - min``.

    Notes
    -----
    Use the NaN-aware reductions (``numpy.nanmean`` etc.) so gaps do not poison the
    statistics. Decide and document whether ``std`` uses ``ddof=0`` or ``1``.

    TODO (student): implement and return the dictionary.
    """
    v = np.asarray(values, dtype="float64")
    finite = v[np.isfinite(v)]
    return {
        "n":         float(v.size),
        "n_missing": float(np.sum(~np.isfinite(v))),
        "mean":      float(np.nanmean(v)),
        "std":       float(np.nanstd(v, ddof=1)),
        "median":    float(np.nanmedian(v)),
        "min":       float(np.nanmin(v)),
        "max":       float(np.nanmax(v)),
        "range":     float(np.nanmax(v) - np.nanmin(v)),
    }


def seasonal_cycle(
    time: np.ndarray,
    values: np.ndarray,
    by: str = "month",
) -> pd.DataFrame:
    s = pd.Series(
        np.asarray(values, dtype="float64"),
        index=pd.DatetimeIndex(time),
    )
    if by == "month":
        groups = s.groupby(s.index.month)
    elif by == "dayofyear":
        groups = s.groupby(s.index.dayofyear)
    else:
        raise ValueError(f"unsupported by={by!r}")
    return groups.agg(["mean", "median"])
    """Climatological seasonal cycle (mean and median per calendar period).

    Parameters
    ----------
    time : numpy.ndarray, dtype ``datetime64``
        Time coordinate, same length as ``values``.
    values : numpy.ndarray, shape (N,)
        The series.
    by : {"month", "dayofyear"}, optional
        Grouping period. Default ``"month"`` (a 12-point climatology).

    Returns
    -------
    clim : pandas.DataFrame
        Indexed by the period (e.g. months 1–12), with columns ``"mean"`` and
        ``"median"``.

    Notes
    -----
    Build a ``pandas`` object indexed by ``time`` and use a ``groupby`` on the
    period (e.g. ``s.groupby(s.index.month)``) with ``.agg(["mean", "median"])``.
    The **seasonal range** is then ``clim["mean"].max() - clim["mean"].min()``.

    TODO (student): implement with a pandas groupby and return the DataFrame.
    """
    


def decorrelation_timescale(
    values: np.ndarray,
    dt: float,
) -> tuple[float, float]:
    
    x = np.asarray(values, dtype="float64")
    N = x.size
    x = x - x.mean()
 
    # Full autocovariance via FFT, then normalise to R(0) = 1
    acov = np.correlate(x, x, mode="full")
    # Take non-negative lags only (second half + centre)
    acov = acov[N - 1:]
    R = acov / acov[0]  # normalise so R(0) = 1
 
    # Integrate up to first zero crossing
    tau = 0.0
    for i in range(len(R) - 1):
        if R[i] < 0:
            break
        tau += dt * (R[i] + R[i + 1]) / 2.0
 
    ndof = N * dt / tau - 1.0
    return tau, ndof
 
    """Integral decorrelation timescale and effective degrees of freedom.

    Parameters
    ----------
    values : numpy.ndarray, shape (N,)
        Evenly sampled series with **no gaps** (interpolate first if needed).
    dt : float
        Sampling interval (same time units you want the timescale in).

    Returns
    -------
    integral_scale : float
        Integral timescale ``tau`` (in units of ``dt``).
    ndof : float
        Effective degrees of freedom, ``N * dt / tau - 1``.

    Notes
    -----
    Integral-timescale method (after E. Frajka-Williams' ``calc_ndof.m``):

    1. Remove the mean.
    2. Form the **normalised autocovariance** ``R`` (autocorrelation, ``R(0) = 1``)
       at lags ``0, dt, 2*dt, ...`` (the non-negative lags).
    3. Integrate ``R`` from lag 0 **until its first zero crossing**, by the
       trapezoidal rule:
       ``tau = sum_i dt * (R[i] + R[i+1]) / 2`` while ``R[i] >= 0``.
       This one-sided integral (not doubled) is the integral timescale.
    4. ``ndof = N * dt / tau - 1``  (``N`` = number of samples).

    A white series gives ``tau`` of order ``dt`` and ``ndof`` close to ``N``; a
    strongly autocorrelated series gives a large ``tau`` and few independent samples.

    TODO (student): implement the autocovariance, the zero-crossing integral, and
    return ``(integral_scale, ndof)``.
    """
    #raise NotImplementedError("Implement the integral-timescale d.o.f. estimate.")
