from __future__ import annotations

import numpy as np
import pytest
from spectra_filtering.analysis import summary_stats
from spectra_filtering.spectra import welch_psd, parseval_ratio


def test_summary_stats_mean_with_gaps() -> None:
    """Mean must equal numpy.nanmean on an array containing NaN gaps."""
    rng = np.random.default_rng(42)
    x = rng.normal(loc=5.0, scale=2.0, size=200)
    x[[10, 50, 150]] = np.nan

    stats = summary_stats(x)

    assert stats["mean"] == pytest.approx(np.nanmean(x), rel=1e-10)
    assert stats["n_missing"] == 3


def test_parseval_welch() -> None:
    """Integral of Welch PSD must equal variance of series to within 5%."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096)

    freq, psd = welch_psd(x, dt_days=1.0, segment_length=512, overlap=0.5)
    ratio = parseval_ratio(x, freq, psd)

    assert ratio == pytest.approx(1.0, abs=0.05)