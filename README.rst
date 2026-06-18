==============================
Assignment 1 — starter package
==============================

*(This folder is generated from ``../solution/`` — do not edit it by hand;
edit the solution and re-run ``generate_assignment.py``.)*

Implement the stubbed functions in ``spectra_filtering/`` until the tests pass.

Layout
------

- ``spectra_filtering/`` — the package. Worked helpers (loading, gap-filling, the
  Tukey filter, the Parseval check, the frequency axis, the synthetic tone) are
  provided. You implement the functions that ``raise NotImplementedError``:
  ``spectra.welch_psd``, ``spectra.raw_periodogram``,
  ``filters.butterworth_squared_response``, and the ``analysis`` helpers.
- ``tests/`` — ``pytest`` checks that pin each function's contract.

For Assignment 1 you only need ``analysis.summary_stats`` and ``spectra.welch_psd``
(plus the provided ``parseval_ratio`` and ``filters.tukey_lowpass``). The rest are
for later.

Run::

    pip install -r requirements.txt
    pytest -q          # implement until the tests you need are green

Each stub's docstring states exactly what to return.
