 
#Assignment 1: SAMBA 34.5°S AMOC analysis.


from __future__ import annotations
 
import matplotlib.pyplot as plt
import numpy as np
 
from spectra_filtering.analysis import summary_stats
from spectra_filtering.data_io import fill_gaps
from spectra_filtering.filters import tukey_lowpass, butterworth_squared_response, nyquist_frequency
from spectra_filtering.spectra import welch_psd, raw_periodogram, parseval_ratio, frequency_axis
 
#Load data 

from amocatlas import read
ds = read.samba()
print(ds)

VAR = "UPPER_TRANSPORT"
time    = ds["TIME"].values
values  = ds[VAR].values.astype("float64")
dt_days = float(np.median(np.diff(time)) / np.timedelta64(1, "D"))
 
print(f"\nRecord length : {len(values)} samples")
print(f"Sampling      : {dt_days:.3f} days")
print(len(ds["TIME"].values))  # kaç gerçek ölçüm var?
print(ds["TIME"].values[0], ds["TIME"].values[-1])
# Gaps 
stats_raw = summary_stats(values)
print(f"\nSummary stats (raw):")
for k, v in stats_raw.items():
    print(f"  {k:12s}: {v:.4g}")
 
""". Just checked if the range is a mistake
ds = read.samba()
v = ds["UPPER_TRANSPORT"].values
t = ds["TIME"].values

# Bu iki değerin tarihleri
idx_max = np.argmax(v)
idx_min = np.argmin(v)
print(f"Max: {v[idx_max]:.2f} Sv — {t[idx_max]}")
print(f"Min: {v[idx_min]:.2f} Sv — {t[idx_min]}")

# Etrafındaki 5 gün
print("\nMax etrafı:")
print(v[idx_max-3:idx_max+4])
print("\nMin etrafı:")
print(v[idx_min-3:idx_min+4])"""

values_filled = fill_gaps(values)
 
# Time-domain filter 
# 30-day Tukey lowpass
CUTOFF_DAYS = 30
window_samples = int(CUTOFF_DAYS / dt_days)
values_lp = tukey_lowpass(values_filled, window=window_samples)
 
# Spectra 
# Segment length: 2 year for daily data → 4 years for 12-hour data
seg_len = int(730 / dt_days)  
freq_w, psd_w         = welch_psd(values_filled, dt_days, segment_length=seg_len, overlap=0.5)
freq_lp, psd_lp       = welch_psd(values_lp,     dt_days, segment_length=seg_len, overlap=0.5)
 
ratio = parseval_ratio(values_filled, freq_w, psd_w)
print(f"\nParseval ratio (Welch): {ratio:.4f}  ")
 
# Chi-squared 95% confidence band
# Approximate dof: 2 * n_segments (for 50% overlap, n_seg ≈ 2*N/seg_len - 1)
n_seg = int(2 * len(values_filled) / seg_len) - 1
dof   = 1.5 * n_seg 
from scipy.stats import chi2
ci_lo = dof / chi2.ppf(0.975, dof)
ci_hi = dof / chi2.ppf(0.025, dof)
 
# Butterworth response for filter illustration (Part C)
f_cut_cpd = 1.0 / CUTOFF_DAYS
freq_resp  = np.linspace(0, nyquist_frequency(dt_days), 1000)
h2         = butterworth_squared_response(freq_resp, f_cut=f_cut_cpd, order=5, zero_phase=True)

#partA
fig0, ax0 = plt.subplots(figsize=(7, 4))
ax0.hist(values_filled, bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
ax0.axvline(stats_raw["mean"],   color="darkorange", lw=1.5, label=f'Mean = {stats_raw["mean"]:.2f} Sv')
ax0.axvline(stats_raw["median"], color="green",      lw=1.5, ls="--", label=f'Median = {stats_raw["median"]:.2f} Sv')
ax0.set_xlabel("UPPER_TRANSPORT anomaly (Sv)")
ax0.set_ylabel("Count")
ax0.set_title("SAMBA 34.5°S — Distribution of upper-cell MOC anomaly")
ax0.legend()
fig0.tight_layout()
fig0.savefig("fig0_histogram.png", dpi=150)

 

# PARTB — Raw vs filtered series
import pandas as pd
t_plot = pd.DatetimeIndex(time)
 
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(t_plot, values_filled, color="steelblue",  lw=0.6, alpha=0.6, label="Original Data")
ax1.plot(t_plot, values_lp,     color="darkblue",  lw=1.6, label=f"{CUTOFF_DAYS}-day Tukey low-pass")
ax1.set_xlabel("Time")
ax1.set_ylabel("MOC anomaly (Sv)")
ax1.set_title(f"SAMBA 34.5°S — {VAR}")
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig("fig1_timeseries.png", dpi=150)

 
# Spectrum: raw vs filtered
fig2, axes = plt.subplots(1, 2, figsize=(13, 5))
 
ax2 = axes[0]
ax2.loglog(freq_w[1:],  psd_w[1:],  color="steelblue",  lw=1.2, label="Welch PSD (raw)")
ax2.loglog(freq_lp[1:], psd_lp[1:], color="darkorange",  lw=1.2, label=f"Welch PSD ({CUTOFF_DAYS}-day LP)")
# Confidence band around raw PSD
ax2.fill_between(freq_w[1:], psd_w[1:]*ci_lo, psd_w[1:]*ci_hi,
                 color="steelblue", alpha=0.15, label="95% CI (raw)")
ax2.axvline(f_cut_cpd, color="grey", ls="--", lw=0.8, label=f"Cutoff (1/{CUTOFF_DAYS} cpd)")
ax2.set_xlabel("Frequency (cycles per day)")
ax2.set_ylabel("PSD (Sv² / cpd)")
ax2.set_title("Power spectrum — SAMBA 34.5°S")
ax2.legend(fontsize=8)
ax2.grid(True, which="both", alpha=0.2)
 
# Part C: filter response
ax3 = axes[1]
ax3.semilogx(freq_resp[1:], h2[1:], color="darkorange", lw=1.8,
             label=f"Butterworth |H|² (n=5, zero-phase)")
ax3.axvline(f_cut_cpd, color="grey", ls="--", lw=0.8, label=f"Cutoff (1/{CUTOFF_DAYS} cpd)")
ax3.axhline(0.5,       color="grey", ls=":",  lw=0.6)
ax3.set_xlabel("Frequency (cycles per day)")
ax3.set_ylabel("|H(f)|²")
ax3.set_title("Filter squared response")
ax3.set_ylim(0, 1.05)
ax3.legend(fontsize=8)
ax3.grid(True, which="both", alpha=0.2)
 
fig2.tight_layout()
fig2.savefig("fig2_spectrum.png", dpi=150)

 
plt.show()


