```python
#!/usr/bin/env python3

"""
RMSD AUTOCORRELATION ANALYSIS
=============================================================

This script computes:

1. RMSD for:
   - Global Cα atoms
   - Global backbone
   - Two user-defined regions

2. Time autocorrelation functions (ACF) of RMSD fluctuations

3. Integrated correlation times (τ)

Purpose:
--------
Autocorrelation reveals how long structural fluctuations remain
memory-dependent in time.

Long τ  -> persistent collective motion
Short τ -> fast decorrelation / flexible motion

Supported formats:
------------------
Topology:   AMBER (.parm7)
Trajectory: AMBER NetCDF (.nc)

Example:
--------
python rmsd_autocorrelation.py -p topology.parm7 -t trajectory.nc
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import pandas as pd


# =============================================================
# ARGUMENTS
# =============================================================

parser = argparse.ArgumentParser(
    description="RMSD autocorrelation analysis for MD trajectories"
)

parser.add_argument("-p", "--topology", required=True,
                    help="Topology file (.parm7)")

parser.add_argument("-t", "--trajectory", required=True,
                    help="Trajectory file (.nc)")

parser.add_argument("--dt", type=float, default=100e-12,
                    help="Time per frame in seconds (default: 100 ps)")

args = parser.parse_args()


# =============================================================
# LOAD SYSTEM
# =============================================================

u = mda.Universe(args.topology, args.trajectory)


# =============================================================
# SELECTIONS
# =============================================================

ca_sel = "protein and name CA"
backbone_sel = "protein and backbone"

# Editable user regions
region1_res = [195, 198, 199, 211, 214, 215, 218, 219]
region2_res = [388, 391, 392, 395, 399, 403, 404, 407]

region1_sel = "protein and name CA and resid " + " ".join(map(str, region1_res))
region2_sel = "protein and name CA and resid " + " ".join(map(str, region2_res))


# =============================================================
# RMSD
# =============================================================

def compute_rmsd(universe, selection):
    R = rms.RMSD(universe, universe, select=selection, ref_frame=0)
    R.run()
    return R.rmsd[:, 2]


# =============================================================
# AUTOCORRELATION
# =============================================================

def compute_autocorrelation(signal):

    signal = np.asarray(signal)
    signal = signal - np.mean(signal)

    n = len(signal)

    padded = np.concatenate([signal, np.zeros(n)])
    fft_signal = np.fft.fft(padded)
    power = fft_signal * np.conjugate(fft_signal)

    acf = np.fft.ifft(power).real[:n]
    acf /= acf[0]

    return acf


# =============================================================
# CORRELATION TIME
# =============================================================

def correlation_time(acf, dt):

    zero_crossings = np.where(acf < 0)[0]

    if len(zero_crossings) > 0:
        cutoff = zero_crossings[0]
    else:
        cutoff = len(acf)

    tau = np.trapezoid(acf[:cutoff], dx=dt)

    return tau


# =============================================================
# HUMAN READABLE TIMESCALE
# =============================================================

def human_timescale(t):

    if t < 1e-12:
        return f"{t*1e15:.2f} fs"
    elif t < 1e-9:
        return f"{t*1e12:.2f} ps"
    elif t < 1e-6:
        return f"{t*1e9:.2f} ns"
    else:
        return f"{t:.2e} s"


# =============================================================
# COMPUTE RMSD
# =============================================================

print("Computing RMSD...")

rmsd_ca = compute_rmsd(u, ca_sel)
rmsd_bb = compute_rmsd(u, backbone_sel)
rmsd_r1 = compute_rmsd(u, region1_sel)
rmsd_r2 = compute_rmsd(u, region2_sel)


# =============================================================
# COMPUTE ACF
# =============================================================

print("Computing autocorrelation...")

acf_ca = compute_autocorrelation(rmsd_ca)
acf_bb = compute_autocorrelation(rmsd_bb)
acf_r1 = compute_autocorrelation(rmsd_r1)
acf_r2 = compute_autocorrelation(rmsd_r2)

time = np.arange(len(acf_ca)) * args.dt


# =============================================================
# CORRELATION TIMES
# =============================================================

tau_ca = correlation_time(acf_ca, args.dt)
tau_bb = correlation_time(acf_bb, args.dt)
tau_r1 = correlation_time(acf_r1, args.dt)
tau_r2 = correlation_time(acf_r2, args.dt)


# =============================================================
# SUMMARY TABLE
# =============================================================

data = pd.DataFrame({
    "Region": [
        "Cα",
        "Backbone",
        "Region 1",
        "Region 2"
    ],
    "τ (human)": [
        human_timescale(tau_ca),
        human_timescale(tau_bb),
        human_timescale(tau_r1),
        human_timescale(tau_r2)
    ]
})


# =============================================================
# PLOT
# =============================================================

fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(2, 2)

# ACF plot
ax1 = fig.add_subplot(gs[:, 0])

ax1.plot(time, acf_ca, label="Cα")
ax1.plot(time, acf_bb, label="Backbone")
ax1.plot(time, acf_r1, "--", label="Region 1")
ax1.plot(time, acf_r2, "--", label="Region 2")

ax1.axhline(0, linestyle="--")

ax1.set_xscale("log")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("C(t)")
ax1.set_title("RMSD Autocorrelation")
ax1.legend()

# Bar plot
ax2 = fig.add_subplot(gs[0, 1])

ax2.bar(data["Region"], [tau_ca, tau_bb, tau_r1, tau_r2])

ax2.set_title("Integrated Correlation Time")
ax2.set_ylabel("τ (s)")

# Table
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis("off")

table = ax3.table(
    cellText=data.values,
    colLabels=data.columns,
    loc="center"
)

table.scale(1, 1.5)

plt.tight_layout()
plt.show()


# =============================================================
# OUTPUT
# =============================================================

print("\n=== Correlation Times ===")

for region, tau in zip(data["Region"], data["τ (human)"]):
    print(f"{region:15s} τ = {tau}")
```
