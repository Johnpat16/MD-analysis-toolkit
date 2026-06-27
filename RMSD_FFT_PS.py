# rmsd_fft_psd.py

```python
#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import rms


# =============================================================
# ARGUMENT PARSER
# =============================================================

parser = argparse.ArgumentParser(
    description="Compute RMSD and FFT-based Power Spectrum from AMBER trajectories"
)

parser.add_argument("-p", "--topology", required=True,
                    help="AMBER topology file (.parm7)")

parser.add_argument("-t", "--trajectory", required=True,
                    help="AMBER trajectory file (.nc)")

parser.add_argument("--dt", type=float, default=100e-12,
                    help="Time step between frames in seconds (default: 100 ps)")

args = parser.parse_args()


# =============================================================
# REGIONS OF INTEREST (EDITABLE)
# =============================================================

region1_res = [195, 198, 199, 211, 214, 215, 218, 219]
region2_res = [388, 391, 392, 395, 399, 403, 404, 407]

region1_sel = "protein and name CA and resid " + " ".join(map(str, region1_res))
region2_sel = "protein and name CA and resid " + " ".join(map(str, region2_res))

ca_sel = "protein and name CA"
backbone_sel = "protein and backbone"


# =============================================================
# LOAD SYSTEM
# =============================================================

u = mda.Universe(args.topology, args.trajectory)


# =============================================================
# RMSD CALCULATION
# =============================================================

def compute_rmsd(universe, selection):
    R = rms.RMSD(universe, universe, select=selection, ref_frame=0)
    R.run()
    return R.rmsd[:, 2]


# =============================================================
# PSD CALCULATION
# =============================================================

def compute_psd(signal, dt):
    n = len(signal)

    signal = signal - np.mean(signal)
    window = np.hanning(n)
    signal_windowed = signal * window

    X = np.fft.rfft(signal_windowed)
    freq = np.fft.rfftfreq(n, d=dt)

    PSD = (np.abs(X) ** 2) / (np.sum(window ** 2) / dt)
    PSD[1:-1] *= 2

    return freq, PSD


# =============================================================
# POWER LAW FIT
# =============================================================

def fit_power_law(freq, psd, fmin=1e9, fmax=5e10):

    mask = (freq > fmin) & (freq < fmax)

    logf = np.log(freq[mask])
    logP = np.log(psd[mask])

    coeffs = np.polyfit(logf, logP, 1)

    slope = coeffs[0]

    f_fit = np.logspace(
        np.log10(freq[mask].min()),
        np.log10(freq[mask].max()),
        250
    )

    p_fit = np.exp(coeffs[1]) * f_fit ** slope

    return slope, f_fit, p_fit


# =============================================================
# RUN ANALYSIS
# =============================================================

print("Computing RMSD...")

rmsd_ca = compute_rmsd(u, ca_sel)
rmsd_bb = compute_rmsd(u, backbone_sel)
rmsd_r1 = compute_rmsd(u, region1_sel)
rmsd_r2 = compute_rmsd(u, region2_sel)

joint_rmsd = 0.5 * (rmsd_r1 + rmsd_r2)

print("Computing PSD...")

freq_ca, psd_ca = compute_psd(rmsd_ca, args.dt)
freq_bb, psd_bb = compute_psd(rmsd_bb, args.dt)
freq_r1, psd_r1 = compute_psd(rmsd_r1, args.dt)
freq_r2, psd_r2 = compute_psd(rmsd_r2, args.dt)
freq_joint, psd_joint = compute_psd(joint_rmsd, args.dt)

alpha_ca, f_ca, fit_ca = fit_power_law(freq_ca, psd_ca)
alpha_bb, f_bb, fit_bb = fit_power_law(freq_bb, psd_bb)
alpha_r1, f_r1, fit_r1 = fit_power_law(freq_r1, psd_r1)
alpha_r2, f_r2, fit_r2 = fit_power_law(freq_r2, psd_r2)
alpha_joint, f_joint, fit_joint = fit_power_law(freq_joint, psd_joint)


# =============================================================
# PLOTTING
# =============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# RMSD global
axes[0, 0].plot(rmsd_ca, label="Cα")
axes[0, 0].plot(rmsd_bb, label="Backbone")
axes[0, 0].set_title("Global RMSD")
axes[0, 0].set_xlabel("Frame")
axes[0, 0].set_ylabel("RMSD (Å)")
axes[0, 0].legend()

# RMSD regional
axes[0, 1].plot(rmsd_r1, label="Region 1")
axes[0, 1].plot(rmsd_r2, label="Region 2")
axes[0, 1].plot(joint_rmsd, label="Joint")
axes[0, 1].set_title("Regional RMSD")
axes[0, 1].set_xlabel("Frame")
axes[0, 1].set_ylabel("RMSD (Å)")
axes[0, 1].legend()

# PSD global
axes[1, 0].loglog(freq_ca, psd_ca)
axes[1, 0].loglog(freq_bb, psd_bb)
axes[1, 0].loglog(f_ca, fit_ca, linewidth=2, label=f"Cα α={alpha_ca:.2f}")
axes[1, 0].loglog(f_bb, fit_bb, linewidth=2, label=f"Backbone α={alpha_bb:.2f}")
axes[1, 0].set_title("Global PSD")
axes[1, 0].set_xlabel("Frequency (Hz)")
axes[1, 0].set_ylabel("PSD")
axes[1, 0].legend()

# PSD regional
axes[1, 1].loglog(freq_r1, psd_r1)
axes[1, 1].loglog(freq_r2, psd_r2)
axes[1, 1].loglog(f_r1, fit_r1, linewidth=2, label=f"Region 1 α={alpha_r1:.2f}")
axes[1, 1].loglog(f_r2, fit_r2, linewidth=2, label=f"Region 2 α={alpha_r2:.2f}")
axes[1, 1].loglog(f_joint, fit_joint, linewidth=2,
                  label=f"Joint α={alpha_joint:.2f}")
axes[1, 1].set_title("Regional PSD")
axes[1, 1].set_xlabel("Frequency (Hz)")
axes[1, 1].set_ylabel("PSD")
axes[1, 1].legend()

plt.tight_layout()
plt.show()


# =============================================================
# OUTPUT
# =============================================================

print("\nPower-law exponents:")
print(f"Cα       : {alpha_ca:.3f}")
print(f"Backbone : {alpha_bb:.3f}")
print(f"Region 1 : {alpha_r1:.3f}")
print(f"Region 2 : {alpha_r2:.3f}")
print(f"Joint    : {alpha_joint:.3f}")
```
