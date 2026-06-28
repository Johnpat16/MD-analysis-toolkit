```python
#!/usr/bin/env python3

"""
DYNAMIC CROSS-CORRELATION MATRIX (DCCM)
=============================================================

This script computes the Dynamic Cross-Correlation Matrix (DCCM)
for molecular dynamics trajectories.

It measures correlated and anti-correlated residue motions.

Correlation scale:
------------------
+1  -> fully correlated motion
 0  -> uncorrelated motion
-1  -> fully anti-correlated motion

Purpose:
--------
DCCM helps identify:

- cooperative domain motions
- allosteric communication
- ligand-induced perturbations
- long-range coupling

Supported formats:
------------------
Topology:   AMBER (.parm7)
Trajectory: AMBER NetCDF (.nc)

Example:
--------
python dccm_analysis.py -p topology.parm7 -t trajectory.nc
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import align


# =============================================================
# ARGUMENTS
# =============================================================

parser = argparse.ArgumentParser(
    description="Dynamic Cross-Correlation Matrix analysis"
)

parser.add_argument("-p", "--topology", required=True,
                    help="Topology file (.parm7)")

parser.add_argument("-t", "--trajectory", required=True,
                    help="Trajectory file (.nc)")

parser.add_argument("-s", "--selection",
                    default="protein and name CA",
                    help="Atom selection (default: protein and name CA)")

args = parser.parse_args()


# =============================================================
# LOAD SYSTEM
# =============================================================

print("Loading system...")

u = mda.Universe(args.topology, args.trajectory)
atoms = u.select_atoms(args.selection)


# =============================================================
# ALIGN TRAJECTORY
# =============================================================

print("Aligning trajectory...")

align.AlignTraj(
    u,
    u,
    select=args.selection,
    in_memory=True
).run()


# =============================================================
# EXTRACT COORDINATES
# =============================================================

print("Extracting coordinates...")

coords = np.array([
    atoms.positions.copy()
    for ts in u.trajectory
])


# =============================================================
# COMPUTE FLUCTUATIONS
# =============================================================

mean_coords = coords.mean(axis=0)
fluctuations = coords - mean_coords

n_res = fluctuations.shape[1]

dccm = np.zeros((n_res, n_res))


# =============================================================
# COMPUTE DCCM
# =============================================================

print("Computing DCCM...")

for i in range(n_res):

    fi = fluctuations[:, i, :]
    denom_i = np.mean(np.sum(fi * fi, axis=1))

    for j in range(n_res):

        fj = fluctuations[:, j, :]
        denom_j = np.mean(np.sum(fj * fj, axis=1))

        numerator = np.mean(np.sum(fi * fj, axis=1))

        dccm[i, j] = numerator / np.sqrt(denom_i * denom_j)


# =============================================================
# PLOT
# =============================================================

plt.figure(figsize=(10, 8))

im = plt.imshow(
    dccm,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    origin="lower",
    interpolation="nearest"
)

plt.xlabel("Residue Index")
plt.ylabel("Residue Index")
plt.title("Dynamic Cross-Correlation Matrix")

plt.colorbar(im, label="Correlation")

plt.tight_layout()
plt.show()


print("\nDone.")
```
