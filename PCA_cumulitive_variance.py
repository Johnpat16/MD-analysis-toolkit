```python
#!/usr/bin/env python3

"""
PRINCIPAL COMPONENT ANALYSIS (PCA)
=============================================================

This script performs Principal Component Analysis (PCA)
on molecular dynamics trajectories.

It computes:

1. Trajectory alignment
2. Coordinate extraction
3. Standardization of coordinates
4. Principal components
5. Explained variance ratios
6. Projection of trajectory along PCs

Purpose:
--------
PCA identifies the dominant collective motions
of the protein and reduces dimensionality.

Interpretation:
---------------
Large variance PCs -> dominant structural motions
Clusters in PC space -> metastable conformations
Transitions -> conformational changes

Supported formats:
------------------
Topology:   AMBER (.parm7)
Trajectory: AMBER NetCDF (.nc)

Example:
--------
python pca_analysis.py -p topology.parm7 -t trajectory.nc
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda

from MDAnalysis.analysis import align
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =============================================================
# ARGUMENTS
# =============================================================

parser = argparse.ArgumentParser(
    description="Principal Component Analysis for MD trajectories"
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

print("Loading trajectory...")

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

coords = []

for ts in u.trajectory:
    coords.append(atoms.positions.flatten())

coords = np.array(coords)

print("Coordinate matrix shape:", coords.shape)


# =============================================================
# STANDARDIZATION
# =============================================================

print("Standardizing coordinates...")

scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)


# =============================================================
# PCA
# =============================================================

print("Running PCA...")

pca = PCA()

pcs = pca.fit_transform(coords_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)


# =============================================================
# TIME VECTOR
# =============================================================

time = np.arange(len(pcs))


# =============================================================
# PLOTTING
# =============================================================

fig = plt.figure(figsize=(18, 12))

n_plot = 20


# Explained variance
ax1 = plt.subplot(2, 3, 1)

ax1.bar(
    np.arange(1, n_plot + 1),
    explained_variance[:n_plot]
)

ax1.set_title("Explained Variance Ratio")
ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Variance Ratio")


# Cumulative variance
ax2 = plt.subplot(2, 3, 4)

ax2.plot(
    np.arange(1, n_plot + 1),
    cumulative_variance[:n_plot],
    linewidth=2
)

ax2.axhline(0.80, linestyle='--', label='80%')
ax2.axhline(0.90, linestyle='--', label='90%')

ax2.set_title("Cumulative Explained Variance")
ax2.set_xlabel("Principal Component")
ax2.set_ylabel("Cumulative Variance")
ax2.legend()


# PC1 vs PC2
ax3 = plt.subplot(2, 3, 2)

scatter = ax3.scatter(
    pcs[:, 0],
    pcs[:, 1],
    c=time,
    cmap='viridis',
    s=12
)

ax3.set_xlabel("PC1")
ax3.set_ylabel("PC2")


# PC1 vs PC3
ax4 = plt.subplot(2, 3, 3)

ax4.scatter(
    pcs[:, 0],
    pcs[:, 2],
    c=time,
    cmap='viridis',
    s=12
)

ax4.set_xlabel("PC1")
ax4.set_ylabel("PC3")


# PC2 vs PC3
ax5 = plt.subplot(2, 3, 5)

ax5.scatter(
    pcs[:, 1],
    pcs[:, 2],
    c=time,
    cmap='viridis',
    s=12
)

ax5.set_xlabel("PC2")
ax5.set_ylabel("PC3")


# PC3 vs PC4
ax6 = plt.subplot(2, 3, 6)

ax6.scatter(
    pcs[:, 2],
    pcs[:, 3],
    c=time,
    cmap='viridis',
    s=12
)

ax6.set_xlabel("PC3")
ax6.set_ylabel("PC4")


# Colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

cbar = fig.colorbar(
    scatter,
    cax=cbar_ax
)

cbar.set_label("Trajectory Frame")


# Title
plt.suptitle(
    "Principal Component Analysis",
    fontsize=18,
    fontweight='bold'
)

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()


# =============================================================
# OUTPUT
# =============================================================

print("\n=== Explained Variance Ratio ===")

for i in range(10):
    print(f"PC{i+1}: {explained_variance[i]:.4f}")

print("\n=== Cumulative Variance ===")

for i in [1, 2, 3, 5, 10, 20]:
    print(f"First {i} PCs: {cumulative_variance[i-1]:.4f}")
```
