```python id="fel_py_01"
#!/usr/bin/env python3

"""
FREE ENERGY LANDSCAPE (FEL) FROM PCA SPACE
=============================================================

This script computes Free Energy Landscapes (FEL) from
molecular dynamics trajectories using Principal Component
Analysis (PCA) as a reduced dimensional space.

Method:
-------
1. Align all trajectories to a reference structure
2. Concatenate datasets
3. Perform global PCA
4. Project each system into shared PC1-PC2 space
5. Estimate probability density via KDE
6. Convert to free energy:

   F = -kBT ln(P)

Purpose:
--------
FEL reveals:

- conformational basins
- metastable states
- ligand-induced shifts
- energy barriers between states

Important:
----------
All systems are projected into a SINGLE PCA space to allow
direct comparison.

Supported formats:
------------------
Topology:   AMBER (.parm7)
Trajectory: AMBER NetCDF (.nc)

Example:
--------
python fel_3d.py -systems config.json
"""

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import align
from scipy.stats import gaussian_kde


# =============================================================
# CONSTANTS
# =============================================================

T = 298.0
kB = 0.0019872041
kBT = kB * T

selection = "protein and name CA"


# =============================================================
# SYSTEMS (EDIT HERE OR MOVE TO CONFIG FILE)
# =============================================================

systems = {
    "Apo": ("topology.parm7", "trajectory.nc"),
    "Ligand": ("ligand.parm7", "ligand.nc"),
    "State3": ("state3.parm7", "state3.nc")
}


# =============================================================
# REFERENCE SYSTEM (IMPORTANT CHOICE)
# =============================================================

ref_top, ref_traj = systems["Apo"]

print("Loading reference system...")

ref = mda.Universe(ref_top, ref_traj)


# =============================================================
# ALIGNMENT FUNCTION
# =============================================================

def load_coords_aligned(universe, reference):

    protein = universe.select_atoms(selection)

    align.AlignTraj(
        universe,
        reference,
        select=selection,
        in_memory=True
    ).run()

    coords = np.array([
        protein.positions.copy()
        for ts in universe.trajectory
    ])

    return coords


# =============================================================
# LOAD ALL SYSTEMS
# =============================================================

print("Loading trajectories...")

traj_data = {}
all_data = []

for name, (top, traj) in systems.items():

    u = mda.Universe(top, traj)
    coords = load_coords_aligned(u, ref)

    traj_data[name] = coords
    all_data.append(coords)

all_data = np.concatenate(all_data, axis=0)


# =============================================================
# GLOBAL PCA (SHARED SPACE)
# =============================================================

print("Computing global PCA...")

mean = all_data.mean(axis=0)

centered = []

for name in systems:
    centered.append(traj_data[name] - mean)

centered = np.concatenate(centered, axis=0)

X = centered.reshape(centered.shape[0], -1)

cov = np.cov(X.T)

eigvals, eigvecs = np.linalg.eigh(cov)

eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]

PCs = X @ eigvecs[:, :2]


# =============================================================
# SPLIT PROJECTIONS
# =============================================================

pc_data = {}
start = 0

for name in systems:

    n = len(traj_data[name])
    pc_data[name] = PCs[start:start+n]
    start += n


# =============================================================
# SHARED GRID LIMITS
# =============================================================

pc1_all = PCs[:, 0]
pc2_all = PCs[:, 1]

pc1_min, pc1_max = pc1_all.min(), pc1_all.max()
pc2_min, pc2_max = pc2_all.min(), pc2_all.max()


# =============================================================
# FREE ENERGY FUNCTION
# =============================================================

def compute_fes(pc1, pc2, grid=120):

    xy = np.vstack([pc1, pc2])
    kde = gaussian_kde(xy)

    x = np.linspace(pc1_min, pc1_max, grid)
    y = np.linspace(pc2_min, pc2_max, grid)

    Xg, Yg = np.meshgrid(x, y)

    Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

    Z = np.clip(Z, 1e-15, None)

    F = -kBT * np.log(Z)
    F -= np.nanmin(F)

    F[F > 4.0] = np.nan

    return Xg, Yg, F


# =============================================================
# PLOTTING
# =============================================================

print("Plotting FELs...")

fig = plt.figure(figsize=(18, 10))

for i, (name, data) in enumerate(pc_data.items(), 1):

    pc1 = data[:, 0]
    pc2 = data[:, 1]

    Xg, Yg, F = compute_fes(pc1, pc2)

    ax = fig.add_subplot(2, 3, i, projection='3d')

    surf = ax.plot_surface(
        Xg, Yg, F,
        cmap='plasma',
        edgecolor='none',
        alpha=0.95
    )

    ax.set_title(name)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("Free Energy (kcal/mol)")

    ax.view_init(elev=35, azim=45)


# =============================================================
# COLORBAR
# =============================================================

cbar = fig.colorbar(surf, shrink=0.6, pad=0.1)
cbar.set_label("Free Energy (kcal/mol)")


plt.tight_layout()
plt.show()
```
