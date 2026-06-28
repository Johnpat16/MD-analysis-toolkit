```python id="rmsf_py_01"
#!/usr/bin/env python3

"""
RMSF ANALYSIS FROM MOLECULAR DYNAMICS TRAJECTORIES
=============================================================

This script computes the Root Mean Square Fluctuation (RMSF)
for protein residues from molecular dynamics simulations.

RMSF measures the time-averaged positional flexibility of atoms.

Definition:
-----------
RMSF(i) = sqrt( < |r_i(t) - <r_i>|^2 > )

where:
- r_i(t)   = position of atom i at time t
- <r_i>    = time-averaged position
- < >      = time average over trajectory

Purpose:
--------
RMSF identifies:

- flexible loops
- rigid structural cores
- binding pocket dynamics
- ligand-induced stabilization/destabilization

Supported formats:
------------------
Topology:   AMBER (.parm7)
Trajectory: AMBER NetCDF (.nc)

Example:
--------
python rmsf_analysis.py -p topology.parm7 -t trajectory.nc
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
    description="RMSF analysis for MD trajectories"
)

parser.add_argument("-p", "--topology", required=True,
                    help="Topology file (.parm7)")

parser.add_argument("-t", "--trajectory", required=True,
                    help="Trajectory file (.nc)")

parser.add_argument("-s", "--selection",
                    default="protein and name CA",
                    help="Atom selection (default: Cα atoms)")

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
    select="protein and backbone",
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
# RMSF CALCULATION
# =============================================================

mean = coords.mean(axis=0)
fluctuations = coords - mean

rmsf = np.sqrt(
    np.mean(np.sum(fluctuations**2, axis=2), axis=0)
)


# =============================================================
# PLOT
# =============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    atoms.resids,
    rmsf,
    color="black",
    linewidth=2
)

plt.fill_between(
    atoms.resids,
    rmsf,
    alpha=0.2,
    color="black"
)

plt.xlabel("Residue Index")
plt.ylabel("RMSF (Å)")
plt.title("RMSF Profile from MD Simulation")

plt.tight_layout()
plt.show()


# =============================================================
# OUTPUT
# =============================================================

print("\nDone.")
print(f"Max RMSF: {np.max(rmsf):.3f} Å")
print(f"Min RMSF: {np.min(rmsf):.3f} Å")
```
