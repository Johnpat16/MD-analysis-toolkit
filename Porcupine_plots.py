```python
#!/usr/bin/env python3

"""
PORCUPINE PLOT GENERATOR
=============================================================

This script generates porcupine plots from PCA eigenvectors
of molecular dynamics trajectories.

Porcupine plots visualize collective motions by drawing
arrows on the average structure.

Arrow meaning:
--------------
Direction -> motion direction
Length    -> motion magnitude

Purpose:
--------
Useful for identifying:

- domain opening/closing
- hinge motions
- ligand-induced motions
- collective structural transitions

Outputs:
--------
- reference_structure.pdb
- pc1_porcupine.tcl
- pc2_porcupine.tcl
- ...
- pca_projection.npy

Supported formats:
------------------
Topology:   AMBER (.parm7)
Trajectory: AMBER NetCDF (.nc)

Example:
--------
python porcupine_plot.py -p topology.parm7 -t trajectory.nc
"""

import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
from sklearn.decomposition import PCA


# =============================================================
# ARGUMENTS
# =============================================================

parser = argparse.ArgumentParser(
    description="Generate porcupine plots from MD trajectories"
)

parser.add_argument("-p", "--topology", required=True,
                    help="Topology file (.parm7)")

parser.add_argument("-t", "--trajectory", required=True,
                    help="Trajectory file (.nc)")

parser.add_argument("-s", "--selection",
                    default="protein and name CA",
                    help="Atom selection (default: protein and name CA)")

parser.add_argument("-n", "--n_components", type=int,
                    default=4,
                    help="Number of principal components (default: 4)")

args = parser.parse_args()


# =============================================================
# LOAD SYSTEM
# =============================================================

print("\nLoading system...")

u = mda.Universe(args.topology, args.trajectory)
atoms = u.select_atoms(args.selection)

print(f"Selected atoms : {len(atoms)}")
print(f"Frames         : {len(u.trajectory)}")


# =============================================================
# ALIGNMENT
# =============================================================

print("\nAligning trajectory...")

align.AlignTraj(
    u,
    u,
    select=args.selection,
    in_memory=True
).run()

print("Alignment complete.")


# =============================================================
# EXTRACT COORDINATES
# =============================================================

print("\nExtracting coordinates...")

coords = np.array([
    atoms.positions.copy().flatten()
    for ts in u.trajectory
])

mean_coords = coords.mean(axis=0)
X = coords - mean_coords


# =============================================================
# PCA
# =============================================================

print("\nRunning PCA...")

pca = PCA(n_components=args.n_components)
projection = pca.fit_transform(X)

eigenvalues = pca.explained_variance_
eigenvectors = pca.components_

print("\nExplained variance ratio:")

for i, v in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {v:.4f}")


# =============================================================
# REBUILD PHYSICAL MODES
# =============================================================

mean_atoms = mean_coords.reshape(len(atoms), 3)

def build_mode(k):

    vec = eigenvectors[k].reshape(len(atoms), 3)

    # physical amplitude scaling
    vec *= np.sqrt(eigenvalues[k])

    return vec


pc_modes = [build_mode(k) for k in range(args.n_components)]


# =============================================================
# WRITE REFERENCE STRUCTURE
# =============================================================

print("\nWriting reference structure...")

protein = u.select_atoms("protein")
protein.write("reference_structure.pdb")


# =============================================================
# PORCUPINE WRITER
# =============================================================

def write_porcupine(mode, filename, color):

    print(f"Writing {filename}")

    max_norm = np.max(np.linalg.norm(mode, axis=1))
    global_scale = 15.0 / max_norm

    with open(filename, "w") as f:

        f.write("draw delete all\n")
        f.write("draw material Opaque\n")
        f.write(f"draw color {color}\n")

        for i in range(len(atoms)):

            start = mean_atoms[i]
            vec = mode[i]

            end = start + vec * global_scale

            sx, sy, sz = start
            ex, ey, ez = end

            f.write(
                f"draw cylinder "
                f"{{{sx:.3f} {sy:.3f} {sz:.3f}}} "
                f"{{{ex:.3f} {ey:.3f} {ez:.3f}}} "
                f"radius 0.35\n"
            )

            direction = vec / (np.linalg.norm(vec) + 1e-12)
            cone_end = end + direction * 1.5

            cx, cy, cz = cone_end

            f.write(
                f"draw cone "
                f"{{{ex:.3f} {ey:.3f} {ez:.3f}}} "
                f"{{{cx:.3f} {cy:.3f} {cz:.3f}}} "
                f"radius 0.7\n"
            )


# =============================================================
# GENERATE MODES
# =============================================================

colors = ["red", "blue", "green", "yellow", "orange", "purple"]

for i, mode in enumerate(pc_modes):

    filename = f"pc{i+1}_porcupine.tcl"
    color = colors[i % len(colors)]

    write_porcupine(mode, filename, color)


# =============================================================
# SAVE PROJECTIONS
# =============================================================

np.save("pca_projection.npy", projection)

print("\nDONE.")
```
