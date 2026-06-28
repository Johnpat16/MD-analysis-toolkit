# HSA Molecular Dynamics Analysis Toolkit

This repository contains analysis scripts used for the molecular dynamics study of **Human Serum Albumin (HSA)** in apo and ligand-bound systems for different % cHSA. The scripts are designed for trajectory analysis of **AMBER-generated NetCDF (`.nc`) trajectories** and topology files (`.parm7`).

The project focuses on understanding structural flexibility, correlated motions, conformational transitions, and energetic landscapes associated with ligand binding in different HSA binding sites.

## Systems Analyzed

The repository includes analysis workflows for:

* **Apo HSA**
* **25%,50%,75% and 100% HSA**
* **Sudlow Site I-bound HSA**
* **Sudlow Site II-bound HSA**
* **Sudlow Site I + II-bound HSA**
* **Oleic acid-bound HSA**

## Analysis Methods

Implemented analyses include:

### Structural Stability

* **RMSD (Root Mean Square Deviation)**
* **FFT and PS ( Fast Fourier Transform and Power Spectrum)**
* **RMSF (Root Mean Square Fluctuation)**

### Dynamic Correlation

* **DCCM (Dynamic Cross-Correlation Matrix)**
* **RMSD Autocorrelation Analysis**

### Conformational Space Exploration

* **Principal Component Analysis (PCA)**
* **Porcupine Plots (Essential Dynamics Visualization)**

### Free Energy Landscape

* **2D Free Energy Landscapes (FEL)**
* **3D Free Energy Landscapes**

## Installation

### Option 1: virtual environment

```bash
python3 -m venv md_env
source md_env/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the same environment with:

```powershell
.\md_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option 2: requirements file

If you already have a Python environment, install the dependencies directly:

```bash
pip install -r requirements.txt
```

The requirements file includes:

- numpy
- scipy
- matplotlib
- pandas
- scikit-learn
- MDAnalysis

## Input Files

The scripts are written for AMBER-format inputs:

- `-p`, `--topology`: topology file, usually `.parm7`
- `-t`, `--trajectory`: trajectory file, usually `.nc`
- `-s`, `--selection`: optional MDAnalysis atom selection

The default atom selection is:

```text
protein and name CA
```

Use a different selection when the scientific question requires it, for example:

```bash
python RMSF/rmsf_analysis.py -p topology.parm7 -t trajectory.nc -s "protein and backbone"
```

## How to Run

Run each analysis from the repository root and pass the topology and trajectory explicitly:

```bash
python RMSF/rmsf_analysis.py -p topology.parm7 -t trajectory.nc
python RMSD/rmsd_analysis.py -p topology.parm7 -t trajectory.nc
python DCCM/dccm_analysis.py -p topology.parm7 -t trajectory.nc
python PCA/pca_analysis.py -p topology.parm7 -t trajectory.nc
python Porcupine/porcupine_plot.py -p topology.parm7 -t trajectory.nc
python FEL/fel_3d.py -p topology.parm7 -t trajectory.nc
python Autocorrelation/rmsd_autocorrelation.py -p topology.parm7 -t trajectory.nc
```

Common options:

```text
-p / --topology      AMBER topology file (.parm7)
-t / --trajectory    AMBER NetCDF trajectory file (.nc)
-s / --selection     Optional atom selection; default is "protein and name CA"
-o / --output-prefix Prefix for output files
--no-show            Save figures without opening an interactive plot window
```

RMSD and autocorrelation scripts also accept `--dt`, the time between frames in seconds. FEL accepts `--temperature` and `--grid`. PCA and porcupine scripts accept `-n / --n-components`.

## Repository Structure

```text
scripts/
├── RMSD/
├── RMSF/
├── DCCM/
├── Autocorrelation/
├── PCA/
├── Porcupine/
└── FEL/
```

## Purpose

This repository was created to provide full computational reproducibility for the molecular dynamics analyses performed in my internship and to serve as an open resource for researchers working on protein dynamics and ligand-induced conformational changes.

## Citation

If you use these scripts, please cite:

Ioannis Patelaros.
*Molecular Dynamics Investigation of Human Serum Albumin Ligand Binding and Conformational Dynamics*.
University of Bologna, 2026.

## License

MIT License
