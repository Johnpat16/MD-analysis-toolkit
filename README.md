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

## Requirements

The scripts require:

* Python 3.x
* NumPy
* Matplotlib
* SciPy
* MDAnalysis

Install dependencies:

```bash
pip install numpy matplotlib scipy MDAnalysis
```

## Input Files

Supported input formats:

* **Topology:** `.parm7`
* **Trajectory:** `.nc`

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
