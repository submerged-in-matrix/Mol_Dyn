# Lennard-Jones 2D Molecular Dynamics (LAMMPS vs Python)

This project reproduces a two-dimensional Lennard-Jones (LJ) system using two independent implementations:
- A custom Python MD code (`lj2d.py`)
- A corresponding LAMMPS input script (`in.lj2d`)

The goal is to verify physical consistency between both simulations—energy conservation, temperature evolution, and structural properties such as the radial distribution function *g(r)*.

---

## 🔧 Simulation Overview

| Property | Value |
|-----------|--------|
| Units | Reduced Lennard-Jones (`ε = σ = 1`) |
| Dimensions | 2D |
| Number of particles | 36 |
| Density (ρ) | 0.8 |
| Cutoff radius | 2.5 σ |
| Integration timestep | 0.005 |
| Initial temperature | 1.0 |
| Ensemble | NVE |

The LAMMPS script reproduces the same setup as the Python code, including the lattice initialization, pair potential parameters, and neighbor list configuration.

---

## 📁 File Summary

| File | Description |
|------|--------------|
| `in.lj2d` | LAMMPS input script for 2D Lennard-Jones simulation |
| `traj_lammps.xyz` | Trajectory from LAMMPS (XYZ dump, readable by Ovito or ASE) |
| `gr_lammps.txt` | Radial distribution function output from LAMMPS |
| `lj2d.py` | Python implementation of the same LJ dynamics |
| `traj_python.xyz` | Trajectory output from Python version |
| `compare_lj_side_by_side.py` | Utility script to animate both runs simultaneously (side-by-side) |

---

## 🧩 Running the Simulations

### LAMMPS
Run from the command line:
```bash
lmp -in in.lj2d
