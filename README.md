<h4 align="center">

![License](https://img.shields.io/badge/license-MIT-blue)
![CI](https://github.com/leiapple/moltensaltcalc/actions/workflows/ci.yml/badge.svg)
![Coverage](badges/coverage.svg)

![Python version](https://img.shields.io/pypi/pyversions/moltensaltcalc)
![PyPI version](https://img.shields.io/pypi/v/moltensaltcalc?label=PyPI)
![PyPI downloads](https://img.shields.io/pypi/dm/moltensaltcalc)

</h4>

# MoltenSaltCalc

A Python package for running and analyzing molecular dynamics (MD) simulations of molten salts using machine-learned interatomic potentials (MLIPs) within the Atomic Simulation Environment (ASE).

## Authors

Daniel Isler, Lei Zhang, Max van Brenk, Süleyman Er

## Features

- System Construction: Construct molten salt systems with customizable compositions in ASE
- MLIP Integration: Support for FAIRCHEM, MACE, GRACE, ... MLIPs (other MLIPs can also be added by the user)
- Molecular Dynamics: Run NPT (constant pressure-temperature) and NVT (constant volume-temperature) simulations
- Property Analysis: Compute thermodynamic and transport properties such as density, diffusion coefficients, viscosity, and heat capacity

## Installation

Create a virtual environment and install the package with the desired MLIP backend. Each MLIP backend has separate and potentially conflicting dependencies. Therefore, only one backend should be installed per environment.

Tested on Python 3.11, 3.12, 3.13 and 3.14. All uMLIPs work on Python 3.12, but some of them do not work on the lower / higher versions. E.g. the fairchem (uma), grace and upet uMLIPs do not work with Python 3.10. On python 3.14, so far only chgnet, mattersim and upet work.

By default, the installation is shipped along with the `torch-dftd3` calculator for long-range interactions. If you do not wish to install/use the dispersion calculator at all, install with `-nodisp` instead, e.g. `pip install moltensaltcalc[grace-nodisp]`. If you want to use the (slower but more accurate) `dftd4` calculator, install the `dftd4` variant, e.g. `pip install moltensaltcalc[grace-nodisp,dftd4]`.

### GRACE

```bash
python3 -m venv .venv        # Or any other name
source .venv/bin/activate   # Linux/macOS
# or
.venv\Scripts\activate      # Windows

pip install moltensaltcalc[grace]
```

### FAIRCHEM

```bash
pip install moltensaltcalc[fairchem]
```

### MACE

```bash
pip install moltensaltcalc[mace]
```

### MatterSim

```bash
pip install moltensaltcalc[mattersim]
```

### 7net

```bash
pip install moltensaltcalc[7net]
```

### Nequip

```bash
pip install moltensaltcalc[nequip]
```

### Nequix

```bash
pip install moltensaltcalc[nequix]
```

### UPET

```bash
pip install moltensaltcalc[upet]
```

### CHGNet

```bash
pip install moltensaltcalc[chgnet]
```

### equiformer_v3

```bash
pip install moltensaltcalc[equiformer_v3]
```

### ORB-V3 (orbitals)

```bash
pip install moltensaltcalc[orbitals]
```

### TACE

```bash
pip install moltensaltcalc[tace]
```

### EquFlash

```bash
pip install moltensaltcalc[equflash]
```

### Development

If you want to contribute or make modifications to the code, clone the repo and install in edit mode. For further details, please check our [contributing guidelines](https://github.com/leiapple/moltensaltcalc/blob/main/CONTRIBUTING.md).

```bash
git clone https://github.com/leiapple/moltensaltcalc.git
cd moltensaltcalc
python3 -m venv .venv        # Or any other name
source .venv/bin/activate   # Linux/macOS
# or
.venv\Scripts\activate      # Windows
pip install -e .[dev,grace]  # Installs the selected MLIP backend and all development dependencies (pytest, etc.) in editable mode
```

## Usage

### Quick start

```bash
pip install moltensaltcalc[grace]
```

```python
import numpy as np

from moltensaltcalc import MoltenSaltSimulator, MoltenSaltAnalyzer

np.random.seed(42)  # Ensure reproducibility (initial random placements)

sim = MoltenSaltSimulator(model_name="GRACE", model_parameters={"model_size": "small", "num_layers": 1, "model_task": "OAM"}, dispersion=None)
atoms = sim.build_system(
    salt_anion=["F", "Cl"],
    salt_cation=["Na"],
    n_anions=[10, 5],  # 10 F atoms and 5 Cl atoms
    n_cations=[15],  # 15 Na atoms
    density_guess=2.0,  # g/cm³
)
sim.run_npt_simulation(
    atoms,
    T=1100,  # K
    steps=1000,  # MD steps
    timestep_fs=1.0,  # fs
    traj_file="npt_simulation.traj",  # Trajectory file
)

analyzer = MoltenSaltAnalyzer(
    traj_files_npt=["npt_simulation.traj"],  # Trajectory file(s)
    temperatures_npt=[1100],  # K
)
density = analyzer.compute_eq_density(T=1100)  # 1.31 g/cm³
C = analyzer.compute_heat_capacity(T=1100, eq_fraction=0.2)  # 0.19 J/g/K
```

### Demo

Run the example notebooks in the `demo/` directory to explore:

- system setup
- running MD simulations
- post-processing and analysis

### Workflow

The workflow of the MoltenSaltCalc aims to provide an optimized environment for molecular dynamics (MD) simulations of molten salts. A typical simulation starts by loading the MLIP backend, done in a lazy manner so the package could also be used without it (e.g. only for analysis or system setup). Next, the system is built starting out from the rocksalt structure, which is different from the usually applied random placements (which can still be used by setting the parameter `lattice` in `build_system` to `"random"`) in order to ensure the absence of clusters of ions with the same charge which typically lead to an initial volume expansion thus requiring a longer volume equilibration simulation. Since the rocksalt contains two atoms per unit cell, but we want to allow an arbitrary number of anions and cations, some random positions are removed from the larger lattice to match the desired composition. The volume of the resulting system is adjusted to match the desired density guess (input variable `density_guess` in g/cm<sup>3</sup>).

Before starting the MD simulation, the velocities are initialized with a Maxwell-Boltzmann distribution according to the desired temperature, while keeping the center of mass and the overall rotation fixed to ensure the temperature is not under-shot because the whole system is moving. Starting out from this, first an NPT (constant particles, pressure, temperature) simulation is run to equilibrate the system volume and obtain the density and thermal expansion of the molten salt (`MoltenSaltAnalyzer`). Then an NVT (constant particles, volume, temperature) simulation is run to obtain more properties such as diffusion, viscosity or heat capacity of the molten salt (`MoltenSaltAnalyzer`). The workflow is illustrated below:

![Workflow](imgs/workflow_diagram.png)

## Project Structure
```
moltensaltcalc/
├── moltensaltcalc/         # Source code
│   ├── __init__.py         # Package exports and available models
│   ├── simulator.py        # MoltenSaltSimulator class
│   ├── analyzer.py         # MoltenSaltAnalyzer class
│   ├── model_discovery.py  # Discovery of available MLIPs
│   ├── model_errors.py     # Error formatting
│   ├── registry.py         # MLIP model registration
|   └── models/             # MLIP model implementations
|       ├── __init__.py
|       ├── 7net.py
|       ├── chgnet.py
|       ├── fairchem.py
|       ├── grace.py
|       ├── mace.py
|       ├── mattersim.py
|       ├── nequip.py
|       ├── nequix.py
|       ├── upet.py
|       ├── equiformer_v3.py
|       ├── orbitals.py
|       ├── tace.py
|       ├── vasp.py
|       └── equflash.py
├── demo/
│   ├── simulator.ipynb     # Demo notebook for the simulator
│   ├── analyzer.ipynb      # Demo notebook for the analyzer
|   └── demo_simulation_results/ # Example trajectory used by the demo
├── tests/                  # PyTests
│   ├── __init__.py
│   ├── test_simulator.py   # Tests for the simulator using the GRACE uMLIP
│   ├── test_analyzer.py    # Tests for the analyzer using the stored trajectories
|   ├── test_uMLIPs.py      # Tests for the different uMLIP backends
│   ├── test_analyzer_trajectories/  # Example trajectories used by the tests
|   └── test_uMLIP_precompiled/  # Precompiled models used by the tests
├── noxfile.py              # Nox configuration for uMLIP testing in different environments
├── pyproject.toml          # Build configuration
├── .gitattributes
├── .gitignore              # Gitignore file: Python template + some custom rules at the end
├── .pre-commit-config.yaml # Pre-commit configuration
├── CITATION.cff            # Citation file
├── CONTRIBUTING.md         # Contributing guidelines
├── LICENSE                 # License file
└── README.md               # This file
```

## License

This project is licensed under the MIT License, see the [LICENSE](https://github.com/leiapple/MoltenSaltCalc/blob/main/LICENSE) file for details.

## Support

For questions, bug reports, or feature requests, please open an issue on [GitHub](https://github.com/leiapple/MoltenSaltCalc/issues).
