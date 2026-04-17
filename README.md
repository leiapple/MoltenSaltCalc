<h4 align="center">

<!-- TODO Add PIP? -->
<!-- ![PyPI - Version](https://img.shields.io/pypi/v/fairchem-core) -->
![Static Badge](https://img.shields.io/badge/python-3.12%2B-blue)
![Coverage](https://img.shields.io/badge/coverage-91%25-green)
![License](https://img.shields.io/badge/license-MIT-blue)
<!-- TODO: Add DOI? -->

</h4>

# MoltenSaltCalc

A Python package for running and analyzing molecular dynamics (MD) simulations of molten salts using machine-learned interatomic potentials (MLIPs) within the Atomic Simulation Environment (ASE).

## Authors

Daniel Isler, Lei Zhang, Max van Brenk, Süleyman Er

## Features

- System Construction: Construct molten salt systems with customizable compositions in ASE
- MLIP Integration: Support for FAIRCHEM, MACE, and GRACE MLIPs
- Molecular Dynamics: Run NPT (constant pressure-temperature) and NVT (constant volume-temperature) simulations
- Property Analysis: Compute thermodynamic and transport properties such as density, diffusion coefficients, viscosity, and heat capacity

## Installation

Create a virtual environment and install the package with the desired MLIP backend. Each MLIP backend has separate and potentially conflicting dependencies. Therefore, only one backend should be installed per environment.

Tested on Python 3.10, 3.11, 3.12. Python 3.13+ is not yet supported due to upstream dependencies (e.g. tensorflow required by the GRACE (tensorpotential) uMLIP).

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

sim = MoltenSaltSimulator(model_name="GRACE", model_parameters={"model_size": "small", "num_layers": 1, "model_task": "OAM"})
atoms = sim.build_system(
    salt_anion=["F", "Cl"],
    salt_cation=["Na"],
    anion_Natoms=[10, 5],  # 7 F atoms and 5 Cl atoms
    cation_Natoms=[15],  # 15 Na atoms
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

## Project Structure
```
moltensaltcalc/
├── moltensaltcalc/         # Source code
│   ├── __init__.py         # Package exports and available models
│   ├── simulator.py        # MoltenSaltSimulator class
│   ├── analyzer.py         # MoltenSaltAnalyzer class
│   ├── model_discovery.py  # Discovery of available MLIPs
│   ├── model_errors.py     # Error formatting
│   ├── registry.py         # Model registration
|   └── models/             # Model implementations
|       ├── __init__.py
|       ├── grace.py
|       ├── fairchem.py
|       └── mace.py
├── demo/
│   ├── simulator.ipynb     # Demo notebook for the simulator
│   ├── analyzer.ipynb      # Demo notebook for the analyzer
|   └── demo_simulation_results/ # Example trajectory used by the demo
├── tests/                  # PyTests
│   ├── __init__.py
│   ├── test_simulator.py
│   └── test_analyzer.py
├── pyproject.toml          # Build configuration
├── requirements_*.txt      # These files contain exact dependency snapshots used during testing for each MLIP backend.
├── .gitattributes
├── .gitignore              # Gitignore file: Python template + some custom rules at the end
├── .pre-commit-config.yaml # Pre-commit configuration
├── CITATION.cff            # Citation file
├── CONTRIBUTING.md         # Contributing guidelines
├── LICENSE                 # License file
└── README.md               # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, bug reports, or feature requests, please open an issue on [GitHub](https://github.com/leiapple/MoltenSaltCalc/issues).
