# MoltenSaltCalc

MoltenSaltCalc is a Python package for molecular dynamics simulations of molten salts using the Atomic Simulation Environment (ASE) and machine-learned interatomic potentials (MLIPs).

It provides a unified interface to multiple MLIP backends, enabling rapid setup and analysis of molten salt simulations.

---

## Features

- Molecular dynamics simulations built on ASE
- Unified interface for multiple MLIP backends (GRACE, FairChem, MACE)
- Easy switching between pretrained models
- Tools for simulation analysis

---

## Installation

Install the package together with the desired MLIP backend:

```bash
pip install moltensaltcalc[grace]
```

```bash
pip install moltensaltcalc[fairchem]
pip install moltensaltcalc[mace]
```

Note: MLIP backends may have conflicting dependencies. It is recommended to use separate environments for each backend.

---

## Quick start

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

---

## Examples

See example notebooks in the repository:

- [Simulator demo](https://github.com/leiapple/MoltenSaltCalc/blob/main/demo/simulator.ipynb)
- [Analyzer demo](https://github.com/leiapple/MoltenSaltCalc/blob/main/demo/analyzer.ipynb)

## Documentation

Use the navigation bar to explore:

- **API references**
    - [Analyzer API](Analyzer_API.md)
    - [Simulator API](Simulator_API.md)
- Available models
- Simulation and analysis tools

## Repository

The source code is available on [GitHub](https://github.com/leiapple/MoltenSaltCalc).
