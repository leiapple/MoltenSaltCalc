# MoltenSaltCalc

A Python package for simulating and analyzing molten salt properties using machine learning interatomic potentials (MLIPs) for the energy and force predictions in molecular dynamics (MD) simulations from atomic simulation environments (ASE).

## Authors

Daniel Isler, Lei Zhang, Max van Brenk, Süleyman Er

## Features

- Build Systems: Construct molten salt systems with customizable compositions
- MLP Integration: Support for FAIRCHEM, MACE, and GRACE MLPs
- Molecular Dynamics: Run NPT (constant pressure-temperature) and NVT (constant volume-temperature) simulations
- Property Analysis: Compute density, thermal expansion, heat capacity, diffusion coefficients, viscosity, and radial distribution functions
- Visualization: Built-in plotting for analysis of the results

## Installation

### Basic Installation

This only needs to be done once and installs the package and its dependencies in virtual environment.

```bash
git clone https://github.com/leiapple/moltensaltcalc.git
cd moltensaltcalc
uv venv --python 3.12 # optionally add a name (do not generate the venv in on a shared-drive (e.g. onedrive), it will cause problems later)
# If uv is not installed, it can be installed with: pip install uv or brev install uv, etc.
.venv/Scripts/activate # for windows (ensure script execution is allowed for remote signed scripts, in case if fails run: "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser")
. .venv/bin/activate # for linux/mac
pip install -e . "[grace]" # Depending on the MLIP you'd like to use, choose from "[grace]", "[fairchem]", "[mace]" (only one per venv, as they have conflicting dependencies)
```

## Usage

See the [demo notebooks](./demo/) for usage examples of the simulator and analyzer classes.

## Project Structure
```
moltensaltcalc/
├── moltensaltcalc/          # Source code
│   ├── __init__.py          # Package exports and available models
│   ├── simulator.py         # MoltenSaltSimulator class
│   ├── analyzer.py          # MoltenSaltAnalyzer class
│   ├── model_discovery.py   # Discovery of available MLIPs
│   ├── model_errors.py      # Error formatting
│   ├── registry.py          # Model registration
|   └── models/              # Model implementations
|       ├── __init__.py
|       ├── grace.py
|       ├── fairchem.py
|       └── mace.py
├── demo/
│   ├── simulator.ipynb      # Demo notebook for the simulator
│   ├── analyzer.ipynb       # Demo notebook for the analyzer
|   └── demo_simulation_results/ # Example trajectory used by the demo
├── tests/
│   ├── __init__.py
│   ├── test_simulator.py
│   └── test_analyzer.py
├── setup.py                # Setup configuration
├── pyproject.toml          # Build configuration
├── requirements_*.txt      # Exact dependency specifications that were tested
├── LICENSE                 # License file
└── README.md               # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or feature requests, please open an issue on the [GitHub repository](https://github.com/leiapple/MoltenSaltCalc).

## Reference

If you use this package in your research, please cite the following paper: [TODO Link](https://todo.com)

TODO Add bibtex

## Acknowledgements

TODO: Same as in Paper the funding?
