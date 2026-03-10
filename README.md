# MoltenSaltCalc

A Python package for simulating and analyzing molten salt properties using machine learning potentials (MLPs).

## Author

Max van Brenk

## Features

- Build Systems: Construct molten salt systems with customizable compositions
- MLP Integration: Support for FAIRCHEM, MACE, and GRACE MLPs
- Molecular Dynamics: Run NPT (constant pressure-temperature) and NVT (constant volume-temperature) simulations
- Property Analysis: Compute density, thermal expansion, heat capacity, diffusion coefficients, viscosity, and radial distribution functions
- Visualization: Built-in plotting for analysis of the results

## Installation

### Basic Installation
```bash
git clone https://github.com/leiapple/moltensaltcalc.git
cd moltensaltcalc
uv venv --python 3.12 # optionally add a name (generate the venv in a non shared-drive (e.g. onedrive) folder, it will cause problems later)
.venv/Scripts/activate # for windows (ensure script execution is allowed for remote signed scripts, in case if fails run: "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser")
. .venv/bin/activate # for linux/mac
uv pip install -r requirements.txt
pip install -e .
```

## Project Structure
```
moltensaltcalc/
├── moltensaltcalc/
│   ├── __init__.py          # Package exports
│   ├── simulator.py         # MoltenSaltSimulator class
│   ├── analyzer.py          # MoltenSaltAnalyzer class
│   └── utils.py             # Utility functions
├── demo/
│   ├── TODO
├── tests/
│   ├── TODO
├── pyproject.toml          # Build configuration
└── README.md               # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or feature requests, please open an issue on the [GitHub repository](https://github.com/leiapple/MoltenSaltCalc).
