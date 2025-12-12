# MoltenSaltCalc

A Python package for simulating and analyzing molten salt properties using machine learning potentials.

## Features

- Build Systems: Construct molten salt systems with customizable compositions
- ML Potential Integration: Support for FAIRCHEM, MACE, and GRACE machine learning potentials
- Molecular Dynamics: Run NPT (constant pressure-temperature) and NVT (constant volume-temperature) simulations
- Property Analysis: Compute density, thermal expansion, heat capacity, diffusion coefficients, viscosity, and radial distribution functions
- Visualization: Built-in plotting for analysis results

## Installation

### Basic Installation
`
git clone https://github.com/yourusername/moltensaltcalc.git
cd moltensaltcalc
pip install -e .
`
### Requirements

* Python >= 3.8
* ASE (Atomic Simulation Environment)
* NumPy
* SciPy
* Matplotlib

## Project tructure

moltensaltcalc/
├── src/
│   ├── __init__.py          # Package exports
│   ├── simulator.py         # MoltenSaltSimulator class
│   ├── analyzer.py          # MoltenSaltAnalyzer class
│   └── utils.py             # Utility functions
├── pyproject.toml          # Build configuration
└── README.md               # This file


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or feature requests, please open an issue on the [GitHub repository](https://github.com/leiapple/MoltenSaltCalc).