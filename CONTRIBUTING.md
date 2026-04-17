# Contributing to MoltenSaltCalc

Thank you for your interest in contributing! This project aims to provide tools for molecular dynamics simulations of molten salts using ASE and machine-learned interatomic potentials (MLIPs)

We welcome contributions of all kinds: bug reports, feature requests, documentation improvements, and code

## Getting Started

You can contribute either via a fork (recommended for external contributors) or by creating a branch directly in the repository (if you have write access)

### Option 1: Fork (recommended)

Fork the repository and clone your fork and add our git to stay up to date with us:

```bash
git clone https://github.com/<your-username>/MoltenSaltCalc.git
git remote add upstream https://github.com/leiapple/MoltenSaltCalc.git
git fetch upstream
cd MoltenSaltCalc
```

Create a new branch from main:

```bash
git checkout -b my-feature-branch
```

Install the package with development dependencies and the MLIP framework you intend to use (in separate virtual environments from others) in editable mode:

```bash
pip install -e .[dev,grace]
```

Install pre-commit hooks:

```bash
pre-commit install
```

### Option 2: Direct branch (for maintainers)

```bash
git checkout -b my-feature-branch
```

## Development Guidelines

- Code style: We use `black` and `isort` for formatting
- Linting & checks: Managed via `pre-commit`
- Testing: Use `pytest`

Before submitting a pull request, please ensure:

```bash
pytest  # all tests should pass in a venv with .[dev,grace] installed
pre-commit run --all-files
```

## Pull Requests

We actively welcome pull requests

To contribute:

1. Ensure your branch is up to date with main
2. Add tests for new functionality. Check with `pytest-cov` that the main parts are covered
3. Update documentation if you change APIs or behavior
4. Ensure all tests pass
5. Ensure code formatting and linting pass

Please write clear commit messages and provide a concise description of your changes in the PR.\

## Issues

We use GitHub Issues to track bugs and feature requests

When reporting a bug, please include:

- A clear description of the problem
- Steps to reproduce
- Specific versions the installed modules (`pip freeze`)
- Expected vs actual behavior
- Relevant logs, error messages or data files

When requesting a feature, please include:


## License

By contributing, you agree that your contributions will be licensed under the MIT License.
