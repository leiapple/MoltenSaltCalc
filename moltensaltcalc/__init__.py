"""
moltensaltcalc: A package for simulating and analyzing molten salt properties employing universal machine-learning interatomic potentials (uMLIPs) and atomic simulation environment (ASE).

Main entry points:
- MoltenSaltSimulator: Build systems and run molecular dynamics simulations
- MoltenSaltAnalyzer: Analyze trajectories and calculate properties

The package supports multiple MLIP foundation models (GRACE, MACE, FAIRCHEM) with lazy loading to minimize dependencies and startup time.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("moltensaltcalc")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback for local/dev installs

__author__ = "Daniel Isler, Lei Zhang, Max van Brenk, Suleyman Er"


# Lazy import for heavy modules
def __getattr__(name: str):
    if name == "MoltenSaltSimulator":
        try:
            return import_module(".simulator", __name__).MoltenSaltSimulator
        except ImportError as e:
            raise ImportError(
                "MoltenSaltSimulator requires additional dependencies.\nInstall with: pip install moltensaltcalc[grace|mace|fairchem]"
            ) from e

    if name == "MoltenSaltAnalyzer":
        return import_module(".analyzer", __name__).MoltenSaltAnalyzer

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Directly expose the lightweight available models function
def available_models():
    """Return available model names without importing them."""
    from .model_discovery import discover_models

    return discover_models()


# Public API
__all__ = [
    "MoltenSaltSimulator",
    "MoltenSaltAnalyzer",
    "available_models",
]
