"""
moltensaltcalc: A package for simulating and analyzing molten salt properties
"""

from .analyzer import MoltenSaltAnalyzer
from .simulator import MoltenSaltSimulator
from .utils import load_results, plot_comparison, save_results

__version__ = "1.0.0"
__author__ = "Your Name"
__all__ = [
    "MoltenSaltSimulator",
    "MoltenSaltAnalyzer",
    "plot_comparison",
    "save_results",
    "load_results",
]
