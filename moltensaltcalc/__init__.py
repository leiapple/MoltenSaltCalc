"""
moltensaltcalc: A package for simulating and analyzing molten salt properties
"""

from .simulator import MoltenSaltSimulator
from .analyzer import MoltenSaltAnalyzer
from .utils import plot_comparison, save_results, load_results

__version__ = "1.0.0"
__author__ = "Your Name"
__all__ = [
    "MoltenSaltSimulator", 
    "MoltenSaltAnalyzer", 
    "plot_comparison", 
    "save_results", 
    "load_results"
]