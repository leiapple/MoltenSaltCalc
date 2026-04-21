from typing import List

from .analyzer import MoltenSaltAnalyzer as MoltenSaltAnalyzer
from .simulator import MoltenSaltSimulator as MoltenSaltSimulator

__all__: list[str]

__version__: str
__author__: str

def available_models() -> List[str]: ...
