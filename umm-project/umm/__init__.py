"""
Universal Measurement Machine (UMM)

AI-programmed adaptive quantum measurement framework.
"""

__version__ = "0.1.0"
__author__ = "Justin Hart"

from .core import QuantumState, UMMSimulator
from .intent import ObjectiveParser

__all__ = [
    "QuantumState",
    "UMMSimulator",
    "ObjectiveParser",
]
