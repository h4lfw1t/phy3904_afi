"""
AFI - Acoustic Field Imaging
A Python package for processing and visualizing acoustic field data.
"""

from .data import AcousticFieldData
from .visualization import AcousticFieldVisualizer

__version__ = "0.1.0"
__all__ = ["AcousticFieldData", "AcousticFieldVisualizer"]