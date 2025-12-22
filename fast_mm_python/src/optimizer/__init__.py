"""
Optimization routines using SciPy.

Note: This is a simplified implementation. The original MATLAB code uses SNOPT,
which is a commercial optimizer. This Python version uses SciPy's optimizers instead.
"""

from .SnObjective import SnObjective
from .SnNonlcon import SnNonlcon
from .SnContinue import SnContinue

__all__ = ['SnObjective', 'SnNonlcon', 'SnContinue']

