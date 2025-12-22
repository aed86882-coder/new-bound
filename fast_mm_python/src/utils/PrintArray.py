"""
Printing utilities for debugging.
"""

import numpy as np


def PrintArray(a, name=None):
    """
    Print an array to the screen (for debugging).
    
    Args:
        a: Array to print
        name: Optional name to display
    """
    if name is not None:
        print(f"{name}: ", end="")
    
    a = np.atleast_1d(a)
    for val in a:
        print(f"{val:.10f} ", end="")
    print()

