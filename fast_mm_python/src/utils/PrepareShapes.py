"""
Given a level, list all shapes (i, j, k) where i + j + k = 2^level.
"""

import numpy as np


def PrepareShapes(level):
    """
    Given a level, list all shapes (i, j, k) where i + j + k = 2^level.
    
    Args:
        level: Level value
    
    Returns:
        shapes: Array of shape (n, 3) containing all valid triples
    """
    shapes = []
    sum_col = 2 ** level
    
    for i in range(sum_col + 1):
        for j in range(sum_col - i + 1):
            k = sum_col - i - j
            shapes.append([i, j, k])
    
    return np.array(shapes, dtype=int)

