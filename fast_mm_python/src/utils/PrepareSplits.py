"""
Given a shape (i, j, k), list all methods to split it into two level-(level-1) shapes.
"""

import numpy as np


def PrepareSplits(shape):
    """
    Given a shape (i, j, k) where i + j + k = 2^level,
    list all methods to split (i, j, k) into two level-(level-1) shapes.
    
    Args:
        shape: Tuple or array (i, j, k)
    
    Returns:
        splits: Array of shape (n, 6) where each row is [i1, j1, k1, i2, j2, k2]
    """
    splits = []
    shape = np.atleast_1d(shape)
    sum_col = np.sum(shape)
    sum_half = sum_col // 2
    
    for i in range(min(sum_half, shape[0]) + 1):
        for j in range(min(sum_half - i, shape[1]) + 1):
            k = sum_half - i - j
            if k > shape[2]:
                continue
            splits.append([i, j, k, shape[0] - i, shape[1] - j, shape[2] - k])
    
    return np.array(splits, dtype=int)

