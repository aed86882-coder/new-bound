"""
Generate transformation matrices to convert joint distribution to marginal distributions.
"""

import numpy as np
from scipy import sparse


def JointToMargin(column, sum_col):
    """
    Generate transformation matrices to convert joint distribution to marginal distributions.
    
    Args:
        column: Array of shape (n_col, 3) representing distribution support space
        sum_col: Sum value for the columns
    
    Returns:
        joint_to_margin: List of 3 sparse matrices, each of shape (n_col, sum_col+1)
    """
    n_col = column.shape[0]
    joint_to_margin = []
    
    for dim in range(3):
        matrix = sparse.lil_matrix((n_col, sum_col + 1), dtype=np.float64)
        for i in range(n_col):
            matrix[i, column[i, dim]] = 1.0
        joint_to_margin.append(matrix.tocsr())
    
    return joint_to_margin

