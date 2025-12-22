"""
Concatenate two Complete Split Distributions using Kronecker product.
"""

import numpy as np
from scipy import sparse


def ConcatCSD(lhs, rhs):
    """
    Concatenate two Complete Split Distributions using Kronecker product.
    
    Complete split distributions are represented with GVars of length 3^(2^(level-1)).
    
    Args:
        lhs: Left-hand side GVar
        rhs: Right-hand side GVar
    
    Returns:
        res: Concatenated GVar using Kronecker product
    """
    from ..autograd import GVar
    
    # Compute Kronecker product of values
    res_value = np.kron(lhs.value, rhs.value)
    
    # Compute Kronecker product of gradients
    # lhs.grad is (num_input, n1), rhs.value is (n2,)
    # We want each column of lhs.grad to be multiplied by rhs.value
    n1 = len(lhs.value)
    n2 = len(rhs.value)
    num_input = lhs.num_input
    
    # First term: kron(lhs.grad, rhs.value)
    # For each column i in lhs.grad, create n2 columns multiplied by rhs.value[j]
    grad1 = sparse.lil_matrix((num_input, n1 * n2))
    for i in range(n1):
        for j in range(n2):
            grad1[:, i * n2 + j] = lhs.grad[:, i] * rhs.value[j]
    
    # Second term: kron(lhs.value, rhs.grad)
    # For each column j in rhs.grad, replicate it n1 times with multipliers lhs.value[i]
    grad2 = sparse.lil_matrix((num_input, n1 * n2))
    for i in range(n1):
        for j in range(n2):
            grad2[:, i * n2 + j] = rhs.grad[:, j] * lhs.value[i]
    
    res_grad = (grad1 + grad2).tocsr()
    
    return GVar(lhs.num_input, res_value, 'grad', res_grad)

