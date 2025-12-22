"""
Rotate a 3-element vector to the left (by n times).
"""

import numpy as np


def Rot3(x, n=1):
    """
    Rotate a 3-element vector to the left (by n times).
    
    Args:
        x: 3-element array, list, or GVar
        n: Number of rotations (default: 1)
    
    Returns:
        Rotated vector (same type as input)
    """
    # Handle GVar objects
    from ..autograd import GVar
    
    is_gvar = isinstance(x, GVar)
    if is_gvar:
        x_val = x.value
    else:
        x_val = np.atleast_1d(x)
    
    # Ensure x_val is at least 1D
    x_val = np.atleast_1d(x_val)
    
    # Check if x_val has at least 3 elements
    if x_val.shape[0] < 3:
        # If less than 3 elements, cannot rotate - return as is
        if is_gvar:
            return x
        else:
            return x_val
    
    n = n % 3
    
    if n == 0:
        return x
    elif n == 1:
        rotated = np.array([x_val[1], x_val[2], x_val[0]])
    elif n == 2:
        rotated = np.array([x_val[2], x_val[0], x_val[1]])
    else:
        rotated = x_val
    
    # Return GVar if input was GVar
    if is_gvar:
        result = GVar(x.num_input, rotated)
        # Copy gradient structure if available
        if hasattr(x, 'grad') and x.grad is not None:
            # Rotate gradients correspondingly
            if n == 1:
                # Rotate gradient columns: [g0, g1, g2] -> [g1, g2, g0]
                result.grad = x.grad[:, [1, 2, 0]]
            elif n == 2:
                # Rotate gradient columns: [g0, g1, g2] -> [g2, g0, g1]
                result.grad = x.grad[:, [2, 0, 1]]
        return result
    else:
        return rotated

