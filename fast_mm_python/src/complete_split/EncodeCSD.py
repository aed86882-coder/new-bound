"""
Encode an index array (i_1, ..., i_power) to a number in [1, 3^power].
"""

import numpy as np


def EncodeCSD(arr, power=None):
    """
    Encode an index array (i_1, ..., i_power) to a number in [1, 3^power].
    
    Args:
        arr: Index array, each element in {0, 1, 2}
        power: Power value (unused, for compatibility with MATLAB version)
    
    Returns:
        id: Encoded ID starting from 1
    """
    arr = np.atleast_1d(arr)
    id_val = 0
    for i in range(len(arr)):
        id_val = id_val * 3 + arr[i]
    return int(id_val) + 1  # Base-3 index starting from 1

