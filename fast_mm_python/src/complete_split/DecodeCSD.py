"""
Decode an ID (from 1 to 3^power) to an index array (i_1, ..., i_power).
"""

import numpy as np


def DecodeCSD(id_val, power):
    """
    Decode an ID (from 1 to 3^power) to an index array (i_1, ..., i_power).
    
    Args:
        id_val: ID to decode, in range [1, 3^power]
        power: Length of the output array
    
    Returns:
        arr: Index array of length power, each element in {0, 1, 2}
    """
    arr = np.zeros(power, dtype=int)
    id_val = id_val - 1  # Base-3 index starting from 0
    
    for i in range(power - 1, -1, -1):
        arr[i] = id_val % 3
        id_val = id_val // 3
    
    return arr

