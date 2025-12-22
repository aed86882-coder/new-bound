"""
Check if every non-zero entry of CSD has index-sum k.
This is only for debug use.
"""

import numpy as np


def CheckCSD(csd, power, k):
    """
    Check if every non-zero entry of CSD has index-sum k.
    This is only for debug use.
    
    Args:
        csd: Complete split distribution (can be GVar or array)
        power: Power value for decoding
        k: Expected sum of indices
    
    Raises:
        ValueError: If any non-zero entry has index-sum != k
    """
    from .DecodeCSD import DecodeCSD
    
    # Extract value if it's a GVar
    from ..autograd import GVar
    if isinstance(csd, GVar):
        csd_val = csd.value
    else:
        csd_val = np.atleast_1d(csd)
    
    # Check each entry
    for id_val in range(1, 3 ** power + 1):
        if csd_val[id_val - 1] != 0:
            idx_sum = np.sum(DecodeCSD(id_val, power))
            if idx_sum != k:
                raise ValueError(f"CSD's sum is not {k} at index {id_val}")

