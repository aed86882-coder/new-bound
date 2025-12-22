"""
Fix alpha distribution with precision issues.
"""

import numpy as np


def FixAlpha(alph):
    """
    Fix alpha distribution with precision issues.
    Ensures non-negativity and normalization.
    
    Args:
        alph: Distribution array
    
    Returns:
        Fixed distribution
    """
    alph = np.maximum(alph, 0)
    total = np.sum(alph)
    if total > 0:
        alph = alph / total
    return alph

