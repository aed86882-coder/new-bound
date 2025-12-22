"""
Same as Rot3, but for list/tuple (cell-array equivalent).
"""


def Rot3c(x, n=1):
    """
    Same as Rot3, but for list/tuple (cell-array equivalent).
    
    Args:
        x: 3-element list or tuple
        n: Number of rotations (default: 1)
    
    Returns:
        Rotated list
    """
    n = n % 3
    
    if n == 0:
        return x
    elif n == 1:
        return [x[1], x[2], x[0]]
    elif n == 2:
        return [x[2], x[0], x[1]]

