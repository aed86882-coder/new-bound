"""
From joint distribution, compute marginal distributions.
"""


def MarginalDist(alph, joint_to_margin):
    """
    From joint distribution, compute marginal distributions.
    Supports GVar input.
    
    Args:
        alph: Joint distribution (can be GVar or array)
        joint_to_margin: List of transformation matrices from JointToMargin
    
    Returns:
        alph_x, alph_y, alph_z: Three marginal distributions
    """
    # Support both GVar and regular arrays
    # The @ operator (matrix multiplication) works for both
    alph_x = alph @ joint_to_margin[0].T
    alph_y = alph @ joint_to_margin[1].T
    alph_z = alph @ joint_to_margin[2].T
    
    return alph_x, alph_y, alph_z

