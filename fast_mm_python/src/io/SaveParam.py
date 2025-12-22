"""
Save parameters to a file.
"""

import numpy as np
import os


def SaveParam(filename='best.npz', x=None, param_manager=None):
    """
    Save parameters to a file.
    
    Args:
        filename: Path to save the parameter file (default: 'best.npz')
        x: Parameter vector to save (default: use param_manager.cur_x)
        param_manager: ParamManager instance (will use global if None)
    """
    if param_manager is None:
        from ..autograd import get_param_manager
        param_manager = get_param_manager()
    
    if x is None:
        x = param_manager.cur_x
    
    # Ensure directory exists
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    
    # Save in NumPy format
    np.savez(filename, params=x)
