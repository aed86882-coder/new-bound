"""
Initialize experiment information.
"""

import os


def InitExp(exp_name, obj_mode, create_dir=True, K=None):
    """
    Initialize experiment information.
    
    Args:
        exp_name: Name of the experiment
        obj_mode: Objective mode ('omega', 'alpha', or 'mu')
        create_dir: Whether to create experiment directory
        K: K parameter value (for omega mode)
    """
    from .ExpInfo import set_expinfo
    
    expinfo = {
        'exp_name': exp_name,
        'obj_mode': obj_mode,
        'K': K if K is not None else 1.0
    }
    
    set_expinfo(expinfo)
    
    # Create experiment directory if requested
    if create_dir and not os.path.exists(exp_name):
        os.makedirs(exp_name)
        
        # Create manual_stop.txt with initial value 0
        with open(os.path.join(exp_name, 'manual_stop.txt'), 'w') as f:
            f.write('0\n')
    
    return expinfo

