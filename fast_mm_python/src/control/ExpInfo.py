"""
Experiment information management.
"""

import os


# Global experiment info and q value
_expinfo = {}
_q = 5


def get_expinfo():
    """Get the global experiment info dictionary."""
    global _expinfo
    return _expinfo


def set_expinfo(info):
    """Set the global experiment info dictionary."""
    global _expinfo
    _expinfo = info


def get_q():
    """Get the global q value."""
    global _q
    return _q


def set_q(q_val):
    """Set the global q value."""
    global _q
    _q = q_val


def InitExp(exp_name, obj_mode, create_dir=True, K=None):
    """
    Initialize experiment information.
    
    Args:
        exp_name: Name of the experiment
        obj_mode: Objective mode ('omega', 'alpha', or 'mu')
        create_dir: Whether to create experiment directory
        K: K parameter value (for omega mode)
    """
    global _expinfo
    
    _expinfo = {
        'exp_name': exp_name,
        'obj_mode': obj_mode,
        'K': K if K is not None else 1.0
    }
    
    # Create experiment directory if requested
    if create_dir and not os.path.exists(exp_name):
        os.makedirs(exp_name)
        
        # Create manual_stop.txt with initial value 0
        with open(os.path.join(exp_name, 'manual_stop.txt'), 'w') as f:
            f.write('0\n')
    
    return _expinfo

