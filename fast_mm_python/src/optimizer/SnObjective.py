"""
Objective function evaluation for the optimization problem.
"""

import numpy as np


def SnObjective(x, param_manager=None, workspace=None, expinfo=None):
    """
    Evaluate the objective function and its gradient.
    
    The objective depends on expinfo.obj_mode:
    - 'omega': minimize omega
    - 'alpha': maximize K (minimize -K)
    - 'mu': minimize K under omega(K) <= 1 + 2*K
    
    Args:
        x: Parameter vector
        param_manager: ParamManager instance (will use global if None)
        workspace: Workspace instance (will use global if None)
        expinfo: Experiment info dict (will use global if None)
    
    Returns:
        f: Objective value
        df: Objective gradient
    """
    if param_manager is None:
        from ..autograd import get_param_manager
        param_manager = get_param_manager()
    
    if workspace is None:
        from ..evaluation import get_workspace
        workspace = get_workspace()
    
    if expinfo is None:
        from ..control import get_expinfo
        expinfo = get_expinfo()
    
    obj_mode = expinfo.get('obj_mode', 'omega')
    df = np.zeros(len(x))
    
    if obj_mode == 'omega':
        omega_pos = workspace.GetOmegaPos()
        f = x[omega_pos]
        df[omega_pos] = 1.0
    elif obj_mode == 'alpha':
        K_pos = workspace.GetKPos()
        f = -x[K_pos]
        df[K_pos] = -1.0
    elif obj_mode == 'mu':
        K_pos = workspace.GetKPos()
        f = x[K_pos]
        df[K_pos] = 1.0
    else:
        raise ValueError(f"Unknown objective mode: {obj_mode}")
    
    return float(f), df

