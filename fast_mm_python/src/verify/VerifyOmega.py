"""
Verification routine for omega bounds.
"""

import numpy as np


def VerifyOmega(q_, K_, datapath):
    """
    Verify if omega(1, 1, K) <= claimed value.
    
    Args:
        q_: q parameter value
        K_: K parameter value
        datapath: Path to the data file containing parameters
    """
    from ..control import InitExp, set_q
    from ..evaluation import Workspace, set_workspace
    from ..autograd import set_param_manager
    from ..io import LoadParam
    from .GetFeasibility import GetFeasibility
    
    # Set global q
    set_q(q_)
    q = q_
    max_level = 3
    
    # Initialize experiment
    InitExp('__verify', 'omega', False, K_)
    
    # Build workspace
    workspace = Workspace()
    workspace.Build(max_level)
    set_workspace(workspace)
    
    # Load parameters
    LoadParam(datapath, K_, workspace.param_manager, workspace, None)
    
    # Get feasibility
    maxViol = GetFeasibility(workspace.param_manager, workspace, None)
    omega = workspace.param_manager.cur_x[workspace.GetOmegaPos()]
    
    print(f"omega({K_:.6f}) <= {omega:.8f} \t(MaxViolation: {maxViol:.6e})")
    
    if maxViol > 1.1e-6:
        print("[WARN] The last result seems wrong (the MaxViolation is too large).")
    elif maxViol > 1.1e-9:
        print("[WARN] The last result is not very accurate (MaxViolation > 1e-9).")

