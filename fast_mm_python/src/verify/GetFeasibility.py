"""
Feasibility checking for constraint satisfaction.
"""

import numpy as np


def GetFeasibility(param_manager=None, workspace=None, expinfo=None):
    """
    Evaluate the max violation of constraints for the current parameters.
    
    Args:
        param_manager: ParamManager instance (will use global if None)
        workspace: Workspace instance (will use global if None)
        expinfo: Experiment info dict (will use global if None)
    
    Returns:
        maxViol: Maximum constraint violation
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
    
    # Nonlinear constraints
    gc, gceq = workspace.Evaluate()
    c, _ = param_manager.PackResults(gc)
    ceq, _ = param_manager.PackResults(gceq)
    
    # Start with nonlinear constraint violations
    maxViol = 0.0
    if len(c) > 0:
        maxViol = max(maxViol, np.max(c))
    if len(ceq) > 0:
        maxViol = max(maxViol, np.max(np.abs(ceq)))
    
    # Linear constraints
    A, b, Aeq, beq = param_manager.GetLinearConstraints()
    
    if A.shape[0] > 0:
        lin_viol = A @ param_manager.cur_x - b
        maxViol = max(maxViol, np.max(lin_viol))
    
    if Aeq.shape[0] > 0:
        lineq_viol = np.abs(Aeq @ param_manager.cur_x - beq)
        maxViol = max(maxViol, np.max(lineq_viol))
    
    # Check bounds
    lb_viol = param_manager.lb - param_manager.cur_x
    ub_viol = param_manager.cur_x - param_manager.ub
    maxViol = max(maxViol, np.max(lb_viol))
    maxViol = max(maxViol, np.max(ub_viol))
    
    # Special check for alpha mode
    obj_mode = expinfo.get('obj_mode', 'omega')
    if obj_mode == 'alpha':
        omega = param_manager.cur_x[workspace.GetOmegaPos()]
        maxViol = max(maxViol, omega - 2.0)
    
    return maxViol

