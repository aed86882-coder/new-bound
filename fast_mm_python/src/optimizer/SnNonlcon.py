"""
Nonlinear constraint evaluation for the optimization problem.
"""

import numpy as np


def SnNonlcon(x, param_manager=None, workspace=None):
    """
    Evaluate nonlinear constraints and their gradients.
    
    Args:
        x: Parameter vector
        param_manager: ParamManager instance (will use global if None)
        workspace: Workspace instance (will use global if None)
    
    Returns:
        c: Inequality constraint values (c <= 0)
        ceq: Equality constraint values (ceq == 0)
        dc: Inequality constraint gradients
        dceq: Equality constraint gradients
    """
    if param_manager is None:
        from ..autograd import get_param_manager
        param_manager = get_param_manager()
    
    if workspace is None:
        from ..evaluation import get_workspace
        workspace = get_workspace()
    
    # Set current parameter values
    param_manager.SetValue(x)
    
    # Evaluate constraints
    gc, gceq = workspace.Evaluate()
    c, dc = param_manager.PackResults(gc)
    ceq, dceq = param_manager.PackResults(gceq)
    
    # Convert to dense arrays if needed
    if hasattr(dc, 'toarray'):
        dc = dc.toarray()
    if hasattr(dceq, 'toarray'):
        dceq = dceq.toarray()
    
    return c, ceq, dc, dceq

