"""
Verification routine for mu bounds.
"""


def VerifyMu(q_, datapath):
    """
    Verify mu bound.
    
    Args:
        q_: q parameter value
        datapath: Path to the data file containing parameters
    """
    from ..control import InitExp, set_q
    from ..evaluation import Workspace, set_workspace
    from ..io import LoadParam
    from .GetFeasibility import GetFeasibility
    
    # Set global q
    set_q(q_)
    max_level = 3
    
    # Initialize experiment
    InitExp('__verify', 'mu', False, None)
    
    # Build workspace
    workspace = Workspace()
    workspace.Build(max_level)
    set_workspace(workspace)
    
    # Load parameters
    LoadParam(datapath, None, workspace.param_manager, workspace, None)
    
    # Get feasibility
    maxViol = GetFeasibility(workspace.param_manager, workspace, None)
    K = workspace.param_manager.cur_x[workspace.GetKPos()]
    
    print(f"mu <= {K:.8f} \t(MaxViolation: {maxViol:.6e})")
    
    if maxViol > 1.1e-6:
        print("[WARN] The last result seems wrong (the MaxViolation is too large).")
    elif maxViol > 1.1e-9:
        print("[WARN] The last result is not very accurate (MaxViolation > 1e-9).")

