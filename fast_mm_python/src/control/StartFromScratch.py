"""
Start optimization from Le Gall's initial point.
"""


def StartFromScratch(q_, K_, exp_name):
    """
    Start from the initial point generated from Le Gall's parameters
    and run the optimization.
    
    Args:
        q_: q parameter value
        K_: K parameter value
        exp_name: Name of the experiment
    
    Example:
        StartFromScratch(5, 1.00, 'K100')
    """
    from .ExpInfo import set_q
    from .InitExp import InitExp
    from ..evaluation import Workspace, set_workspace
    from ..init import InitialPoint
    from ..optimizer import SnContinue
    
    # Set global q
    set_q(q_)
    
    # Initialize experiment
    InitExp(exp_name, 'omega', True, K_)
    
    # Build workspace
    workspace = Workspace()
    workspace.Build(max_level=3)
    set_workspace(workspace)
    
    # Get initial point
    InitialPoint()
    
    # Run optimization
    print(f"[INFO] Starting optimization for K={K_} from scratch.")
    print("[INFO] This is a simplified implementation - full SNOPT optimization not available.")
    print("[INFO] For full optimization, use the original MATLAB code.")
    
    # In the full implementation, this would call:
    # SnContinue()
    # KeepClimb()

