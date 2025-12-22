"""
Start alpha optimization from a nearby solution.
"""


def StartAlphaFromAnother(q_, experiment_name, datafile_path):
    """
    Start alpha optimization from a nearby solution.
    
    Args:
        q_: q parameter value
        experiment_name: Name of the experiment
        datafile_path: Path to the data file with initial parameters
    """
    from .ExpInfo import set_q
    from .InitExp import InitExp
    from ..evaluation import Workspace, set_workspace
    from ..io import LoadParam
    
    set_q(q_)
    InitExp(experiment_name, 'alpha', True, None)
    
    workspace = Workspace()
    workspace.Build(max_level=3)
    set_workspace(workspace)
    
    LoadParam(datafile_path, None, workspace.param_manager, workspace, None)
    
    print(f"[INFO] Starting alpha optimization from provided solution.")
    print("[INFO] This is a simplified implementation - full SNOPT optimization not available.")

