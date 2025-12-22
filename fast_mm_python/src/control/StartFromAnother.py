"""
Start optimization from a nearby solution for a new K value.
"""


def StartFromAnother(q_, new_K, experiment_name, old_K, datafile_path):
    """
    Start optimization from a nearby solution for a new K value.
    
    Args:
        q_: q parameter value
        new_K: New K value to optimize for
        experiment_name: Name of the new experiment
        old_K: Old K value from the loaded parameters
        datafile_path: Path to the data file with old parameters
    """
    from .ExpInfo import set_q
    from .InitExp import InitExp
    from ..evaluation import Workspace, set_workspace
    from ..io import LoadParam
    
    set_q(q_)
    InitExp(experiment_name, 'omega', True, new_K)
    
    workspace = Workspace()
    workspace.Build(max_level=3)
    set_workspace(workspace)
    
    # Load parameters from file
    LoadParam(datafile_path, old_K, workspace.param_manager, workspace, None)
    
    print(f"[INFO] Starting optimization for K={new_K} from K={old_K} solution.")
    print("[INFO] This is a simplified implementation - full SNOPT optimization not available.")

