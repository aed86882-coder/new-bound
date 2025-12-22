"""
Load parameters from a file to param_manager.
"""

import numpy as np


def LoadParam(filename='best.npz', K_=None, param_manager=None, workspace=None, expinfo=None):
    """
    Load parameters from a file to param_manager.
    It requires the whole workspace to be built in advance, and expinfo contains the correct information.
    
    Args:
        filename: Path to the parameter file (default: 'best.npz')
        K_: Optional K value for old-version parameters
        param_manager: ParamManager instance (will use global if None)
        workspace: Workspace instance (will use global if None)
        expinfo: Experiment info dict (will use global if None)
    
    Returns:
        vec: Loaded parameter vector
    """
    # Import here to avoid circular dependency
    if param_manager is None:
        from ..autograd import get_param_manager
        param_manager = get_param_manager()
    
    if workspace is None:
        from ..evaluation import get_workspace
        workspace = get_workspace()
    
    if expinfo is None:
        from ..control import get_expinfo
        expinfo = get_expinfo()
    
    # Load parameters from file
    # Support both .npz (NumPy) and .mat (MATLAB) formats
    if filename.endswith('.mat'):
        # Load MATLAB format
        try:
            import scipy.io as sio
            file_data = sio.loadmat(filename)
            vec = file_data['params'].flatten()
        except:
            raise ValueError(f"Failed to load MATLAB file: {filename}")
    else:
        # Load NumPy format
        file_data = np.load(filename)
        vec = file_data['params']
    
    # Check version compatibility
    if len(param_manager.lb) == len(vec):
        # Versions match, just plug in
        param_manager.SetValue(vec)
    elif len(param_manager.lb) == len(vec) + 1:
        # Loading parameters are old version, need to add K
        if K_ is None:
            K_ = expinfo.get('K', 1.0)
        print(f'[WARN] Old-version parameters loading. Setting K = {K_:.4f}')
        K_pos = workspace.GetKPos()
        vec = np.concatenate([vec[:K_pos], [K_], vec[K_pos:]])
        param_manager.SetValue(vec)
    else:
        raise ValueError(f"Parameter dimension mismatch: expected {len(param_manager.lb)}, got {len(vec)}")
    
    return vec
