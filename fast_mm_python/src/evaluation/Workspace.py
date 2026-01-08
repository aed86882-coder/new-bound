"""
Workspace class - the outermost structure for managing optimization variables.

Represents the process of hashing at all levels and manages all parts and the global stage.
"""

import numpy as np
from ..autograd import ParamManager, GVar


# Global workspace instance
_workspace = None


def get_workspace():
    """Get the global workspace instance."""
    global _workspace
    return _workspace


def set_workspace(ws):
    """Set the global workspace instance."""
    global _workspace
    _workspace = ws


class Workspace:
    """
    Workspace manages all optimization variables and constraint evaluation.
    
    It creates instances for parts and global stage recursively and
    evaluates all constraints of the system.
    """
    
    def __init__(self):
        """Initialize empty workspace."""
        self.num_retain_comp_id = None
        self.num_retain_comp = None
        self.num_retain_comp_lasteval = None
        self.num_retain_glob_id = None
        self.num_retain_glob = None
        self.num_retain_glob_lasteval = None
        self.single_mat_size_id = None
        self.single_mat_size = None
        self.single_mat_size_lasteval = None
        self.value = None
        self.omega_id = None
        self.omega = None
        self.K_id = None
        self.K = None
        
        self.num_constr_c = 0
        self.num_constr_ceq = 0
        
        self.max_level = None
        self.param_manager = None
        self.parts = None  # Dictionary: {level: {id: part}}
        self.globstage = None
    
    def Build(self, max_level_):
        """
        Initialize the workspace and register variables.
        Build the complete PartInfo hierarchy recursively.
        
        Args:
            max_level_: Maximum level for the computation
        """
        from ..autograd import set_param_manager
        from ..control import get_expinfo
        from .GlobalStage import GlobalStage
        
        self.max_level = max_level_
        max_level = max_level_
        
        # Initialize parameter manager
        self.param_manager = ParamManager()
        set_param_manager(self.param_manager)
        set_workspace(self)
        
        expinfo = get_expinfo()
        
        # Register num_retain_comp variables
        self.num_retain_comp_id = [None, None, None]
        self.num_retain_glob_id = [None, None, None]
        
        for r in range(3):
            self.num_retain_comp_id[r] = [None] * (max_level + 1)
            for lv in range(2, max_level + 1):
                self.num_retain_comp_id[r][lv] = self.param_manager.Register(
                    1, lb=0, ub=np.inf, initializer=(0, 0.01)
                )
            self.num_retain_glob_id[r] = self.param_manager.Register(
                1, lb=0, ub=np.inf, initializer=(0, 0.01)
            )
        
        self.single_mat_size_id = self.param_manager.Register(
            1, lb=0, ub=np.inf, initializer=(0, 0.01)
        )
        
        # Register objectives based on mode
        obj_mode = expinfo.get('obj_mode', 'omega')
        
        if obj_mode == 'alpha':
            # Restrict omega == 2
            goal_eps = 1e-9
            self.omega_id = self.param_manager.Register(1, lb=2.0, ub=2.0 + goal_eps)
        else:
            self.omega_id = self.param_manager.Register(1, lb=0, ub=np.inf)
        
        if obj_mode == 'omega':
            # Restrict K == expinfo.K
            K_val = expinfo.get('K', 1.0)
            self.K_id = self.param_manager.Register(1, lb=K_val, ub=K_val)
        else:
            self.K_id = self.param_manager.Register(1, lb=0, ub=np.inf)
        
        if obj_mode == 'mu':
            # Restrict omega(K) <= 1 + 2 * K
            self.param_manager.AddLinearConstraint([
                (self.omega_id, np.array([[1.0]])),
                (self.K_id, np.array([[-2.0]]))
            ], np.array([1.0]))
        
        # Initialize parts dictionary
        self.parts = {}
        
        # Initialize global stage and build recursively
        self.globstage = GlobalStage()
        self.globstage.Build(max_level, self.parts)
        
        # print(f"[INFO] Workspace built with {self.param_manager.num_input} parameters")
        # print(f"[INFO] Parts created at levels 2-{max_level}")
        for level in range(2, max_level + 1):
            if level in self.parts:
                print(f"       Level {level}: {len(self.parts[level])} parts")
    
    def SetInitial(self):
        """
        Set initial values for auxiliary variables.
        Before calling, user should set K and omega to desired values.
        """
        # Evaluate to get auxiliary variable values
        self.Evaluate()
        
        # Set auxiliary variables
        for r in range(3):
            for l in range(2, self.max_level + 1):
                self.param_manager.SetSingleParam(
                    self.num_retain_comp_id[r][l],
                    self.num_retain_comp_lasteval[r][l]
                )
            self.param_manager.SetSingleParam(
                self.num_retain_glob_id[r],
                self.num_retain_glob_lasteval[r]
            )
        
        self.param_manager.SetSingleParam(
            self.single_mat_size_id,
            self.single_mat_size_lasteval
        )
    
    def Evaluate(self):
        """
        Evaluate all constraints.
        
        This is the core evaluation function that:
        1. Loads all parameters into GVars
        2. Evaluates all parts in three phases (Init, Pre, Post)
        3. Evaluates the global stage
        4. Collects all constraints
        
        Returns:
            c, ceq: Lists of GVar instances representing inequality and equality constraints
        """
        from ..control import get_q
        
        q = get_q()
        
        # Extract optimizable variables
        self.num_retain_comp = [None, None, None]
        self.num_retain_glob = [None, None, None]
        self.num_retain_comp_lasteval = [None, None, None]
        self.num_retain_glob_lasteval = [None, None, None]
        
        for r in range(3):
            self.num_retain_comp[r] = [None] * (self.max_level + 1)
            self.num_retain_comp_lasteval[r] = [None] * (self.max_level + 1)
            
            for lv in range(2, self.max_level + 1):
                var = self.param_manager.GetParam(self.num_retain_comp_id[r][lv])
                self.num_retain_comp[r][lv] = var
                self.num_retain_comp_lasteval[r][lv] = var.value[0]
            
            var = self.param_manager.GetParam(self.num_retain_glob_id[r])
            self.num_retain_glob[r] = var
            self.num_retain_glob_lasteval[r] = var.value[0]
        
        var = self.param_manager.GetParam(self.single_mat_size_id)
        self.single_mat_size = var
        self.single_mat_size_lasteval = var.value[0]
        
        self.omega = self.param_manager.GetParam(self.omega_id)
        self.K = self.param_manager.GetParam(self.K_id)
        
        # Initialize constraint lists
        c = []
        ceq = []
        
        # Phase 1: EvaluateInit - Initialize all parts
        for level in range(2, self.max_level + 1):
            if level in self.parts:
                for part in self.parts[level].values():
                    part.EvaluateInit()
        
        self.globstage.EvaluateInit()
        
        # Phase 2: EvaluatePre - Propagate part_frac from top to bottom
        self.globstage.EvaluatePre()
        
        for level in range(self.max_level, 1, -1):  # From max_level down to 2
            if level in self.parts:
                for part in self.parts[level].values():
                    if hasattr(part, 'EvaluatePre'):
                        part.EvaluatePre()
        
        # Phase 3: EvaluatePost - Compute contributions from bottom to top
        for level in range(2, self.max_level + 1):  # From 2 up to max_level
            if level in self.parts:
                for part in self.parts[level].values():
                    if hasattr(part, 'EvaluatePost'):
                        part.EvaluatePost()
        
        self.globstage.EvaluatePost(self.parts)
        
        # Collect constraints from parts
        
        # Lagrange constraints (if needed for optimization)
        # for level in range(3, self.max_level + 1):
        #     if level in self.parts:
        #         for part in self.parts[level].values():
        #             if hasattr(part, 'GetLagrangeConstraints'):
        #                 ceq.extend(part.GetLagrangeConstraints())
        
        # Global stage Lagrange constraints
        # if hasattr(self.globstage, 'GetLagrangeConstraints'):
        #     ceq.extend(self.globstage.GetLagrangeConstraints())
        
        return c, ceq
    
    def GetOmegaPos(self):
        """Get the position of omega in the parameter vector."""
        return self.param_manager.group_startpos[self.omega_id]
    
    def GetKPos(self):
        """Get the position of K in the parameter vector."""
        return self.param_manager.group_startpos[self.K_id]
    
    def GetObjective(self):
        """
        Compute the objective value based on the mode.
        
        Returns:
            objective: Scalar value to minimize
        """
        from ..control import get_expinfo
        expinfo = get_expinfo()
        obj_mode = expinfo.get('obj_mode', 'omega')
        
        if obj_mode == 'omega':
            # Minimize omega
            return self.omega.value[0]
        elif obj_mode == 'alpha':
            # Maximize K (minimize -K)
            return -self.K.value[0]
        elif obj_mode == 'mu':
            # Minimize K
            return self.K.value[0]
        else:
            raise ValueError(f"Unknown objective mode: {obj_mode}")
