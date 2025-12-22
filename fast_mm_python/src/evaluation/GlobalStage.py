"""
GlobalStage class for managing the global hashing stage.
This class is similar to PartInfo. It stores the ID and GVar of optimizable variables
in the global stage. Also, hash penalty term is taken into account.
"""

import numpy as np
from ..autograd import GVar
from ..utils import PrepareShapes, JointToMargin, MarginalDist
from .FindOrCreatePart import CreatePartInstance


# Global instance
_global_stage = None


def get_global_stage():
    """Get the global GlobalStage instance."""
    global _global_stage
    return _global_stage


def set_global_stage(gs):
    """Set the global GlobalStage instance."""
    global _global_stage
    _global_stage = gs


class GlobalStage:
    """
    Manages the global hashing stage.
    Similar to PartInfo but at the highest level.
    """
    
    def __init__(self):
        """Initialize empty GlobalStage."""
        self.level = None
        self.power = None
        self.sum_col = None
        self.shapes = None
        self.num_shape = None
        self.joint_to_margin = None
        
        # Pointers to highest-level parts (1x3 lists)
        self.part_id = None
        self.part_ptr = None
        
        # Optimizable variables
        self.region_prop_id = None
        self.region_prop = None  # 1x3 GVar array, sums to 1
        
        # Parameters within each region (1x3 lists)
        self.dist_id = None
        self.dist = None
        self.dist_max_id = None
        self.dist_max = None
        self.lam_margin_id = None  # 1x3 list, each cell is 1x(sum_col+1) array
        self.lam_sum_id = None  # scalar
        self.lam_margin = None
        self.lam_sum = None
        
        # GVars (outputs)
        self.mat_size = None
        self.num_block = None  # Number of max-level blocks in global stage hashing
        self.hash_penalty_term = None
        self.p_comp = None
        self.complete_split = None
    
    def Build(self, max_level, parts):
        """
        Build the global stage and recursively construct top-level parts.
        
        Args:
            max_level: Maximum level
            parts: Global parts dictionary {level: {id: part}}
        """
        from ..autograd import get_param_manager
        from ..control import get_expinfo
        
        param_manager = get_param_manager()
        expinfo = get_expinfo()
        
        self.level = max_level
        self.power = 2 ** (self.level - 1)
        self.sum_col = 2 ** self.level
        self.shapes = PrepareShapes(self.level)
        self.num_shape = self.shapes.shape[0]
        self.joint_to_margin = JointToMargin(self.shapes, self.sum_col)
        
        self.RegisterVariablesAndLinearConstraints(expinfo)
        
        # Recursively build the parts
        self.part_id = [None, None, None]
        self.part_ptr = [None, None, None]
        
        if self.level not in parts:
            parts[self.level] = {}
        
        for r in range(3):
            self.part_id[r] = np.zeros(self.num_shape, dtype=int)
            self.part_ptr[r] = [None] * self.num_shape
            
            for i in range(self.num_shape):
                # Assign new ID
                if parts[self.level]:
                    cur_id = max(parts[self.level].keys()) + 1
                else:
                    cur_id = 0
                
                # Create instance
                shape = tuple(self.shapes[i, :])
                ptr_cur = CreatePartInstance(self.level, shape)
                parts[self.level][cur_id] = ptr_cur
                
                # Build
                ptr_cur.Build(self.level, cur_id, shape, (0, r), parts)
                
                self.part_id[r][i] = cur_id
                self.part_ptr[r][i] = ptr_cur
    
    def RegisterVariablesAndLinearConstraints(self, expinfo):
        """
        Register all optimizable variables and linear constraints.
        
        Args:
            expinfo: Experiment info dictionary
        """
        from ..autograd import get_param_manager
        param_manager = get_param_manager()
        
        # Register variables
        # region_prop
        self.region_prop_id = param_manager.Register(3, lb=0, ub=1, initializer=(0, 1.0))
        
        # dist and dist_max
        self.dist_id = [None, None, None]
        self.dist_max_id = [None, None, None]
        for r in range(3):
            self.dist_id[r] = param_manager.Register(
                self.num_shape, lb=0, ub=1, initializer=(0, 1.0 / self.num_shape)
            )
            self.dist_max_id[r] = param_manager.Register(
                self.num_shape, lb=0, ub=1, initializer=(0, 1.0 / self.num_shape)
            )
        
        # Lagrange multipliers
        self.lam_margin_id = [None, None, None]
        self.lam_sum_id = [None, None, None]
        for r in range(3):
            self.lam_margin_id[r] = [None, None, None]
            for t in range(3):
                self.lam_margin_id[r][t] = param_manager.Register(
                    self.sum_col + 1, lb=-np.inf, ub=np.inf, initializer=(-0.01, 0.01)
                )
            self.lam_sum_id[r] = param_manager.Register(
                1, lb=-np.inf, ub=np.inf, initializer=(-0.01, 0.01)
            )
        
        # Add linear constraints
        # sum(region_prop) == 1
        param_manager.AddLinearConstraintEq([
            (self.region_prop_id, np.ones((1, 3)))
        ], np.array([1.0]))
        
        for r in range(3):
            # sum(dist) == 1
            param_manager.AddLinearConstraintEq([
                (self.dist_id[r], np.ones((1, self.num_shape)))
            ], np.array([1.0]))
            
            # dist and dist_max share marginals
            for t in range(3):
                A = self.joint_to_margin[t]
                if A.shape[1] != self.num_shape:
                    continue
                
                margin_size = A.shape[0]
                A_array = A.T.toarray() if hasattr(A, 'toarray') else A.T
                
                param_manager.AddLinearConstraintEq([
                    (self.dist_id[r], A_array),
                    (self.dist_max_id[r], -A_array)
                ], np.zeros(margin_size))
        
        # Y and Z are symmetric
        param_manager.AddLinearConstraintEq([
            (self.region_prop_id, np.array([[0, 1, -1]]))
        ], np.array([0.0]))
        
        # If K == 1, X, Y, Z are all symmetric
        if expinfo.get('obj_mode') == 'omega' and expinfo.get('K', 1.0) == 1.0:
            param_manager.AddLinearConstraintEq([
                (self.region_prop_id, np.array([[1, -1, 0]]))
            ], np.array([0.0]))
    
    def SetInitial(self, json):
        """
        Load distributions from the json file (i.e., Le Gall's parameters).
        
        Args:
            json: List of {shape, dist} dictionaries
        """
        from ..autograd import get_param_manager
        param_manager = get_param_manager()
        
        # Set region proportions to uniform
        param_manager.SetSingleParam(self.region_prop_id, np.array([1/3, 1/3, 1/3]))
        
        # Find shape (0, 0, 0) in json
        for item in json:
            cur_shape = tuple(item['shape'])
            cur_dist = np.array(item['dist'])
            
            if cur_shape == (0, 0, 0):
                # Set distributions
                for r in range(3):
                    param_manager.SetSingleParam(self.dist_id[r], cur_dist)
                    param_manager.SetSingleParam(self.dist_max_id[r], cur_dist)
                
                # TODO: Get lambdas using GetLambda function
                # For now, set to zero
                for r in range(3):
                    param_manager.SetSingleParam(self.lam_sum_id[r], np.array([0.0]))
                    for t in range(3):
                        param_manager.SetSingleParam(
                            self.lam_margin_id[r][t], 
                            np.zeros(self.sum_col + 1)
                        )
                
                return
        
        # If not found, error
        raise ValueError('Cannot find initial distribution for global stage (shape 0,0,0)')
    
    def EvaluateInit(self):
        """
        Initialize evaluation: load parameter values.
        """
        from ..autograd import get_param_manager
        param_manager = get_param_manager()
        
        self.region_prop = param_manager.GetParam(self.region_prop_id)
        self.dist = [None, None, None]
        self.dist_max = [None, None, None]
        self.lam_margin = [None, None, None]
        self.lam_sum = [None, None, None]
        
        for r in range(3):
            self.dist[r] = param_manager.GetParam(self.dist_id[r])
            self.dist_max[r] = param_manager.GetParam(self.dist_max_id[r])
            
            self.lam_margin[r] = [None, None, None]
            for t in range(3):
                self.lam_margin[r][t] = param_manager.GetParam(self.lam_margin_id[r][t])
            
            self.lam_sum[r] = param_manager.GetParam(self.lam_sum_id[r])
    
    def EvaluatePre(self):
        """
        Pre-evaluation: propagate part_frac to the max level.
        """
        # Safely extract region_prop values
        region_prop_vals = self.region_prop.value
        if len(region_prop_vals) < 3:
            # If less than 3 elements, extend with defaults
            region_prop_vals = np.array([1/3, 1/3, 1/3])
        
        for r in range(3):
            # Safely access region_prop_vals
            region_val = region_prop_vals[r] if r < len(region_prop_vals) else (1/3)
            
            for i in range(self.num_shape):
                # Set part_frac keeping gradient information
                dist_val = self.dist[r].value[i] if i < len(self.dist[r].value) else 0
                self.part_ptr[r][i].part_frac = dist_val * region_val
    
    def EvaluatePost(self, parts):
        """
        Post-evaluation: compute contributions.
        
        Args:
            parts: Global parts dictionary
        """
        from ..autograd import get_param_manager
        param_manager = get_param_manager()
        
        # Safely extract region_prop values
        region_prop_vals = self.region_prop.value
        if len(region_prop_vals) < 3:
            region_prop_vals = np.array([1/3, 1/3, 1/3])
        
        # hash_penalty_term - count three regions separately
        self.hash_penalty_term = [None, None, None]
        for r in range(3):
            entropy_max = self.dist_max[r].Entropy()
            entropy_actual = self.dist[r].Entropy()
            region_val = region_prop_vals[r] if r < len(region_prop_vals) else (1/3)
            self.hash_penalty_term[r] = (entropy_max - entropy_actual) * region_val
        
        # num_block - count three regions separately
        self.num_block = [None, None, None]
        for r in range(3):
            dist_x, dist_y, dist_z = MarginalDist(self.dist[r], self.joint_to_margin)
            region_val = region_prop_vals[r] if r < len(region_prop_vals) else (1/3)
            self.num_block[r] = GVar(param_manager.num_input, [
                dist_x.Entropy(),
                dist_y.Entropy(),
                dist_z.Entropy()
            ]) * region_val
        
        # mat_size: sum over all parts of all levels
        self.mat_size = GVar(param_manager.num_input, [0, 0, 0])
        for level in range(2, self.level + 1):
            if level in parts:
                for part in parts[level].values():
                    if hasattr(part, 'mat_size_contribution') and part.mat_size_contribution is not None:
                        self.mat_size = self.mat_size + part.mat_size_contribution
        
        # complete_split
        self.complete_split = [[None] * 3 for _ in range(3)]
        csd_size = 3 ** self.power
        
        for r in range(3):
            for t in range(3):
                self.complete_split[r][t] = GVar.Zeros(1, csd_size)
                
                for i in range(self.num_shape):
                    ptr = self.part_ptr[r][i]
                    if hasattr(ptr, 'complete_split') and ptr.complete_split is not None:
                        self.complete_split[r][t] = self.complete_split[r][t] + \
                                                    self.dist[r].value[i] * ptr.complete_split[t]
        
        # p_comp - count three regions separately
        self.p_comp = [None, None, None]
        for r in range(3):
            numerator = self.GetPcompNumerator(r)
            denominator = self.GetPcompDenominator(r)
            region_val = region_prop_vals[r] if r < len(region_prop_vals) else (1/3)
            self.p_comp[r] = (numerator - denominator) * region_val
    
    def GetPcompNumerator(self, region):
        """
        Compute numerator for p_comp.
        
        Args:
            region: Hashing region (0, 1, or 2)
        
        Returns:
            GVar: Numerator value
        """
        from ..autograd import get_param_manager
        param_manager = get_param_manager()
        
        res = GVar(param_manager.num_input, 0)
        
        # Weighted sum of CSDs
        weighted_sum_csd = [GVar.Zeros(1, 3 ** self.power) for _ in range(self.sum_col + 1)]
        prob_sum_csd = GVar.Zeros(1, self.sum_col + 1)
        used_csd = np.zeros(self.sum_col + 1, dtype=bool)
        
        for i in range(self.num_shape):
            ptr = self.part_ptr[region][i]
            
            if ptr.shape[region] == 0:
                continue  # Zero dimension doesn't contribute
            
            if min(ptr.shape) == 0:
                # Contains zero: directly contribute
                res = res + self.dist[region].value[i] * ptr.complete_split[region].Entropy()
            else:
                # Help compute average CSD
                idx = ptr.shape[region]
                weighted_sum_csd[idx] = weighted_sum_csd[idx] + \
                                       self.dist[region].value[i] * ptr.complete_split[region]
                prob_sum_csd.value[idx] += self.dist[region].value[i]
                used_csd[idx] = True
        
        # Type-2 contribution
        for i in range(1, self.sum_col + 1):  # Skip 0
            if used_csd[i]:
                res = res + weighted_sum_csd[i].NormalizedEntropy(prob_sum_csd.value[i])
        
        return res
    
    def GetPcompDenominator(self, region):
        """
        Compute denominator for p_comp: H(CSD) - H(alpha_Z).
        
        Args:
            region: Hashing region (0, 1, or 2)
        
        Returns:
            GVar: Denominator value
        """
        res = self.complete_split[region][region].Entropy()
        
        dist_x, dist_y, dist_z = MarginalDist(self.dist[region], self.joint_to_margin)
        dist_margins = [dist_x, dist_y, dist_z]
        res = res - dist_margins[region].Entropy()
        
        return res
    
    def GetLagrangeConstraints(self):
        """
        Get Lagrange multiplier constraints for optimization.
        
        Returns:
            List of GVar constraints
        """
        ceq = []
        
        for r in range(3):
            for t in range(self.num_shape):
                cur_shape = self.shapes[t, :]
                
                constr = (self.lam_margin[r][0].value[cur_shape[0]] +
                         self.lam_margin[r][1].value[cur_shape[1]] +
                         self.lam_margin[r][2].value[cur_shape[2]] +
                         self.lam_sum[r].value[0] - 1).exp() - \
                         self.dist_max[r].value[t]
                
                ceq.append(constr)
        
        return ceq
