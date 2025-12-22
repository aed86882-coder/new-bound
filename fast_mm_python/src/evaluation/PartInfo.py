"""
This class records the IDs of optimizable parameters inside a part,
together with several static information, like the set of possible splits,
the joint_to_margin matrices, etc.
This class only serves for level >= 3, not level 2.

When the shape (i, j, k) contains zeros, we use another class "PartInfoZero".

The evaluation consists of 4 steps:
 - EvaluateInit: Initialize the part_frac to be 0.
 - EvaluatePre: Propagate the part_frac from higher levels to lower levels.
 - EvaluatePost: Compute the contribution to mat_size and num_block, and store inside the class.
 - Some entity outside (class Workspace) will collect the contributions and finish the hashing.

The following linear constraints are registered:
 - split_dist: sum(split_dist) = 1;
 - split_dist_max: sum(split_dist_max) = 1;  (redundant constraint)
 - split_dist and split_dist_max share the same marginals.
"""

import numpy as np
from ..autograd import GVar
from ..utils import PrepareSplits, JointToMargin, MarginalDist
from ..complete_split import ConcatCSD


class PartInfo:
    """
    Main PartInfo class for level >= 3 shapes without zeros.
    Handles recursive construction and constraint evaluation.
    """
    
    def __init__(self):
        """Initialize empty PartInfo."""
        self.level = None
        self.power = None
        self.sum_col = None
        self.sum_half = None
        self.part_id = None
        self.shape = None
        self.splits = None
        self.num_split = None
        self.joint_to_margin = None
        
        # Pointers to lower level (each is a 1x3 list)
        self.left_idx = None
        self.right_idx = None
        self.left_ptr = None
        self.right_ptr = None
        
        # Optimizable variables (many are 1x3 lists)
        self.region_prop_id = None
        self.region_prop = None
        self.split_dist_id = None  # 1x3 list of arrays
        self.split_dist = None
        self.split_dist_max_id = None
        self.split_dist_max = None
        self.lam_margin_low = None  # integer array 1x3
        self.lam_margin_high = None
        self.lam_margin_id = None  # 3x3 list
        self.lam_sum_id = None  # 1x3 list
        self.lam_margin = None
        self.lam_sum = None
        
        # GVars
        self.mat_size_contribution = None
        self.num_block_contribution = None
        self.part_frac = None
        self.hash_penalty_term = None
        self.complete_split = None
        self.complete_split_region = None
        self.p_comp = None
        
        self.identifier = None
    
    def Build(self, level, part_id, shape, identifier, parts=None):
        """
        Build the PartInfo instance and recursively construct lower-level parts.
        
        Args:
            level: Level of this part (>= 3)
            part_id: ID of this part
            shape: Shape tuple (i, j, k)
            identifier: Identifier tuple [parent_id, region]
            parts: Global parts dictionary
        """
        from ..autograd import get_param_manager
        from .FindOrCreatePart import FindOrCreatePart
        
        if parts is None:
            from ..evaluation import get_workspace
            workspace = get_workspace()
            parts = workspace.parts if hasattr(workspace, 'parts') else {}
        
        param_manager = get_param_manager()
        
        self.level = level
        self.power = 2 ** (level - 1)
        self.sum_col = 2 ** level
        self.sum_half = self.sum_col // 2
        self.part_id = part_id
        self.shape = tuple(shape) if not isinstance(shape, tuple) else shape
        self.identifier = tuple(identifier) if not isinstance(identifier, tuple) else identifier
        
        # Get possible splits
        self.splits = PrepareSplits(self.shape)
        self.num_split = self.splits.shape[0]
        self.joint_to_margin = JointToMargin(self.splits, self.sum_half)
        
        # Lagrange multiplier ranges
        self.lam_margin_low = np.full(3, np.inf)
        self.lam_margin_high = np.full(3, -np.inf)
        
        for i in range(self.num_split):
            for t in range(3):
                self.lam_margin_low[t] = min(self.lam_margin_low[t], self.splits[i, t])
                self.lam_margin_high[t] = max(self.lam_margin_high[t], self.splits[i, t])
        
        self.lam_margin_low = self.lam_margin_low.astype(int)
        self.lam_margin_high = self.lam_margin_high.astype(int)
        
        # Register variables and linear constraints
        self.RegisterVariablesAndLinearConstraints()
        
        # Recursively build low-level parts
        self.left_idx = [None, None, None]
        self.right_idx = [None, None, None]
        self.left_ptr = [None, None, None]
        self.right_ptr = [None, None, None]
        
        for r in range(3):  # r is the hashing region
            self.left_idx[r] = np.zeros(self.num_split, dtype=int)
            self.right_idx[r] = np.zeros(self.num_split, dtype=int)
            self.left_ptr[r] = [None] * self.num_split
            self.right_ptr[r] = [None] * self.num_split
            
            for i in range(self.num_split):
                # Find/Create the left part
                left_shape = tuple(self.splits[i, 0:3])
                left_id, left_ptr_cur, is_new = FindOrCreatePart(
                    parts, level - 1, left_shape, (self.part_id, r)
                )
                if is_new:
                    left_ptr_cur.Build(level - 1, left_id, left_shape, (self.part_id, r), parts)
                
                # Find/Create the right part
                right_shape = tuple(self.splits[i, 3:6])
                right_id, right_ptr_cur, is_new = FindOrCreatePart(
                    parts, level - 1, right_shape, (self.part_id, r)
                )
                if is_new:
                    right_ptr_cur.Build(level - 1, right_id, right_shape, (self.part_id, r), parts)
                
                # Assign properties
                self.left_idx[r][i] = left_id
                self.right_idx[r][i] = right_id
                self.left_ptr[r][i] = left_ptr_cur
                self.right_ptr[r][i] = right_ptr_cur
    
    def RegisterVariablesAndLinearConstraints(self):
        """
        Register all optimizable variables and linear constraints.
        """
        from ..autograd import get_param_manager
        param_manager = get_param_manager()
        
        # Register distributions
        self.split_dist_id = [None, None, None]
        self.split_dist_max_id = [None, None, None]
        
        for r in range(3):
            self.split_dist_id[r] = param_manager.Register(
                self.num_split, lb=0, ub=1, initializer=(0, 1.0 / self.num_split)
            )
            self.split_dist_max_id[r] = param_manager.Register(
                self.num_split, lb=0, ub=1, initializer=(0, 1.0 / self.num_split)
            )
        
        # Region proportions
        self.region_prop_id = param_manager.Register(3, lb=0, ub=1, initializer=(0, 1.0))
        
        # Lagrange multipliers
        self.lam_margin_id = [[None] * 3 for _ in range(3)]
        self.lam_sum_id = [None, None, None]
        
        for r in range(3):
            for t in range(3):
                lam_size = self.lam_margin_high[t] - self.lam_margin_low[t] + 1
                self.lam_margin_id[r][t] = param_manager.Register(
                    lam_size, lb=-np.inf, ub=np.inf, initializer=(-0.01, 0.01)
                )
            self.lam_sum_id[r] = param_manager.Register(
                1, lb=-np.inf, ub=np.inf, initializer=(-0.01, 0.01)
            )
        
        # Add linear constraints
        for r in range(3):
            # sum(split_dist) == 1
            param_manager.AddLinearConstraintEq([
                (self.split_dist_id[r], np.ones((1, self.num_split)))
            ], np.array([1.0]))
            
            # split_dist and split_dist_max share marginals
            for t in range(3):
                A = self.joint_to_margin[t]
                if A.shape[1] != self.num_split:
                    continue
                
                margin_size = A.shape[0]
                param_manager.AddLinearConstraintEq([
                    (self.split_dist_id[r], A.T.toarray() if hasattr(A, 'toarray') else A.T),
                    (self.split_dist_max_id[r], -A.T.toarray() if hasattr(A, 'toarray') else -A.T)
                ], np.zeros(margin_size))
        
        # sum(region_prop) == 1
        param_manager.AddLinearConstraintEq([
            (self.region_prop_id, np.ones((1, 3)))
        ], np.array([1.0]))
    
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
        
        # Find matching shape in json
        for item in json:
            cur_shape = tuple(item['shape'])
            cur_dist = np.array(item['dist'])
            
            if cur_shape == self.shape:
                # Set split distributions
                for r in range(3):
                    param_manager.SetSingleParam(self.split_dist_id[r], cur_dist)
                    param_manager.SetSingleParam(self.split_dist_max_id[r], cur_dist)
                
                # TODO: Get lambdas using GetLambda function
                # For now, set to zero
                for r in range(3):
                    param_manager.SetSingleParam(self.lam_sum_id[r], np.array([0.0]))
                    for t in range(3):
                        lam_size = self.lam_margin_high[t] - self.lam_margin_low[t] + 1
                        param_manager.SetSingleParam(self.lam_margin_id[r][t], np.zeros(lam_size))
                
                return
        
        # If not found, use uniform distribution
        print(f'[WARN] Cannot find initial distribution for shape {self.shape}, using uniform')
        uniform_dist = np.ones(self.num_split) / self.num_split
        for r in range(3):
            param_manager.SetSingleParam(self.split_dist_id[r], uniform_dist)
            param_manager.SetSingleParam(self.split_dist_max_id[r], uniform_dist)
    
    def EvaluateInit(self):
        """
        Initialize evaluation: load parameters into GVars and clear part_frac.
        """
        from ..autograd import get_param_manager
        param_manager = get_param_manager()
        
        self.region_prop = param_manager.GetParam(self.region_prop_id)
        
        # Ensure region_prop has 3 elements (safety check)
        if len(self.region_prop.value) < 3:
            # Expand to uniform distribution [1/3, 1/3, 1/3]
            self.region_prop = GVar(param_manager.num_input, [1/3, 1/3, 1/3])
        
        self.split_dist = [None, None, None]
        self.split_dist_max = [None, None, None]
        self.lam_sum = [None, None, None]
        self.lam_margin = [[None] * 3 for _ in range(3)]
        
        self.part_frac = GVar(param_manager.num_input, 0)  # Clear part_frac
        
        for r in range(3):
            self.split_dist[r] = param_manager.GetParam(self.split_dist_id[r])
            self.split_dist_max[r] = param_manager.GetParam(self.split_dist_max_id[r])
            self.lam_sum[r] = param_manager.GetParam(self.lam_sum_id[r])
            
            for t in range(3):
                self.lam_margin[r][t] = param_manager.GetParam(self.lam_margin_id[r][t])
    
    def EvaluatePre(self):
        """
        Pre-evaluation: propagate part_frac from higher levels to lower levels.
        """
        # Safely get region_prop values
        region_prop_vals = self.region_prop.value
        if len(region_prop_vals) < 3:
            region_prop_vals = np.array([1/3, 1/3, 1/3])
        
        for i in range(self.num_split):
            for r in range(3):
                region_val = region_prop_vals[r] if r < len(region_prop_vals) else (1/3)
                frac_left = self.part_frac * self.split_dist[r].value[i] * region_val
                frac_right = self.part_frac * self.split_dist[r].value[i] * region_val
                
                self.left_ptr[r][i].part_frac = self.left_ptr[r][i].part_frac + frac_left
                self.right_ptr[r][i].part_frac = self.right_ptr[r][i].part_frac + frac_right
    
    def EvaluatePost(self):
        """
        Post-evaluation: compute contributions and complete split distributions.
        """
        from ..autograd import get_param_manager
        param_manager = get_param_manager()
        
        # Safely get region_prop values
        region_prop_vals = self.region_prop.value
        if len(region_prop_vals) < 3:
            region_prop_vals = np.array([1/3, 1/3, 1/3])
        
        # hash_penalty_term
        self.hash_penalty_term = [None, None, None]
        for r in range(3):
            entropy_max = self.split_dist_max[r].Entropy()
            entropy_actual = self.split_dist[r].Entropy()
            region_val = region_prop_vals[r] if r < len(region_prop_vals) else (1/3)
            self.hash_penalty_term[r] = (entropy_max - entropy_actual) * \
                                        self.part_frac * region_val
        
        # num_block
        self.num_block_contribution = [None, None, None]
        for r in range(3):
            dist_x, dist_y, dist_z = MarginalDist(self.split_dist[r], self.joint_to_margin)
            entropy_x = dist_x.Entropy()
            entropy_y = dist_y.Entropy()
            entropy_z = dist_z.Entropy()
            
            region_val = region_prop_vals[r] if r < len(region_prop_vals) else (1/3)
            self.num_block_contribution[r] = self.part_frac * region_val * \
                                            GVar(param_manager.num_input, [entropy_x, entropy_y, entropy_z])
        
        # mat_size should be zeros
        self.mat_size_contribution = GVar(param_manager.num_input, [0, 0, 0])
        
        # complete_split_region and complete_split
        self.complete_split_region = [[None] * 3 for _ in range(3)]
        csd_size = 3 ** self.power
        
        for r in range(3):
            for t in range(3):
                self.complete_split_region[r][t] = GVar.Zeros(1, csd_size)
            
            for i in range(self.num_split):
                for t in range(3):
                    left_csd = self.left_ptr[r][i].complete_split[t]
                    right_csd = self.right_ptr[r][i].complete_split[t]
                    concat_csd = ConcatCSD(left_csd, right_csd)
                    
                    self.complete_split_region[r][t] = self.complete_split_region[r][t] + \
                                                       self.split_dist[r].value[i] * concat_csd
        
        self.complete_split = [None, None, None]
        
        for t in range(3):
            region_val_0 = region_prop_vals[0] if 0 < len(region_prop_vals) else (1/3)
            region_val_1 = region_prop_vals[1] if 1 < len(region_prop_vals) else (1/3)
            region_val_2 = region_prop_vals[2] if 2 < len(region_prop_vals) else (1/3)
            
            self.complete_split[t] = self.complete_split_region[0][t] * region_val_0 + \
                                    self.complete_split_region[1][t] * region_val_1 + \
                                    self.complete_split_region[2][t] * region_val_2
        
        # p_comp
        self.p_comp = [None, None, None]
        for r in range(3):
            numerator = self.GetPcompNumerator(r)
            denominator = self.GetPcompDenominator(r)
            region_val = region_prop_vals[r] if r < len(region_prop_vals) else (1/3)
            self.p_comp[r] = (numerator - denominator) * self.part_frac * region_val
    
    def GetPcompNumerator(self, region):
        """
        Compute numerator for p_comp: the number of compatible components.
        
        Args:
            region: Hashing region (0, 1, or 2)
        
        Returns:
            GVar: Numerator value
        """
        from ..autograd import get_param_manager
        param_manager = get_param_manager()
        
        res = GVar(param_manager.num_input, 0)
        
        # Weighted sum of CSDs
        weighted_sum_csd = [GVar.Zeros(1, 3 ** (self.power // 2)) for _ in range(self.sum_half + 1)]
        prob_sum_csd = GVar.Zeros(1, self.sum_half + 1)
        used_csd = np.zeros(self.sum_half + 1, dtype=bool)
        
        for i in range(self.num_split):
            ptrs = [self.left_ptr[region][i], self.right_ptr[region][i]]
            
            for ptr in ptrs:
                if ptr.shape[region] == 0:
                    continue  # Zero dimension doesn't contribute
                
                if min(ptr.shape) == 0:
                    # Contains zero: directly contribute
                    res = res + self.split_dist[region].value[i] * ptr.complete_split[region].Entropy()
                else:
                    # Help compute average CSD
                    idx = ptr.shape[region]
                    weighted_sum_csd[idx] = weighted_sum_csd[idx] + \
                                           self.split_dist[region].value[i] * ptr.complete_split[region]
                    prob_sum_csd.value[idx] += self.split_dist[region].value[i]
                    used_csd[idx] = True
        
        # Type-2 contribution
        for i in range(1, self.sum_half + 1):  # Skip 0
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
        res = self.complete_split_region[region][region].Entropy()
        
        dist_x, dist_y, dist_z = MarginalDist(self.split_dist[region], self.joint_to_margin)
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
            for t in range(self.num_split):
                left_shape = self.splits[t, 0:3]
                
                lam_x_idx = int(left_shape[0] - self.lam_margin_low[0])
                lam_y_idx = int(left_shape[1] - self.lam_margin_low[1])
                lam_z_idx = int(left_shape[2] - self.lam_margin_low[2])
                
                constr = (self.lam_margin[r][0].value[lam_x_idx] +
                         self.lam_margin[r][1].value[lam_y_idx] +
                         self.lam_margin[r][2].value[lam_z_idx] +
                         self.lam_sum[r].value[0] - 1).exp() - \
                         self.split_dist_max[r].value[t]
                
                ceq.append(constr)
        
        return ceq
