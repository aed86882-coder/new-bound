"""
This is similar to PartInfo for level >= 3, but specialized for (i, j, k) containing zeros.
In this case, many variables are not used. We do not register for them.
"""

import numpy as np
from ..autograd import GVar
from ..complete_split import EncodeCSD, DecodeCSD


class PartInfoZero:
    """
    PartInfo for shapes containing zeros at level >= 3.
    Simplified compared to regular PartInfo.
    """
    
    def __init__(self):
        """Initialize empty PartInfoZero."""
        self.level = None
        self.power = None
        self.sum_col = None
        self.sum_half = None
        self.part_id = None
        self.shape = None
        self.zero_dim = None  # Which dimension is zero (1, 2, or 3)
        self.nonzero_dim_1 = None
        self.nonzero_dim_2 = None
        
        # Optimizable vars
        self.complete_split = None  # Output 1x3 list of CSD
        self.complete_split_id = None  # 1x3 list of parameter IDs
        
        # Outputs
        self.base_mat_size = None  # [1, 0, 0] or its permutations
        self.mat_size_contribution = None  # 1x3 GVar
        self.num_block_contribution = None  # Always [0, 0, 0]
        self.part_frac = None
        self.hash_penalty_term = None  # Always 0
        self.p_comp = None  # Always 0
        
        self.identifier = None
    
    def Build(self, level, part_id, shape, identifier, parts=None):
        """
        Build the PartInfoZero instance.
        
        Args:
            level: Level (>= 3)
            part_id: ID of this part
            shape: Shape tuple (i, j, k) with at least one zero
            identifier: Identifier tuple [parent_id, region]
            parts: Global parts dictionary (not used for zero shapes)
        """
        from ..autograd import get_param_manager
        param_manager = get_param_manager()
        
        self.level = level
        self.power = 2 ** (level - 1)
        self.sum_col = 2 ** level
        self.sum_half = self.sum_col // 2
        self.part_id = part_id
        self.shape = tuple(shape) if not isinstance(shape, tuple) else shape
        self.identifier = tuple(identifier) if not isinstance(identifier, tuple) else identifier
        
        # Determine which dimension is zero
        shape_array = np.array(self.shape)
        if shape_array[0] == 0:
            self.zero_dim = 0
            self.nonzero_dim_1 = 1
            self.nonzero_dim_2 = 2
            self.base_mat_size = np.array([0, 0, 1])
        elif shape_array[1] == 0:
            self.zero_dim = 1
            self.nonzero_dim_1 = 0
            self.nonzero_dim_2 = 2
            self.base_mat_size = np.array([1, 0, 0])
        else:  # shape_array[2] == 0
            self.zero_dim = 2
            self.nonzero_dim_1 = 0
            self.nonzero_dim_2 = 1
            self.base_mat_size = np.array([0, 1, 0])
        
        # Complete split distributions
        csd_size = 3 ** self.power
        self.complete_split_id = [None, None, None]
        
        for t in range(3):
            # -1 represents always zero, -2 represents always one
            self.complete_split_id[t] = np.full(csd_size, -1, dtype=int)
        
        # Zero dimension is always (0, 0, ..., 0) -> encoded as 1
        self.complete_split_id[self.zero_dim][0] = -2  # Always one
        
        # For nonzero dimensions, register parameters
        param_id_map = {}  # Map from CSD id to parameter id
        
        for csd_id in range(1, csd_size + 1):
            arr = DecodeCSD(csd_id, self.power)
            
            # Check if this CSD matches the nonzero dimension's constraint
            if np.sum(arr) != shape_array[self.nonzero_dim_1]:
                continue
            
            # Register parameter for this CSD
            param_id = param_manager.Register(1, lb=0, ub=1, initializer=(0, 0.01))
            self.complete_split_id[self.nonzero_dim_1][csd_id - 1] = param_id
            
            # The opposite dimension has the complementary CSD
            arr_opposite = 2 * np.ones(self.power, dtype=int) - arr
            csd_id_opposite = EncodeCSD(arr_opposite)
            self.complete_split_id[self.nonzero_dim_2][csd_id_opposite - 1] = param_id
        
        # Register linear constraint: sum of complete split distribution = 1
        lincon_entries = []
        for csd_id in range(csd_size):
            param_id = self.complete_split_id[self.nonzero_dim_1][csd_id]
            if param_id >= 0:  # Not -1 or -2
                lincon_entries.append((param_id, np.array([[1.0]])))
        
        if lincon_entries:
            param_manager.AddLinearConstraintEq(lincon_entries, np.array([1.0]))
        
        # Placeholder variables
        self.num_block_contribution = [np.array([0, 0, 0])] * 3
        self.hash_penalty_term = [0, 0, 0]
        self.p_comp = [0, 0, 0]
    
    def SetInitial(self, json=None):
        """
        Set initial parameters using heuristics.
        We simply choose the CSD that maximizes the number of entries in the result.
        
        Note: This is a simplified version without CVX optimization.
        """
        from ..autograd import get_param_manager
        from ..control import get_q
        
        param_manager = get_param_manager()
        q = get_q()
        
        # Simple heuristic: uniform distribution over valid CSDs
        csd_size = 3 ** self.power
        valid_csds = []
        valid_param_ids = []
        
        for csd_id in range(csd_size):
            param_id = self.complete_split_id[self.nonzero_dim_1][csd_id]
            if param_id >= 0:
                valid_csds.append(csd_id + 1)
                valid_param_ids.append(param_id)
        
        if valid_param_ids:
            # Uniform distribution
            uniform_prob = 1.0 / len(valid_param_ids)
            for param_id in valid_param_ids:
                param_manager.SetSingleParam(param_id, np.array([uniform_prob]))
    
    def EvaluateInit(self):
        """
        Initialize evaluation: load parameters into GVars.
        """
        from ..autograd import get_param_manager
        param_manager = get_param_manager()
        
        self.part_frac = GVar(param_manager.num_input, 0)
        self.complete_split = [None, None, None]
        
        csd_size = 3 ** self.power
        for t in range(3):
            self.complete_split[t] = GVar.Zeros(csd_size)
            
            for csd_id in range(csd_size):
                param_id = self.complete_split_id[t][csd_id]
                
                if param_id == -1:
                    # Always zero
                    self.complete_split[t].value[csd_id] = 0
                elif param_id == -2:
                    # Always one
                    self.complete_split[t].value[csd_id] = 1
                else:
                    # Optimizable parameter
                    param_gvar = param_manager.GetParam(param_id)
                    self.complete_split[t].value[csd_id] = param_gvar.value[0]
                    # Copy gradient information
                    self.complete_split[t].grad[:, csd_id] = param_gvar.grad[:, 0]
    
    def EvaluatePre(self):
        """
        Pre-evaluation step. We do not split, so do not propagate the fraction.
        """
        # Do nothing
        pass
    
    def EvaluatePost(self):
        """
        Post-evaluation: compute matrix size contribution.
        Should equal 2^entropy * q^(number of ones), summed over CSD.
        """
        from ..control import get_q
        q = get_q()
        
        # Compute inner product size
        inner_prod_size = self.complete_split[self.nonzero_dim_1].Entropy()
        
        csd_size = 3 ** self.power
        for csd_id in range(1, csd_size + 1):
            arr = DecodeCSD(csd_id, self.power)
            num_ones = np.sum(arr == 1)
            
            csd_prob = self.complete_split[self.nonzero_dim_1].value[csd_id - 1]
            if csd_prob > 0:
                inner_prod_size = inner_prod_size + csd_prob * num_ones * np.log(q)
        
        # Combine with part_frac and base_mat_size
        self.mat_size_contribution = self.part_frac * inner_prod_size * GVar(
            self.part_frac.num_input, self.base_mat_size
        )

