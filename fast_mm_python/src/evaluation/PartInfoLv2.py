"""
Same as PartInfo, but only for level 2.
We do not explicitly construct level-1 parts, but instead use closed-forms at level 2.
One parameter is created for 112 and 022 parts.

For all of the components, we let "complete_split{1:3}" represent the complete split distributions
on X, Y, Z dimensions, respectively. Each of them is a 1x9 GVar array.
"""

import numpy as np
from ..autograd import GVar
from ..complete_split import EncodeCSD
from ..utils import Rot3, Rot3c


class PartInfoLv2:
    """
    PartInfo for level 2 constituent tensors.
    Uses closed-form expressions instead of recursive construction.
    """
    
    def __init__(self):
        """Initialize empty PartInfoLv2."""
        self.level = None
        self.power = None
        self.sum_col = None
        self.sum_half = None
        self.part_id = None
        self.shape = None
        self.shape_type = None  # '112', '022', '013', '031', or '004'
        self.rotate_num = None  # Number of left rotations to get standard shape
        
        # Optimizable variables
        self.split_0_id = None  # Probability of splitting into 0+2 or 2+0
        self.split_0 = None
        
        # GVars (outputs)
        self.mat_size_contribution = None  # 1x3 GVar
        self.num_block_contribution = None  # 1x3 GVar
        self.part_frac = None  # Decided by upper levels
        self.hash_penalty_term = None  # Placeholder, always 0
        self.complete_split = None  # 1x3 cell array
        self.p_comp = None  # Placeholder, always 0
        
        self.identifier = None
    
    def Build(self, level, part_id, shape, identifier, parts=None):
        """
        Build the PartInfoLv2 instance.
        
        Args:
            level: Should be 2
            part_id: ID of this part
            shape: Shape tuple (i, j, k)
            identifier: Identifier tuple [parent_id, region]
            parts: Global parts dictionary (not used for level 2)
        """
        if level != 2:
            raise ValueError('PartInfoLv2: level should be 2.')
        
        self.level = level
        self.power = 2 ** (level - 1)
        self.sum_col = 2 ** level
        self.sum_half = self.sum_col // 2
        self.part_id = part_id
        self.shape = tuple(shape) if not isinstance(shape, tuple) else shape
        self.identifier = tuple(identifier) if not isinstance(identifier, tuple) else identifier
        
        # Determine shape type and rotation
        shape_array = np.array(self.shape)
        
        if max(shape_array) == 2 and min(shape_array) == 0:
            self.shape_type = '022'
            standard_shape = np.array([0, 2, 2])
        elif max(shape_array) == 2:
            self.shape_type = '112'
            standard_shape = np.array([1, 1, 2])
        elif max(shape_array) == 3:
            if tuple(shape_array) in [(0, 1, 3), (1, 3, 0), (3, 0, 1)]:
                self.shape_type = '013'
                standard_shape = np.array([0, 1, 3])
            else:
                self.shape_type = '031'
                standard_shape = np.array([0, 3, 1])
        else:
            self.shape_type = '004'
            standard_shape = np.array([0, 0, 4])
        
        # Find rotation number
        self.rotate_num = 0
        current_shape = shape_array.copy()
        while not np.array_equal(current_shape, standard_shape):
            # Rotate to the right
            current_shape = np.array([current_shape[2], current_shape[0], current_shape[1]])
            self.rotate_num += 1
            if self.rotate_num >= 3:
                break
        
        # Register variables for '112' and '022'
        if self.shape_type in ['112', '022']:
            self.RegisterVariables()
        
        # Placeholders
        self.hash_penalty_term = 0
        self.p_comp = 0
    
    def RegisterVariables(self):
        """Register optimizable variables for 112 and 022 shapes."""
        from ..autograd import get_param_manager
        param_manager = get_param_manager()
        self.split_0_id = param_manager.Register(1, lb=0, ub=0.5, initializer=(0, 0.01))
    
    def SetInitial(self, split_0_initial):
        """
        Set initial value for split_0 parameter.
        
        Args:
            split_0_initial: Initial value for split_0
        """
        from ..autograd import get_param_manager
        param_manager = get_param_manager()
        
        if self.shape_type in ['112', '022']:
            param_manager.SetSingleParam(self.split_0_id, split_0_initial)
    
    def EvaluateInit(self):
        """
        Initialize evaluation: load parameters.
        """
        from ..autograd import get_param_manager
        param_manager = get_param_manager()
        
        if self.shape_type in ['112', '022']:
            self.split_0 = param_manager.GetParam(self.split_0_id)
        
        self.part_frac = GVar(param_manager.num_input, 0)
    
    def EvaluatePre(self):
        """
        Pre-evaluation step. Part_frac is already computed by upper levels.
        """
        # Do nothing - part_frac is set by upper levels
        pass
    
    def EvaluatePost(self):
        """
        Post-evaluation: compute contributions and complete split distributions.
        Uses closed-form expressions for level-2 shapes.
        """
        from ..autograd import get_param_manager, GVar
        from ..control import get_q
        
        param_manager = get_param_manager()
        q = get_q()
        
        self.complete_split = [None, None, None]
        
        if self.shape_type == '022':
            # Shape (0, 2, 2): whole thing is an inner product tensor <1, 1, m>
            self.num_block_contribution = GVar(param_manager.num_input, [0, 0, 0])
            
            # Matrix size contribution
            split_val = self.split_0.value[0]
            # Entropy of distribution [split_val, split_val, 1-2*split_val]
            dist_array = np.array([split_val, split_val, 1 - 2 * split_val])
            dist_array = dist_array[dist_array > 0]  # Filter positive values
            entropy_val = -np.sum(dist_array * np.log2(dist_array))
            inner_prod_size = entropy_val + 2 * np.log(q) * (1 - 2 * split_val)
            self.mat_size_contribution = GVar(param_manager.num_input, [0, 0, inner_prod_size])
            
            # Complete split distributions
            self.complete_split[0] = GVar.Zeros(1, 9)
            self.complete_split[0].value[0,EncodeCSD([0, 0]) - 1] = 1 # Add 0
            
            self.complete_split[1] = GVar.Zeros(1, 9)
            idx_02 = EncodeCSD([0, 2]) - 1
            idx_20 = EncodeCSD([2, 0]) - 1
            idx_11 = EncodeCSD([1, 1]) - 1
            self.complete_split[1].value[0,idx_02] = self.split_0.value[0]
            self.complete_split[1].value[0,idx_20] = self.split_0.value[0]
            self.complete_split[1].value[0,idx_11] = 1 - 2 * self.split_0.value[0]
            
            self.complete_split[2] = self.complete_split[1]
        
        elif self.shape_type == '112':
            # Shape (1, 1, 2): laser method
            # num_block
            dist = np.array([self.split_0.value[0], self.split_0.value[0], 1 - 2 * self.split_0.value[0]])
            entropy_val = -np.sum(dist[dist > 0] * np.log2(dist[dist > 0]))
            self.num_block_contribution = GVar(param_manager.num_input, [np.log(2), np.log(2), entropy_val])
            
            # mat_size
            mat_size = 2 * self.split_0.value[0] * np.array([0, np.log(q), 0]) + \
                       (1 - 2 * self.split_0.value[0]) * np.array([np.log(q), 0, np.log(q)])
            self.mat_size_contribution = GVar(param_manager.num_input, mat_size)
            
            # Complete split distributions
            self.complete_split[0] = GVar.Zeros(1, 9)
            idx_01 = EncodeCSD([0, 1]) - 1
            idx_10 = EncodeCSD([1, 0]) - 1
            self.complete_split[0].value[0,idx_01] = 0.5
            self.complete_split[0].value[0,idx_10] = 0.5
            
            self.complete_split[1] = GVar.Zeros(1, 9)
            self.complete_split[1].value[0,idx_01] = 0.5
            self.complete_split[1].value[0,idx_10] = 0.5
            
            self.complete_split[2] = GVar.Zeros(1, 9)
            idx_02 = EncodeCSD([0, 2]) - 1
            idx_20 = EncodeCSD([2, 0]) - 1
            idx_11 = EncodeCSD([1, 1]) - 1
            self.complete_split[2].value[0,idx_02] = self.split_0.value[0]
            self.complete_split[2].value[0,idx_20] = self.split_0.value[0]
            self.complete_split[2].value[0,idx_11] = 1 - 2 * self.split_0.value[0]
        
        elif self.shape_type in ['013', '031']:
            # Inner product tensor
            self.num_block_contribution = GVar(param_manager.num_input, [0, 0, 0])
            inner_prod_size = np.log(2) + np.log(q)
            self.mat_size_contribution = GVar(param_manager.num_input, [0, 0, inner_prod_size])
            
            # Complete split distributions
            self.complete_split[0] = GVar.Zeros(1, 9)
            self.complete_split[0].value[0,EncodeCSD([0, 0]) - 1] = 1
            
            self.complete_split[1] = GVar.Zeros(1, 9)
            self.complete_split[2] = GVar.Zeros(1, 9)
            
            if self.shape_type == '013':
                idx_01 = EncodeCSD([0, 1]) - 1
                idx_10 = EncodeCSD([1, 0]) - 1
                idx_12 = EncodeCSD([1, 2]) - 1
                idx_21 = EncodeCSD([2, 1]) - 1
                self.complete_split[1].value[0,idx_01] = 0.5
                self.complete_split[1].value[0,idx_10] = 0.5
                self.complete_split[2].value[0,idx_12] = 0.5
                self.complete_split[2].value[0,idx_21] = 0.5
            else:  # '031'
                idx_12 = EncodeCSD([1, 2]) - 1
                idx_21 = EncodeCSD([2, 1]) - 1
                idx_01 = EncodeCSD([0, 1]) - 1
                idx_10 = EncodeCSD([1, 0]) - 1
                self.complete_split[1].value[0,idx_12] = 0.5
                self.complete_split[1].value[0,idx_21] = 0.5
                self.complete_split[2].value[0,idx_01] = 0.5
                self.complete_split[2].value[0,idx_10] = 0.5
        
        else:  # '004'
            self.num_block_contribution = GVar(param_manager.num_input, [0, 0, 0])
            self.mat_size_contribution = GVar(param_manager.num_input, [0, 0, 0])
            
            # Complete split distributions
            self.complete_split[0] = GVar.Zeros(1, 9)
            self.complete_split[0].value[0,EncodeCSD([0, 0]) - 1] = 1
            
            self.complete_split[1] = GVar.Zeros(1, 9)
            self.complete_split[1].value[0,EncodeCSD([0, 0]) - 1] = 1
            
            self.complete_split[2] = GVar.Zeros(1, 9)
            self.complete_split[2].value[0, EncodeCSD([2, 2]) - 1] = 1
            #self.complete_split[2].value[EncodeCSD([2, 2]) - 1] = 1
        
        # Rotate and multiply by part_frac
        # Note: Simplified implementation - rotate values directly
        num_block_rotated = Rot3(self.num_block_contribution.value, self.rotate_num) if self.rotate_num > 0 else self.num_block_contribution.value
        mat_size_rotated = Rot3(self.mat_size_contribution.value, self.rotate_num) if self.rotate_num > 0 else self.mat_size_contribution.value
        
        # Multiply by part_frac (scalar)
        part_frac_val = self.part_frac.value[0] if hasattr(self.part_frac.value, '__len__') else self.part_frac.value
        self.num_block_contribution = GVar(param_manager.num_input, num_block_rotated * part_frac_val)
        self.mat_size_contribution = GVar(param_manager.num_input, mat_size_rotated * part_frac_val)
        
        # Rotate complete_split (list of GVars)
        self.complete_split = Rot3c(self.complete_split, self.rotate_num)

