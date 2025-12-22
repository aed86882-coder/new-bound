"""
Parameter Manager for managing all optimizable parameters.

This class registers parameters, manages their bounds, and provides
access to GVar instances for constraint evaluation.
"""

import numpy as np
from scipy import sparse
from .GVar import GVar


# Global parameter manager instance
_param_manager = None


def get_param_manager():
    """Get the global parameter manager instance."""
    global _param_manager
    if _param_manager is None:
        _param_manager = ParamManager()
    return _param_manager


def set_param_manager(pm):
    """Set the global parameter manager instance."""
    global _param_manager
    _param_manager = pm


class ParamManager:
    """
    Manages all optimizable parameters for the optimization problem.
    
    Each parameter group can be a scalar or a vector. The manager
    assigns IDs to parameter groups and provides access through GVar instances.
    """
    
    def __init__(self):
        """Initialize empty parameter manager."""
        self.num_group = 0
        self.num_input = 0
        self.group_startpos = []  # Starting position of each group
        self.group_size = []      # Size of each group
        self.lb = np.array([])    # Lower bounds
        self.ub = np.array([])    # Upper bounds
        self.cur_x = np.array([]) # Current parameter values
        self.initializer = []     # Initializer functions for each group
        
        # Linear constraints: A @ x <= b
        self.lin_A = sparse.csr_matrix((0, 0), dtype=np.float64)
        self.lin_b = np.array([])
        
        # Linear equality constraints: Aeq @ x == beq
        self.lin_Aeq = sparse.csr_matrix((0, 0), dtype=np.float64)
        self.lin_beq = np.array([])
    
    def Register(self, param_len, lb=-np.inf, ub=np.inf, initializer=None):
        """
        Register a new parameter group.
        
        Args:
            param_len: Length of the parameter vector
            lb: Lower bound(s), scalar or array of length param_len
            ub: Upper bound(s), scalar or array of length param_len
            initializer: Initialization function or tuple (low, high) for uniform random
        
        Returns:
            group_id: ID of the registered parameter group
        """
        # Handle scalar bounds
        if np.isscalar(lb):
            lb = np.full(param_len, lb, dtype=np.float64)
        else:
            lb = np.array(lb, dtype=np.float64)
            
        if np.isscalar(ub):
            ub = np.full(param_len, ub, dtype=np.float64)
        else:
            ub = np.array(ub, dtype=np.float64)
        
        # Assign group ID
        group_id = self.num_group
        self.num_group += 1
        self.group_startpos.append(self.num_input)
        self.group_size.append(param_len)
        
        # Update bounds
        self.lb = np.concatenate([self.lb, lb])
        self.ub = np.concatenate([self.ub, ub])
        self.num_input += param_len
        
        # Set initializer
        if initializer is None:
            initializer = (0.0, 0.01)
        
        if callable(initializer):
            self.initializer.append(initializer)
        else:
            # Assume it's a tuple (low, high) for uniform random
            low, high = initializer
            self.initializer.append(lambda sz: np.random.uniform(low, high, sz))
        
        return group_id
    
    def SetValue(self, x):
        """
        Set current parameter values, clipped to bounds.
        
        Args:
            x: Parameter vector of length num_input
        """
        x = np.array(x, dtype=np.float64)
        self.cur_x = np.clip(x, self.lb, self.ub)
    
    def SetSingleParam(self, group_id, value):
        """
        Set the value for a single parameter group.
        
        Args:
            group_id: ID of the parameter group
            value: New value(s) for the group
        """
        start = self.group_startpos[group_id]
        end = start + self.group_size[group_id]
        expected_size = self.group_size[group_id]
        
        # Convert value to array and check size
        value_array = np.atleast_1d(value)
        if len(value_array) != expected_size:
            print(f"[WARN] SetSingleParam: group_id={group_id}, expected size={expected_size}, got size={len(value_array)}")
            # Pad or truncate as needed
            if len(value_array) < expected_size:
                value_array = np.pad(value_array, (0, expected_size - len(value_array)), constant_values=value_array[0])
            else:
                value_array = value_array[:expected_size]
        
        self.cur_x[start:end] = value_array
    
    def RandomInit(self):
        """
        Randomly initialize all parameters using their initializers.
        
        Returns:
            x: Randomly initialized parameter vector
        """
        x = np.zeros(self.num_input)
        for i in range(self.num_group):
            start = self.group_startpos[i]
            end = start + self.group_size[i]
            x[start:end] = self.initializer[i](self.group_size[i])
        
        self.cur_x = x
        return x
    
    def GetParam(self, group_id):
        """
        Get a GVar for a parameter group.
        
        Args:
            group_id: ID of the parameter group
        
        Returns:
            GVar instance with current values and appropriate gradients
        """
        start = self.group_startpos[group_id]
        end = start + self.group_size[group_id]
        param_size = self.group_size[group_id]
        
        # Ensure cur_x is initialized
        if len(self.cur_x) < self.num_input:
            # Initialize cur_x if not done yet
            self.cur_x = np.zeros(self.num_input)
        
        value = self.cur_x[start:end]
        
        # Double check we got the right size
        if len(value) != param_size:
            # This shouldn't happen, but if it does, return zeros
            value = np.zeros(param_size)
        
        return GVar(self.num_input, value, 'startpos', start)
    
    def PackResults(self, y_gvar_list):
        """
        Pack a list of GVars into constraint values and gradients.
        
        Args:
            y_gvar_list: List of GVar instances representing constraints
        
        Returns:
            y: Constraint values, shape (num_constraints,)
            dy: Constraint gradients, shape (num_constraints, num_input)
        """
        y_list = []
        dy_list = []
        
        for gvar in y_gvar_list:
            y_list.append(gvar.value)
            # Transpose: grad is (num_input, n) -> we want (n, num_input)
            dy_list.append(gvar.grad.T)
        
        # Concatenate all constraints
        if len(y_list) > 0:
            y = np.concatenate(y_list)
            dy = sparse.vstack(dy_list)
        else:
            y = np.array([])
            dy = sparse.csr_matrix((0, self.num_input))
        
        return y, dy
    
    def AddLinearConstraint(self, A_entries, b):
        """
        Add linear inequality constraints: A @ x <= b
        
        Args:
            A_entries: List of tuples (group_id, coeff_matrix)
                      where coeff_matrix has shape (n_constraints, group_size)
            b: Right-hand side vector of length n_constraints
        """
        b = np.atleast_1d(b)
        n_constraints = len(b)
        
        # Expand constraint matrix
        row_start = self.lin_A.shape[0]
        new_rows = n_constraints
        
        # Create new matrix with expanded size
        new_A = sparse.lil_matrix((row_start + new_rows, self.num_input))
        if row_start > 0:
            new_A[:row_start, :self.lin_A.shape[1]] = self.lin_A
        
        # Fill in new constraints
        for group_id, coeff_matrix in A_entries:
            start = self.group_startpos[group_id]
            end = start + self.group_size[group_id]
            new_A[row_start:row_start + new_rows, start:end] = coeff_matrix
        
        self.lin_A = new_A.tocsr()
        self.lin_b = np.concatenate([self.lin_b, b])
    
    def AddLinearConstraintEq(self, A_entries, b):
        """
        Add linear equality constraints: Aeq @ x == beq
        
        Args:
            A_entries: List of tuples (group_id, coeff_matrix)
            b: Right-hand side vector
        """
        b = np.atleast_1d(b)
        n_constraints = len(b)
        
        row_start = self.lin_Aeq.shape[0]
        new_rows = n_constraints
        
        new_Aeq = sparse.lil_matrix((row_start + new_rows, self.num_input))
        if row_start > 0:
            new_Aeq[:row_start, :self.lin_Aeq.shape[1]] = self.lin_Aeq
        
        for group_id, coeff_matrix in A_entries:
            start = self.group_startpos[group_id]
            end = start + self.group_size[group_id]
            new_Aeq[row_start:row_start + new_rows, start:end] = coeff_matrix
        
        self.lin_Aeq = new_Aeq.tocsr()
        self.lin_beq = np.concatenate([self.lin_beq, b])
    
    def GetLinearConstraints(self):
        """
        Get all linear constraints in standard form.
        
        Returns:
            A, b, Aeq, beq: Constraint matrices and vectors
        """
        # Ensure proper dimensions
        if self.lin_A.shape[0] > 0 and self.lin_A.shape[1] < self.num_input:
            new_A = sparse.lil_matrix((self.lin_A.shape[0], self.num_input))
            new_A[:, :self.lin_A.shape[1]] = self.lin_A
            self.lin_A = new_A.tocsr()
        
        if self.lin_Aeq.shape[0] > 0 and self.lin_Aeq.shape[1] < self.num_input:
            new_Aeq = sparse.lil_matrix((self.lin_Aeq.shape[0], self.num_input))
            new_Aeq[:, :self.lin_Aeq.shape[1]] = self.lin_Aeq
            self.lin_Aeq = new_Aeq.tocsr()
        
        # Remove empty rows
        if self.lin_A.shape[0] > 0:
            non_empty = np.array(self.lin_A.sum(axis=1)).flatten() != 0
            A = self.lin_A[non_empty, :]
            b = self.lin_b[non_empty]
        else:
            A = self.lin_A
            b = self.lin_b
        
        if self.lin_Aeq.shape[0] > 0:
            non_empty = np.array(self.lin_Aeq.sum(axis=1)).flatten() != 0
            Aeq = self.lin_Aeq[non_empty, :]
            beq = self.lin_beq[non_empty]
        else:
            Aeq = self.lin_Aeq
            beq = self.lin_beq
        
        # Save back cleaned versions
        self.lin_A = A
        self.lin_b = b
        self.lin_Aeq = Aeq
        self.lin_beq = beq
        
        return A, b, Aeq, beq
    
    def Perturb(self, perturb_size):
        """
        Add random perturbation to current parameters.
        
        Args:
            perturb_size: Standard deviation of Gaussian noise
        """
        noise = np.random.normal(0, perturb_size, self.num_input)
        self.cur_x = np.clip(self.cur_x + noise, self.lb, self.ub)

