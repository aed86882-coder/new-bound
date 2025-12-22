"""
GVar (Gradient Variable) class for automatic differentiation using forward propagation.

This class maintains:
- num_input: number of input variables
- value: 1D numpy array of values (shape: (n,))
- grad: gradient matrix (shape: (num_input, n))

Supports basic operations: +, -, *, /, element-wise operations, and entropy calculation.
"""

import numpy as np
from scipy import sparse


class GVar:
    """
    Gradient Variable class that tracks values and their gradients.
    
    Attributes:
        num_input (int): Number of input variables
        value (np.ndarray): Value array, shape (n,)
        grad (sparse matrix): Gradient matrix, shape (num_input, n)
    """
    
    def __init__(self, num_input, value=None, grad_type=None, grad_param=None):
        """
        Initialize GVar.
        
        Args:
            num_input: Number of input variables
            value: Initial value(s), scalar or 1D array
            grad_type: Type of gradient initialization ('pos', 'startpos', 'grad', or None)
            grad_param: Parameter for gradient initialization
        """
        self.num_input = num_input
        
        if value is None:
            value = 0.0
        
        # Ensure value is a 1D numpy array
        if np.isscalar(value):
            self.value = np.array([value], dtype=np.float64)
        else:
            self.value = np.atleast_1d(np.array(value, dtype=np.float64))
        
        n = len(self.value)
        self.grad = sparse.csr_matrix((num_input, n), dtype=np.float64)
        
        # Initialize gradient based on grad_type
        if grad_type == 'pos':
            # grad_param is a 1D array of positions
            grad_param = np.atleast_1d(grad_param)
            for i, pos in enumerate(grad_param):
                self.grad[int(pos), i] = 1.0
                
        elif grad_type == 'startpos':
            # grad_param is a scalar starting position
            for i in range(n):
                self.grad[int(grad_param) + i, i] = 1.0
                
        elif grad_type == 'grad':
            # grad_param is the gradient matrix directly
            self.grad = sparse.csr_matrix(grad_param)
    
    def __len__(self):
        """Return length of value array."""
        return len(self.value)
    
    def __getitem__(self, idx):
        """Get item(s) by index."""
        if isinstance(idx, (int, np.integer)):
            return GVar(self.num_input, self.value[idx:idx+1], 'grad', self.grad[:, idx:idx+1])
        else:
            return GVar(self.num_input, self.value[idx], 'grad', self.grad[:, idx])
    
    def __setitem__(self, idx, value):
        """Set item(s) by index."""
        if isinstance(value, GVar):
            self.value[idx] = value.value
            if isinstance(idx, slice):
                self.grad[:, idx] = value.grad
            else:
                self.grad[:, idx] = value.grad.toarray().flatten()
        else:
            self.value[idx] = value
            if isinstance(idx, slice):
                n_elem = len(self.value[idx])
                self.grad[:, idx] = sparse.csr_matrix((self.num_input, n_elem))
            else:
                self.grad[:, idx] = 0.0
    
    def __add__(self, other):
        """Override + operator."""
        if not isinstance(other, GVar):
            other = GVar(self.num_input, other)
        
        new_val = self.value + other.value
        new_grad = self.grad + other.grad
        return GVar(self.num_input, new_val, 'grad', new_grad)
    
    def __radd__(self, other):
        """Override right + operator."""
        return self.__add__(other)
    
    def __sub__(self, other):
        """Override - operator."""
        if not isinstance(other, GVar):
            other = GVar(self.num_input, other)
        
        new_val = self.value - other.value
        new_grad = self.grad - other.grad
        return GVar(self.num_input, new_val, 'grad', new_grad)
    
    def __rsub__(self, other):
        """Override right - operator."""
        if not isinstance(other, GVar):
            other = GVar(self.num_input, other)
        return other.__sub__(self)
    
    def __mul__(self, other):
        """
        Override * operator (element-wise or matrix multiplication).
        Supports scalar, element-wise, and matrix multiplication.
        """
        if not isinstance(other, GVar):
            # other is a scalar or array
            if np.isscalar(other) or (isinstance(other, np.ndarray) and other.size == 1):
                # Scalar multiplication
                new_val = self.value * other
                new_grad = self.grad * other
                return GVar(self.num_input, new_val, 'grad', new_grad)
            elif isinstance(other, np.ndarray) and other.ndim == 2:
                # Matrix multiplication: (1, n) * (n, m) -> (1, m)
                new_val = self.value @ other
                new_grad = self.grad @ other
                return GVar(self.num_input, new_val, 'grad', new_grad)
            else:
                # Element-wise with array
                other_gvar = GVar(self.num_input, other)
                return self.__mul__(other_gvar)
        else:
            # Both are GVars - element-wise multiplication
            new_val = self.value * other.value
            # d(a*b)/dx = da/dx * b + a * db/dx
            new_grad = self.grad.multiply(other.value) + other.grad.multiply(self.value)
            return GVar(self.num_input, new_val, 'grad', new_grad)
    
    def __rmul__(self, other):
        """Override right * operator."""
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """Override / operator (element-wise division)."""
        if not isinstance(other, GVar):
            other = GVar(self.num_input, other)
        
        eps = np.finfo(float).eps
        # Avoid division by zero
        safe_other_val = np.maximum(other.value, eps)
        safe_other_val_sq = np.maximum(other.value ** 2, eps ** 2)
        
        new_val = self.value / safe_other_val
        # d(a/b)/dx = (da/dx * b - a * db/dx) / b^2
        numerator_grad = self.grad.multiply(other.value) - other.grad.multiply(self.value)
        new_grad = numerator_grad.multiply(1.0 / safe_other_val_sq)
        
        return GVar(self.num_input, new_val, 'grad', new_grad)
    
    def __rtruediv__(self, other):
        """Override right / operator."""
        if not isinstance(other, GVar):
            other = GVar(self.num_input, other)
        return other.__truediv__(self)
    
    def __matmul__(self, other):
        """Override @ operator for matrix multiplication."""
        if isinstance(other, np.ndarray):
            # GVar @ matrix
            new_val = self.value @ other
            new_grad = self.grad @ other
            return GVar(self.num_input, new_val, 'grad', new_grad)
        else:
            raise NotImplementedError("@ operator only supports GVar @ ndarray")
    
    def Exp(self):
        """Exponential function."""
        new_val = np.exp(self.value)
        new_grad = self.grad.multiply(new_val)
        return GVar(self.num_input, new_val, 'grad', new_grad)
    
    def Entropy(self):
        """
        Calculate entropy: sum(-value_i * log(value_i))
        Uses natural logarithm (e-based).
        Handles values close to zero gracefully.
        """
        eps_entr = np.finfo(float).eps  # ~1e-16
        
        # Function value: exact even for small values
        safe_val = self.value + (self.value <= 0)  # Replace non-positive with 1 for log
        func_val = -np.sum(self.value * np.log(safe_val))
        
        # Gradient: use eps for values close to zero
        safe_val_grad = np.maximum(self.value, eps_entr)
        grad_coeff = -np.log(safe_val_grad) - 1
        new_grad = self.grad @ grad_coeff
        
        return GVar(self.num_input, func_val, 'grad', new_grad.reshape(-1, 1))
    
    def NormalizedEntropy(self, p):
        """
        Calculate normalized entropy: sum(-value_i * log(value_i / p))
        where p should equal sum(value_i).
        
        Args:
            p: GVar representing the sum of values
        """
        eps_entr = np.finfo(float).eps
        
        if p.value[0] <= 0:
            # Corner case: not differentiable at p=0
            # Use approximation to help escape this point
            n = len(self.value)
            func_val = 0.0
            grad_coeff = np.log(n)
            new_grad = np.sum(self.grad, axis=1) * grad_coeff
            return GVar(self.num_input, func_val, 'grad', new_grad.reshape(-1, 1))
        else:
            # Standard case
            ratio = self.value / p.value[0]
            safe_ratio = ratio + (self.value <= 0)  # Replace non-positive with 1
            func_val = -np.sum(self.value * np.log(safe_ratio))
            
            # Gradient (approximated for values close to zero)
            safe_ratio_grad = np.maximum(ratio, eps_entr)
            grad_coeff = -np.log(safe_ratio_grad)
            new_grad = self.grad @ grad_coeff
            
            return GVar(self.num_input, func_val, 'grad', new_grad.reshape(-1, 1))
    
    @staticmethod
    def Convert(value, num_input=None):
        """
        Convert a scalar or array to GVar with zero gradient.
        Requires global param_manager if num_input is not provided.
        """
        if num_input is None:
            from .ParamManager import get_param_manager
            pm = get_param_manager()
            num_input = pm.num_input
        return GVar(num_input, value)
    
    @staticmethod
    def Zeros(shape, num_input=None):
        """Create a zero GVar with given shape."""
        if num_input is None:
            from .ParamManager import get_param_manager
            pm = get_param_manager()
            num_input = pm.num_input
        if isinstance(shape, int):
            shape = (shape,)
        return GVar(num_input, np.zeros(shape))
    
    def to_array(self):
        """Return value as numpy array."""
        return self.value
    
    def to_scalar(self):
        """Return value as scalar (if length is 1)."""
        if len(self.value) == 1:
            return float(self.value[0])
        else:
            raise ValueError(f"Cannot convert GVar of length {len(self.value)} to scalar")

