# Create a class named 'GVar'. It maintains a scalar (or a row vector) of variables with gradient information. Inside the class, we store:
#  - num_input: the number of input variables. It decides the size of the gradient matrix.
#  - value: the value of the variable. It is 1 * n for an arbitrary n >= 1.
#  - grad: the gradient of the variable. It is num_input * n.
# When initializing the class, we need to specify the number of input variables and maybe the value of the
#   variable. If the latter is not specified, we will initialize it to [0]. Note that we also need to know
#   the size (n) of the variable, but it is already included in the size of the value.
# Support basic operations: +, -, *, .*, ./, Entropy (e-based). The gradient is automatically computed.

# Known issues:
#  - ./ operator is not accurate when the denominator is below eps (around 1e-16).
#  - The extreme case handling in Entropy() will probably slow down the program.
#  - Entropy() is the e-based entropy instead of 2.
#  - No longer support log() or log2(). Not needed in the current implementation.

# Note:
#  - The corner case handling hyperparameters (like eps) cannot be too extreme (like 1e100),
#      otherwise the optimizer does not work. Default value of eps == 1e-16 works well.
#  - In some of the functions we used approximated or inaccurate partial derivatives,
#      but we tried the best to make the function value precise, so the correctness of the verification
#      program still holds.
import numpy as np
from scipy import sparse

class GVar:
    # Properties: num_input, value, grad
    num_input: int
    value: np.ndarray
    grad: sparse.csr_matrix

    # Constructor 1: GVar(num_input). value = 0, grad = zero vector.
    # Constructor 2: GVar(num_input, value). grad = zero vector/matrix.
    # Constructor 3: GVar(num_input, value, 'pos'/'startpos'/'grad', param). grad is generated in three ways:
    #   - 'pos': param is 1 * n vector. Gradient of the i-th entry equals to double(1:num_input == param(i))'.
    #   - 'startpos': param is a scalar. Gradient of the i-th entry equals to double(1:num_input == param + i - 1)'.
    #   - 'grad': param is num_input * n matrix. Gradient equals param.
    def __init__(self, num_input, value=None, grad_type=None, grad_param=None):
        if value is None:
            value = 0.0
        self.num_input = num_input
        # Ensure value is a 1D numpy array
        if np.isscalar(value):
            self.value = np.array([value], dtype=np.float64)
        else:
            self.value = np.atleast_1d(np.array(value, dtype=np.float64))
        self.grad = sparse.csr_matrix((num_input, len(self.value)), dtype=np.float64)
        if grad_type is not None:
            if grad_type == 'pos':
                grad_param = np.atleast_1d(grad_param)
                for i, pos in enumerate(grad_param):
                    self.grad[int(pos), i] = 1.0
            elif grad_type == 'startpos':
                for i in range(len(self.value)):
                    self.grad[int(grad_param) + i, i] = 1.0
            elif grad_type == 'grad':
                self.grad = sparse.csr_matrix(grad_param)
            else:
                raise ValueError('Unknown grad_type')

    # Override the assign operator. When assigning a scalar/row vector, we will assign the value and set the
    #   gradient to zero; when assigning a GVar, we will assign the value and the gradient.
    # a[i] = b: assign scalar to an entry. b can be a scalar or a GVar of length 1.
    # a[:] = b: assign row vector to the whole vector. We also change the size.
    # a[i:j] = b: assign row vector to a subvector.
    def __setitem__(self, idx, B):
        if isinstance(B, GVar):
            B_val = B.value
            B_grad = B.grad
        else:
            B_val = B
            if isinstance(idx, slice):
                n_elem = len(self.value[idx])
                B_grad = sparse.csr_matrix((self.num_input, n_elem))
            else:
                B_grad = 0.0

        # if idx == ':' (slice(None))
        if isinstance(idx, slice) and idx == slice(None):
            self.value = np.atleast_1d(np.array(B_val, dtype=np.float64))
            self.grad = sparse.csr_matrix(B_grad)
        else:
            self.value[idx] = B_val
            if isinstance(idx, slice):
                self.grad[:, idx] = B_grad
            elif isinstance(B, GVar):
                self.grad[:, idx] = B_grad.toarray().flatten()
            else:
                self.grad[:, idx] = B_grad

    # Override the subsref operator. Supported cases:
    # a[i]: get the i-th entry. Return a scalar GVar.
    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return GVar(self.num_input, self.value[idx:idx+1], 'grad', self.grad[:, idx:idx+1])
        else:
            return GVar(self.num_input, self.value[idx], 'grad', self.grad[:, idx])

    # Override the horzcat operator. In Python, use numpy.concatenate or GVar.horzcat() instead.
    @staticmethod
    def horzcat(*args):
        """Concatenate multiple GVars horizontally: GVar.horzcat(a, b, c, ...)"""
        result = args[0]
        for i in range(1, len(args)):
            other = args[i]
            # If both are non-GVars
            if not isinstance(result, GVar) and not isinstance(other, GVar):
                result = np.concatenate([np.atleast_1d(result), np.atleast_1d(other)])
            else:
                # Convert both to GVar
                if not isinstance(result, GVar):
                    result = GVar(other.num_input, result)
                if not isinstance(other, GVar):
                    other = GVar(result.num_input, other)
                new_value = np.concatenate([result.value, other.value])
                new_grad = sparse.hstack([result.grad, other.grad])
                result = GVar(result.num_input, new_value, 'grad', new_grad)
        return result

    # Override the plus operator. Both sides can be GVar or scalar/row vector.
    def __add__(self, other):
        if not isinstance(other, GVar):
            other = GVar(self.num_input, other)
        return GVar(self.num_input, self.value + other.value, 'grad', self.grad + other.grad)

    def __radd__(self, other):
        return self.__add__(other)

    # Override the minus operator. Both sides can be GVar or scalar/row vector.
    def __sub__(self, other):
        if not isinstance(other, GVar):
            other = GVar(self.num_input, other)
        return GVar(self.num_input, self.value - other.value, 'grad', self.grad - other.grad)

    def __rsub__(self, other):
        if not isinstance(other, GVar):
            other = GVar(self.num_input, other)
        return other.__sub__(self)

    # Override the .* operator (element-wise multiplication). Both sides can be GVar or scalar/row vector.
    # Supported cases: (1 by n) * (1 by n), (1 by n) * scalar, and scalar * (1 by n).
    def __mul__(self, other):
        if not isinstance(other, GVar):
            other = GVar(self.num_input, other)
        new_val = self.value * other.value
        new_grad = self.grad.multiply(other.value) + other.grad.multiply(self.value)
        return GVar(self.num_input, new_val, 'grad', new_grad)
        # Note: numpy arrays support (n,) * (n,), broadcasting works similarly to MATLAB.

    def __rmul__(self, other):
        return self.__mul__(other)

    # Override the ./ operator (element-wise division). Both sides can be GVar or scalar/row vector.
    def __truediv__(self, other):
        if not isinstance(other, GVar):
            other = GVar(self.num_input, other)
        starting_point = np.finfo(float).eps
        safe_denom = np.maximum(other.value, starting_point)
        safe_denom_sq = np.maximum(other.value ** 2, starting_point ** 2)
        new_val = self.value / safe_denom
        new_grad = (self.grad.multiply(other.value) - other.grad.multiply(self.value)).multiply(1.0 / safe_denom_sq)
        return GVar(self.num_input, new_val, 'grad', new_grad)

    def __rtruediv__(self, other):
        if not isinstance(other, GVar):
            other = GVar(self.num_input, other)
        return other.__truediv__(self)

    # Override the * operator for matrix multiplication. Supported cases for a * b:
    # one of a and b is a scalar, and the other is a GVar: same as element-wise;
    # a is a GVar (1 by n), and b is a matrix (n by m): return a GVar of 1 by m.
    def __matmul__(self, other):
        # See if scalar case applies
        if np.isscalar(other) or (isinstance(other, np.ndarray) and other.size == 1):
            return self.__mul__(other)
        # The matrix case: GVar @ matrix
        if not isinstance(other, GVar):
            # The second parameter (matrix) does not have gradients.
            return GVar(self.num_input, self.value @ other, 'grad', self.grad @ other)
        raise NotImplementedError("@ operator only supports GVar @ ndarray")

    # Override the exp operator.
    def Exp(self):
        return GVar(self.num_input, np.exp(self.value), 'grad', self.grad.multiply(np.exp(self.value)))

    # Entropy(a): sum of -a_i .* ln(a_i).
    # If some entry a_i is less than eps (a predetermined hyperparameter),
    # we take log(eps) instead of log(a_i) to avoid infinite derivative.
    # Even for entries close to zero, the function value (instead of the derivative) is always accurate.
    def Entropy(self):
        eps_entr = np.finfo(float).eps  # roughly 1e-16
        func_val = np.sum(-self.value * np.log(self.value + (self.value <= 0)))
        grad_coeff = -np.log(np.maximum(self.value, eps_entr)) - 1
        return GVar(self.num_input, func_val, 'grad', (self.grad @ grad_coeff).reshape(-1, 1))

    # NormalizedEntropy(a, p): returns p * Entropy(a / p). 'a' is a vector with sum p, while p can be very small.
    # It equals the sum of -a_i .* ln(a_i / p).
    # p should always be sum(a_i), but we still pass in this parameter to avoid repetitive calculations (and hence errors).
    # Again, when any entry is close to zero, the function value is accurate
    # but the derivative will compute log(eps) instead of the precise formula.
    # This is for avoiding NaN or Inf in the derivative.
    def NormalizedEntropy(self, p):
        eps_entr = np.finfo(float).eps
        if p.value[0] <= 0:
            # Corner case, special derivatives.
            # Actually, at the point p == 0 (thus all a_i == 0), the function NormalizedEntropy() is not differentiable.
            # In this case, we overestimate the partial derivatives in order to help escape this point (if possible).
            n = len(self.value)
            return GVar(self.num_input, 0.0, 'grad', (np.sum(self.grad, axis=1) * np.log(n)).reshape(-1, 1))
        else:
            ratio = self.value / p.value[0]
            func_val = np.sum(-self.value * np.log(ratio + (self.value <= 0)))
            grad_coeff = -np.log(np.maximum(ratio, eps_entr))  # Inaccurate when self.value is close to zero.
            return GVar(self.num_input, func_val, 'grad', (self.grad @ grad_coeff).reshape(-1, 1))

    # Override size() function. Return the shape of the value.
    def shape(self, dim=None):
        if dim is None:
            return self.value.shape
        else:
            return self.value.shape[dim]

    # Override length() function. Return the length of the value.
    def __len__(self):
        return len(self.value)

# Static methods
    # We can use GVar.Convert([1, 2, 3]) to initialize a GVar with zero gradient.
    @staticmethod
    def Convert(value, num_input=None):
        if num_input is None:
            from .ParamManager import get_param_manager
            num_input = get_param_manager().num_input
        return GVar(num_input, value)

    # We can use GVar.Zeros(100) to initialize a 1 by 100 zero vector (with zero gradient).
    @staticmethod
    def Zeros(size, num_input=None):
        if num_input is None:
            from .ParamManager import get_param_manager
            num_input = get_param_manager().num_input
        return GVar(num_input, np.zeros(size))
