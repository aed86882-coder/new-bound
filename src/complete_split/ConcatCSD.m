% ConcatCSD: Concatenate two Complete Split Distributions.
% Complete split distributions are represented with GVars of length 3 ^ (2 ^ (level - 1)).

function res = ConcatCSD(lhs, rhs)
  res_value = kron(lhs.value, rhs.value);
  res_grad = kron(lhs.grad, rhs.value) + kron(lhs.value, rhs.grad);
  res = GVar(lhs.num_input, res_value, 'grad', res_grad);
end
