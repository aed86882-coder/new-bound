% Given a distribution alpha_max and the support of the joint distribution that has the maximum
% entropy among all distributions that share the same marginals, we want to get the Lagrange
% multipliers that satisfy:
% lamX(i) + lamY(j) + lamZ(k) + lamSum == ln(alpha_max(i, j, k)) + 1, for all (i, j, k) in the support.

function [lamX, lamY, lamZ, lamSum] = GetLambda(alph, sum_col, column)
  % Solve the equations
  num_col = size(column, 1);
  num_var = 3 * (sum_col + 1) + 1;
  A = zeros(num_col, num_var);
  b = zeros(num_col, 1);
  for t = 1 : num_col
    i = column(t, 1);
    j = column(t, 2);
    k = column(t, 3);
    A(t, i + 1) = 1;
    A(t, j + 1 + (sum_col + 1)) = 1;
    A(t, k + 1 + 2 * (sum_col + 1)) = 1;
    A(t, 1 + 3 * (sum_col + 1)) = 1;
    b(t) = log(alph(t)) + 1;
  end
  % X = linsolve(A, b);
  X = lsqr(A, b);  % We can use the least-squares method when the input does not have high precision.
  lamX = X(1 : sum_col + 1)';
  lamY = X(sum_col + 2 : 2 * sum_col + 2)';
  lamZ = X(2 * sum_col + 3 : 3 * sum_col + 3)';
  lamSum = X(3 * sum_col + 4);
end
