% This function inputs some information of joint distribution
% support space, and output three (n_col) * (sum_col + 1) matrices,
% which are the coefficients to transform the joint distribution
% to the marginal distributions.

function joint_to_margin = JointToMargin(column, sum_col)
  n_col = size(column, 1);
  joint_to_margin = cell(1, 3);
  joint_to_margin{1} = sparse(n_col, sum_col + 1);
  joint_to_margin{2} = sparse(n_col, sum_col + 1);
  joint_to_margin{3} = sparse(n_col, sum_col + 1);
  for i = 1 : n_col
    joint_to_margin{1}(i, column(i, 1) + 1) = 1;
    joint_to_margin{2}(i, column(i, 2) + 1) = 1;
    joint_to_margin{3}(i, column(i, 3) + 1) = 1;
  end
end
