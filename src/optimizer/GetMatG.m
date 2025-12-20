% G is a matrix used in SNOPT optimizer that indicates which functions have nonzero gradients
% with respect to which variables.
% We randomly choose 5 points in the solution space, evaluate G matrices,
% and report the union of their nonzero entries.
% The return value is two row vectors such that (iGfun(t), jGvar(t)) indicates the t-th nonzero term.

function [iGfun, jGvar] = GetMatG()
  global param_manager;
  param_backup = param_manager.cur_x;  % Backup the values before calling this function.
  for t = 1 : 5
    param_manager.RandomInit();
    [~, ~, dc, dceq] = SnNonlcon(param_manager.cur_x');
    [~, df] = SnObjective(param_manager.cur_x');
    G = [df'; dc; dceq];
    if t == 1
      G_sum = G ~= 0;
    else
      G_sum = G_sum + (G ~= 0);
    end
  end
  [iGfun, jGvar, ~] = find(G_sum);
  param_manager.cur_x = param_backup;  % Restore the backup values.
end
