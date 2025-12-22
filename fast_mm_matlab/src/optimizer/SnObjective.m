% This function evaluates the objective function of the optimization problem, as well as its gradient.
% The objective depends on expinfo.obj_mode:
% - 'omega': minimize omega.
% - 'alpha': maximize K.
% - 'mu': minimize K under omega(K) <= 1 + 2 * K.
% This function should be fast, so we do not evaluate anything or build GVars.
% Instead, we just read the corresponding entry from the parameter vector x.

function [f, df] = SnObjective(x)
  global workspace expinfo;
  df = sparse(length(x), 1);
  if strcmp(expinfo.obj_mode, 'omega')
    omega_pos = workspace.GetOmegaPos();
    f = x(omega_pos);
    df(omega_pos, 1) = 1;
  elseif strcmp(expinfo.obj_mode, 'alpha')
    K_pos = workspace.GetKPos();
    f = -x(K_pos);
    df(K_pos, 1) = -1;
  elseif strcmp(expinfo.obj_mode, 'mu')
    K_pos = workspace.GetKPos();
    f = x(K_pos);
    df(K_pos, 1) = 1;
  else
    error('Unknown objective mode.');
  end
  f = full(f);
  df = full(df);
end
