% This function is to evaluate all nonlinear constraints and their gradients.
% The output c is a column vector, where every entry c(t) indicates a constraint c(t) <= 0.
% ceq is similar, where every entry ceq(t) indicates ceq(t) == 0.
% dc and dceq are gradient matrices, where each row dc(t, :) indicates the partial derivatives
% of c(t) with respect to every optimizable variable (there are param_manager.num_input many).
% dceq is similar.
% The input x is a column vector.
% The end-hook's behavior depends on expinfo.is_refine_mode: in the refining mode,
% precision requirement is higher.
% There are few constraints that depends on expinfo.obj_mode, which are handled in Workspace,
% e.g., in the 'alpha' mode we require omega == 2, etc.

function [c, ceq, dc_, dceq_] = SnNonlcon(x)
  global param_manager workspace;
  param_manager.SetValue(x');
  [gc, gceq] = workspace.Evaluate();  % gc, gceq are cell arrays of GVar
  [c, dc] = param_manager.PackResults(gc);
  [ceq, dceq] = param_manager.PackResults(gceq);
  c = full(c);
  ceq = full(ceq);
  if nargout > 2
    dc_ = dc;
    dceq_ = dceq;
  end
  SnNonlconEndHook(x, c, ceq);
end
