% Evaluate the max violation of constraints for the current parameters.
% The workspace should be built and the parameters need to be loaded to param_manager before calling.

function maxViol = GetFeasibility()
  global param_manager expinfo workspace;

  % Nonlinear Constraints
  [gc, gceq] = workspace.Evaluate();  % gc, gceq are cell-arrays of GVars
  [c, ~] = param_manager.PackResults(gc);
  [ceq, ~] = param_manager.PackResults(gceq);

  % Linear Constraints
  [A, b, Aeq, beq] = param_manager.GetLinearConstraints();
  maxViol = max(max(c), max(abs(ceq)));  % >= 0.
  if ~isempty(A)
    maxViol = max(maxViol, max(A * param_manager.cur_x' - b));
  end
  if ~isempty(Aeq)
    maxViol = max(maxViol, max(abs(Aeq * param_manager.cur_x' - beq)));
  end

  % Check if x is between lb and ub. If the parameters are correct, this should always be true.
  lb = param_manager.lb;  % row vector
  ub = param_manager.ub;
  maxViol = max(maxViol, max(lb - param_manager.cur_x));
  maxViol = max(maxViol, max(param_manager.cur_x - ub));

  % If objective mode == 'alpha', especially check if omega == 2.
  % This additional check is because some precision error is allowed during optimization.
  if strcmp(expinfo.obj_mode, 'alpha')
    omega = param_manager.cur_x(workspace.GetOmegaPos());
    maxViol = max(maxViol, omega - 2);
  end
  % Other two objectives do not require additional check because the constraints used for optimization
  % are already accurate.
end
