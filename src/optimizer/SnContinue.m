% Before calling this function, we require the following variables to be set:
%  - workspace, q, expinfo, param_manager, max_level
% This function will continue with the current values in the param_manager.
% When calling the nonlinear-constraint evaluator, the best known solution is
% automatically stored in "experiment_name/best.mat".
% This function SnContinue() is NOT guaranteed to exit normally. The user may
% want to manually stop it by writing a single "1" to "experiment_name/manual_stop.txt".

function SnContinue()
  global workspace param_manager expinfo;

  fid = fopen(strcat(expinfo.exp_name, '/manual_stop.txt'), 'w');  % Clear the manual stop signal.
  fprintf(fid, '0\n');
  fclose(fid);

  % Read the lower & upper bounds & linear constraints.
  lb = param_manager.lb';
  ub = param_manager.ub';
  [lin_A, lin_b, lin_Aeq, lin_beq] = param_manager.GetLinearConstraints();

  % SNOPT has bugs about sparse vectors, so we convert all vectors to full versions.
  lin_b = full(lin_b);
  lin_beq = full(lin_beq);

  % Function handles for evaluating the objective & nonlinear constraints.
  objective = @SnObjective;
  nonlcon = @SnNonlcon;

  % Options are mainly specified by the ".spc" file. Which ".spc" file to use depends on the mode.
  options = struct;
  options.screen = 'on';
  options.printfile = strcat(expinfo.exp_name, '/sn_output.out');
  if expinfo.is_refine_mode
    options.specsfile = 'sn_options_for_refine.spc';
  else
    options.specsfile = 'sn_options.spc';
  end
  options.stop = @CheckAbort;

  % Get gradient structure.
  x_start = param_manager.cur_x';
  [iGfun, jGvar] = GetMatG();
  options.iGfun = iGfun;
  options.jGvar = jGvar;

  global sn_output;
  [x, fmin, INFO, output, lambda, states] = snsolve_hack(objective, x_start, lin_A, lin_b, lin_Aeq, lin_beq, lb, ub, nonlcon, options);
  sn_output = struct;
  sn_output.x = x;
  sn_output.fmin = fmin;
  sn_output.INFO = INFO;
  sn_output.output = output;
  sn_output.lambda = lambda;
  sn_output.states = states;

  fprintf("SNOPT Completed. fmin = %.12f\n", fmin);
  param_manager.SetValue(x');
  workspace.Evaluate();
  % If exited non-manually, save the final parameters to "experiment_name/sn_param.mat".
  SaveParam(strcat(expinfo.exp_name, '/sn_param.mat'));
end
