% This function is called at the end of SnNonlcon.

function SnNonlconEndHook(x, c, ceq)
  % compute obj and maxViol
  obj = SnObjective(x);
  global param_manager;
  A = param_manager.lin_A;
  b = param_manager.lin_b;
  Aeq = param_manager.lin_Aeq;
  beq = param_manager.lin_beq;
  maxViol = max(max(c), max(abs(ceq)));  % >= 0.
  if ~isempty(A)
    maxViol = max(maxViol, max(A*x - b));
  end
  if ~isempty(Aeq)
    maxViol = max(maxViol, max(abs(Aeq*x - beq)));
  end

  global expinfo;
  exp_name = expinfo.exp_name;

  % Fix obj with maxViol.
  % Below, we will implement the following logic: Suppose the best 'obj' given that 'maxViol < 1e-6'
  % was not improved within an hour, we stop the optimization process and wait for the caller
  % to take further actions (maybe perturb and restart).
  % However, consider we obtain a solution by directly decreasing omega by 1e-6, then it is still
  % within our tolerance, but this 'obj' would be much better than what we actually have.
  % Meanwhile, the optimizer focuses on seeking for a true feasible and better solution, which might
  % be slow and cannot improve 1e-6 within an hour. As a result, the program would be killed.
  % To prevent this, we add a small heuristic fix on obj, letting such ill-posed solutions have a
  % penalty proportional to 'maxViol'.
  obj = obj + maxViol;  % Maybe we can multiply by a coefficient, e.g., maxViol * 2.

  if ~expinfo.is_refine_mode
    % Normal mode has lower precision requirement.
    if maxViol < expinfo.best_feasibility
      expinfo.best_feasibility = maxViol;
      WriteLog(sprintf('[INFO] New best feasibility: %.8f', maxViol));
      expinfo.SetClock();
    end
    if maxViol < 1e-6 && obj < expinfo.best_feasible_obj
      last_best = expinfo.best_feasible_obj;
      expinfo.best_feasible_obj = obj;
      WriteLog(sprintf('[INFO] New best solution: %.8f', obj));
      % Save the best solution.
      SaveParam(strcat(exp_name, '/best.mat'), x);
      expinfo.SetClock();
    end
    if maxViol < 1e-3 && obj < expinfo.best_semifeasible_obj
      expinfo.best_semifeasible_obj = obj;
      WriteLog(sprintf('[INFO] New best semi-feasible solution: %.8f', obj));
      expinfo.SetClock();  % Prevents too-early termination at the very beginning.
    end
  else
    % Refine mode seek for high-precision solutions.
    if maxViol < expinfo.best_feasibility
      expinfo.best_feasibility = maxViol;
      WriteLog(sprintf('[INFO] New best feasibility: %.16f', maxViol));
      expinfo.SetClock();
    end
    goal_eps = 1e-9;
    if maxViol < goal_eps && obj < expinfo.best_feasible_obj
      expinfo.best_feasible_obj = obj;
      WriteLog(sprintf('[INFO] New best solution: %.16f. Feasibility: %.16f', obj, maxViol));
      % Save the best solution.
      SaveParam(strcat(exp_name, '/best.mat'), x);
      expinfo.SetClock();
    end
  end
end
