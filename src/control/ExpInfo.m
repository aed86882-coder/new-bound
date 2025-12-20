% Global class storing experiment hyperparameters and other information.

classdef ExpInfo < handle
properties
  exp_name

  % Control the Climbing Process
  best_feasible_obj
  best_semifeasible_obj
  best_feasibility
  last_update_or_climb_start_time

  % Mode Control
  obj_mode        % string, 'omega' or 'alpha' or 'mu'. Each mode will disable some of the optimizable variables.
  is_refine_mode  % boolean, decides the behavior of end_hook and spc file to be used.
  K               % only enabled when obj_mode == 'omega'.
end
methods
  function SetClock(obj)
    obj.last_update_or_climb_start_time = tic();
  end

  function t = TimeElapsed(obj)
    t = toc(obj.last_update_or_climb_start_time);
  end
end
end
