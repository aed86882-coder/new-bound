% SnRefine is the variant of SnContinue which runs under the "refining mode".
% It sets expinfo.is_refine_mode and clear the best known solution (the latter is because the best
% known solution might be obtained in non-refining mode, which has lower precision requirement).

function SnRefine()
  global expinfo;
  expinfo.is_refine_mode = true;
  expinfo.best_feasible_obj = inf;
  SnContinue();
end
