% Initialization procedure for the 'alpha' objective.

function SnInitAlpha(q_, exp_name)
  global workspace param_manager max_level q;
  InitSNOPT;
  q = q_;
  max_level = 3;
  InitExp(exp_name, 'alpha', false);
  
  workspace = Workspace();
  workspace.Build(max_level);
  param_manager.RandomInit();
end
