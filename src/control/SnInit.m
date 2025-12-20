% This function is called to initialize the environment for SNOPT.
% After this function is called, the parameters are set to random values.
% Then, we can call 'SnContinue' to run the optimization, or further load the parameters.
% Notice: This is the initialization function for 'omega' mode.

function SnInit(q_, K_, exp_name)
  global workspace param_manager max_level q;
  InitSNOPT;
  q = q_;
  max_level = 3;
  InitExp(exp_name, 'omega', false, K_);

  workspace = Workspace();
  workspace.Build(max_level);
  param_manager.RandomInit();
end
