% Start from parameters 'xxx.mat' and run the climber controller for objective 'alpha'.

function StartAlphaFromAnother(q_, exp_name, param_path)
  SnInitAlpha(q_, exp_name);  % Build the environment
  LoadParam(param_path);  % Load the starting point
  global expinfo param_manager workspace;
  SnContinue();
  KeepClimb();
end

%{
Use example:
  AddPath;
  experiment_name = 'Alpha';
  StartAlphaFromAnother(5, experiment_name, 'data/K33_2.00009991.mat');
%}
