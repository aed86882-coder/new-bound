% Start from parameters 'xxx.mat' and run the climber controller for objective 'mu'.

function StartMuFromAnother(q_, exp_name, param_path)
  SnInitMu(q_, exp_name);  % Build the environment
  LoadParam(param_path);  % Load the starting point
  global expinfo param_manager workspace;
  SnContinue();
  KeepClimb();
end

%{
Use example:
  AddPath;
  experiment_name = 'Mu';
  StartMuFromAnother(5, experiment_name, 'data/K55_2.06613393.mat');
%}
