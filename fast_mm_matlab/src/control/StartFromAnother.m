% Start from parameters 'xxx.mat' and run the climber controller. Note that we need to adjust the parameters because K is shifted.

function StartFromAnother(q_, K_, exp_name, orig_K, param_path)
  SnInit(q_, K_, exp_name);  % Build the environment
  LoadParam(param_path);  % Load the starting point
  global expinfo param_manager workspace;
  if orig_K < expinfo.K
    % change omega
    omega_id = workspace.omega_id;
    orig_omega = param_manager.GetParam(omega_id);
    orig_omega = orig_omega.value;
    param_manager.SetSingleParam(omega_id, orig_omega * expinfo.K / orig_K);
  end
  workspace.SetInitial();
  SnContinue();
  KeepClimb();
end

%{
Use example:
  AddPath;
  experiment_name = 'K110';
  StartFromAnother(5, 1.10, experiment_name, 1.00, 'data/K100_2.37155181.mat');
%}
