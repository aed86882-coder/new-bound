% Start with the current best solution and try to improve it.
% Should be called if the previous run is terminated due to numerical difficulties.
% The current best parameter is always stored in 'best.mat'.

function ClimbStep()
  global expinfo param_manager;
  exp_name = expinfo.exp_name;
  expinfo.SetClock();
  file_name = strcat(exp_name, '/best.mat');
  if ~exist(file_name, 'file')
    fprintf('[ERR] best.mat does not exist\n');
    % Use the current values in param_manager
  else
    LoadParam(file_name);
  end

  % Important: Evaluate once. This is to make sure that the best objective recorded in expinfo is no worse than the actual best objective.
  SnNonlcon(param_manager.cur_x');

  global sn_output;

  if ~isempty(sn_output) && isfield(sn_output, 'INFO') && sn_output.INFO ~= 32
    % Perturb a little bit. If INFO == 32, it means the previous round is terminated due to iteration limit. Do not perturb.
    param_manager.Perturb(1e-10);
    WriteLog('Perturb 1e-10');
  end

  % Run the experiment
  SnContinue();  % If improved, the new best parameter is automatically saved to 'best.mat'.

  WriteLog(sprintf('ClimbStep finished. SNOPT INFO = %d', sn_output.INFO));
end
