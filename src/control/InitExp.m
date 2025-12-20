% This function only initializes the experiment environment, but do not start the experiment.
% It initializes the properties obj_mode and is_refine_mode in expinfo, as well as K, if provided.

function InitExp(exp_name, obj_mode, is_refine_mode, K)
  global expinfo;
  expinfo = ExpInfo();
  expinfo.exp_name = exp_name;
  expinfo.best_feasibility = Inf;
  expinfo.best_feasible_obj = Inf;
  expinfo.best_semifeasible_obj = Inf;
  expinfo.last_update_or_climb_start_time = tic();
  expinfo.obj_mode = obj_mode;
  expinfo.is_refine_mode = is_refine_mode;
  if nargin >= 4
    expinfo.K = K;
  end

  % Create experiment directory
  % Do not clean up! We want to keep the backup files (if any).
  if ~exist(exp_name, 'dir')
    mkdir(exp_name);
  end

  % Create or clean up files
  % manual_stop.txt: write a single line 0
  fid = fopen(strcat(exp_name, '/manual_stop.txt'), 'w');
  fprintf(fid, '0\n');
  fclose(fid);

  % Log file: clean up
  fid = fopen(strcat(exp_name, '/log.txt'), 'w');
  fclose(fid);
end
