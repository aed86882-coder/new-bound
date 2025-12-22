% Load parameters from a file to param_manager.
% It requires the whole workspace to be built in advance, and expinfo contains the correct information.

function vec = LoadParam(filename, K_)
  if nargin < 1
    filename = 'best.mat';
  end
  file = load(filename);
  vec = file.params;

  global param_manager expinfo workspace;
  if length(param_manager.lb) == length(vec)
    % Versions match, just plug in.
    param_manager.SetValue(vec);
  elseif length(param_manager.lb) == length(vec) + 1
    % Loading parameters are old version, need to add K.
    % K is from either the second argument or expinfo.K (the latter applies when obj_mode == 'omega')
    if nargin < 2
      K_ = expinfo.K;
    end
    fprintf('[WARN] Old-version parameters loading. Setting K = %.4f\n', K_);
    K_pos = workspace.GetKPos();
    vec = [vec(1:K_pos-1), K_, vec(K_pos:end)];
    param_manager.SetValue(vec);
  end
end
