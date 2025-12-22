% Initialize CVX. This needs to be called in every session before using CVX.
% MOSEK is a commercial solver for CVX that is more efficient than the default solver SDPT3.
% However, using SDPT3 is also enough for this project, because we only use CVX when setting
% the initial point (which is not costly).

function InitCVX(mosek_path)
  if nargin < 1
    mosek_path = 'C:\Program Files\Mosek\10.0\toolbox\r2017a';
  end
  addpath(mosek_path);
  cvx_setup;
  fprintf('Init finished.\n');
end
