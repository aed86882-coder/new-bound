% This function reads data from a file and verify if omega(1, 1, K) <= 1 + 2 * K.
% (mu is defined as the solution of omega(1, 1, K) == 1 + 2 * K. It is not difficult to see that
%  once the above inequality holds, the K forms an upper bound of mu.)

function VerifyMu(q_, datapath)
  global workspace param_manager max_level q;
  q = q_;
  max_level = 3;
  InitExp('__verify', 'mu', false);
  workspace = Workspace();
  workspace.Build(max_level);
  LoadParam(datapath);

  maxViol = GetFeasibility();  % Get the max violation of all constraints.
  K = param_manager.cur_x(workspace.GetKPos());
  % omega = param_manager.cur_x(workspace.GetOmegaPos());

  fprintf("mu <= %.8f \t(MaxViolation: %e)\n", K, maxViol);
  if maxViol > 1.1e-6
    fprintf("[WARN] The last result seems wrong (the MaxViolation is too large).\n");
  elseif maxViol > 1.1e-9
    fprintf("[WARN] The last result is not very accurate (MaxViolation > 1e-9).\n");
  end
end
