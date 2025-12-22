% Use concave maximization toolbox CVX to obtain the max-entropy distribution given its marginals.

function dist_max = GetMaxDist(dist, joint_to_margin)
  [dist_x, dist_y, dist_z] = MarginalDist(dist, joint_to_margin);
  cvx_begin quiet
    variables dist_max(1, length(dist));
    [ax, ay, az] = MarginalDist(dist_max, joint_to_margin);
    maximize sum(entr(dist_max))
    subject to
      sum(dist_max) == 1;
      0 <= dist_max <= 1;
      ax == dist_x;
      ay == dist_y;
      az == dist_z;
  cvx_end
  dist_max = max(dist_max, 0);
  dist_max = dist_max / sum(dist_max);
end
