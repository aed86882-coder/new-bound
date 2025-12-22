% This class is similar to PartInfo. It stores the ID and GVar of optimizable variables in the global stage.
% Also, hash penalty term is taken into account.

classdef GlobalStage < handle
properties
  level
  power
  sum_col
  shapes
  num_shape
  joint_to_margin

  % Pointers to higest-level parts. 1 by 3 cell arrays.
  part_id
  part_ptr

  % Optimizable variables
  region_prop_id
  region_prop  % 1 by 3 GVar array, sum up to 1
  % Below are parameters within each region. Everything is a cell array of 1 by 3 containing desired information.
  dist_id
  dist
  dist_max_id
  dist_max
  lam_margin_id  % 1 by 3 cell array, each cell is 1 by sum_col + 1
  lam_sum_id  % scalar
  lam_margin
  lam_sum

  % GVars
  mat_size
  num_block  % The number of max-level blocks in the global stage hashing
  hash_penalty_term
  p_comp
  complete_split
end
methods
  function obj = GlobalStage()
    % Do nothing
  end

  function obj = Build(obj)
    global parts max_level;
    obj.level = max_level;
    obj.power = 2 ^ (obj.level - 1);
    obj.sum_col = 2 ^ obj.level;
    obj.shapes = PrepareShapes(obj.level);
    obj.num_shape = size(obj.shapes, 1);
    obj.joint_to_margin = JointToMargin(obj.shapes, obj.sum_col);
    obj.RegisterVariablesAndLinearConstraints();

    % Recursively build the parts
    obj.part_id = cell(1, 3);
    obj.part_ptr = cell(1, 3);
    for r = 1 : 3
      obj.part_id{r} = zeros(1, obj.num_shape);
      obj.part_ptr{r} = cell(1, obj.num_shape);
      for i = 1 : obj.num_shape
        cur_id = length(parts{obj.level}) + 1;
        parts{obj.level}{cur_id} = PartInfo.CreateInstance(obj.level, obj.shapes(i, :));
        ptr_cur = parts{obj.level}{cur_id}.Build(obj.level, cur_id, obj.shapes(i, :), [0, r]);
        obj.part_id{r}(i) = cur_id;
        obj.part_ptr{r}{i} = ptr_cur;
      end
    end
  end

  function RegisterVariablesAndLinearConstraints(obj)
    global param_manager expinfo;
    % Register variables.
    % region_prop
    obj.region_prop_id = param_manager.Register(3, 0, 1, [0, 1]);
    % dist and dist_max
    obj.dist_id = cell(1, 3);
    obj.dist_max_id = cell(1, 3);
    for r = 1 : 3
      obj.dist_id{r} = param_manager.Register(obj.num_shape, 0, 1, [0, 1 / obj.num_shape]);
      obj.dist_max_id{r} = param_manager.Register(obj.num_shape, 0, 1, [0, 1 / obj.num_shape]);
    end
    % Lagrange multipliers
    obj.lam_margin_id = cell(1, 3);
    obj.lam_sum_id = cell(1, 3);
    for r = 1 : 3
      obj.lam_margin_id{r} = cell(1, 3);
      for t = 1 : 3
        obj.lam_margin_id{r}{t} = param_manager.Register(obj.sum_col + 1, -Inf, Inf, [-1 / 100, 1 / 100]);
      end
      obj.lam_sum_id{r} = param_manager.Register(1, -Inf, Inf, [-1 / 100, 1 / 100]);
    end

    % Add linear constraints
    % sum(region_prop) == 1
    param_manager.AddLinearConstraintEq({obj.region_prop_id, ones(3, 1)}, 1);
    for r = 1 : 3
      % sum(dist) == 1
      param_manager.AddLinearConstraintEq({obj.dist_id{r}, ones(obj.num_shape, 1)}, 1);
      % dist and dist_max share marginals
      for t = 1 : 3
        A = obj.joint_to_margin{t};
        param_manager.AddLinearConstraintEq({obj.dist_id{r}, A; obj.dist_max_id{r}, -A}, zeros(1, obj.sum_col + 1));
      end
    end

    % Y and Z are symmetric
    param_manager.AddLinearConstraintEq({obj.region_prop_id, [0; 1; -1]}, 0);
    % If K == 1, X, Y, Z are all symmetric
    if strcmp(expinfo.obj_mode, 'omega') && expinfo.K == 1
      param_manager.AddLinearConstraintEq({obj.region_prop_id, [1; -1; 0]}, 0);
    end
  end

  % Load the distributions from the json file (i.e., Le Gall's parameters)
  function SetInitial(obj, json)
    global param_manager;
    param_manager.SetSingleParam(obj.region_prop_id, [1/3, 1/3, 1/3]);
    for i = 1 : length(json)
      cur_shape = json(i).shape';
      cur_dist = json(i).dist';
      if isequal(cur_shape, [0, 0, 0])
        for r = 1 : 3
          param_manager.SetSingleParam(obj.dist_id{r}, cur_dist);
          param_manager.SetSingleParam(obj.dist_max_id{r}, cur_dist);
        end
        % Get lambdas
        [lamX, lamY, lamZ, lamS] = GetLambda(cur_dist, obj.sum_col, obj.shapes);
        lam_margins = {lamX, lamY, lamZ};
        for r = 1 : 3
          param_manager.SetSingleParam(obj.lam_sum_id{r}, lamS);
          for t = 1 : 3
            param_manager.SetSingleParam(obj.lam_margin_id{r}{t}, lam_margins{t});
          end
        end
        return;
      end
    end
    % If not found, error
    error('Cannot find initial distribution for global stage');
  end

  function EvaluateInit(obj)
    % Load parameter values
    global param_manager;
    obj.region_prop = param_manager.GetParam(obj.region_prop_id);
    obj.dist = cell(1, 3);
    obj.dist_max = cell(1, 3);
    obj.lam_margin = cell(1, 3);
    obj.lam_sum = cell(1, 3);
    for r = 1 : 3
      obj.dist{r} = param_manager.GetParam(obj.dist_id{r});
      obj.dist_max{r} = param_manager.GetParam(obj.dist_max_id{r});
      obj.lam_margin{r} = cell(1, 3);
      for t = 1 : 3
        obj.lam_margin{r}{t} = param_manager.GetParam(obj.lam_margin_id{r}{t});
      end
      obj.lam_sum{r} = param_manager.GetParam(obj.lam_sum_id{r});
    end
  end

  function EvaluatePre(obj)
    % Propagate part_frac to the max level
    for r = 1 : 3
      for i = 1 : obj.num_shape
        obj.part_ptr{r}{i}.part_frac = obj.dist{r}(i) * obj.region_prop(r);  % Keeping the gradient information
      end
    end
  end

  function EvaluatePost(obj)
    global param_manager parts;
    % hash_penalty_term. Count three regions separately.
    obj.hash_penalty_term = cell(1, 3);
    for r = 1 : 3
      obj.hash_penalty_term{r} = (obj.dist_max{r}.Entropy() - obj.dist{r}.Entropy()) * obj.region_prop(r);
    end
    % num_block. Count three regions separately.
    obj.num_block = cell(1, 3);
    for r = 1 : 3
      [dist_x, dist_y, dist_z] = MarginalDist(obj.dist{r}, obj.joint_to_margin);
      obj.num_block{r} = [dist_x.Entropy(), dist_y.Entropy(), dist_z.Entropy()] * obj.region_prop(r);
    end
    % mat_size: sum over all parts of all levels
    obj.mat_size = GVar(param_manager.num_input, [0, 0, 0]);
    for l = 2 : obj.level
      for i = 1 : length(parts{l})
        obj.mat_size = obj.mat_size + parts{l}{i}.mat_size_contribution;
      end
    end
    % complete_split
    obj.complete_split = cell(1, 3);
    for r = 1 : 3
      obj.complete_split{r} = cell(1, 3);
      for t = 1 : 3
        obj.complete_split{r}{t} = GVar.Zeros(1, 3 ^ obj.power);
        for i = 1 : obj.num_shape
          ptr = obj.part_ptr{r}{i};
          obj.complete_split{r}{t} = obj.complete_split{r}{t} + obj.dist{r}(i) * ptr.complete_split{t};
        end
      end
    end
    % p_comp. Count three regions separately.
    obj.p_comp = cell(1, 3);
    for r = 1 : 3
      obj.p_comp{r} = (obj.GetPcompNumerator(r) - obj.GetPcompDenominator(r)) * obj.region_prop(r);
    end
  end

  function res = GetPcompNumerator(obj, region)
    % Numerator: the number of Z_{\hat K} \in Z_K that are compatible with X_I
    % Be careful which dimension is shared.
    global param_manager;
    res = GVar(param_manager.num_input, 0);
    % Slots for average complete split distribution tilde_alpha(+, +, k)
    weighted_sum_csd = cell(1, obj.sum_col + 1);
    for i = 1 : obj.sum_col + 1
      weighted_sum_csd{i} = GVar.Zeros(1, 3 ^ obj.power);
    end
    prob_sum_csd = GVar.Zeros(1, obj.sum_col + 1);
    used_csd = zeros(1, obj.sum_col + 1);  % If some k does not appear, we need to avoid 0/0 issue.

    for i = 1 : obj.num_shape
      ptr = obj.part_ptr{region}{i};
      if ptr.shape(region) == 0
        continue;  % 0 has no different split methods and therefore does not contribute to p_comp. H(CSD of this part) == 0.
      end
      if min(ptr.shape) == 0
        % It is a component containing 0. Directly contribute to res.
        res = res + obj.dist{region}(i) * ptr.complete_split{region}.Entropy();
      else
        % Help compute average CSD. obj.dist(i) fraction of ptr.complete_split{region} is appearing.
        weighted_sum_csd{ptr.shape(region) + 1} = weighted_sum_csd{ptr.shape(region) + 1} + obj.dist{region}(i) * ptr.complete_split{region};
        prob_sum_csd(ptr.shape(region) + 1) = prob_sum_csd(ptr.shape(region) + 1) + obj.dist{region}(i);
        used_csd(ptr.shape(region) + 1) = 1;
      end
    end

    % Compute the average CSDs and the type-2 contributions.
    for i = 2 : obj.sum_col + 1  % 0 is skipped
      if used_csd(i) == 0
        continue;
      end
      % Method 1: correct but bad precision.
      % - weighted_sum_csd{i} = weighted_sum_csd{i} ./ prob_sum_csd(i);
      % - res = res + prob_sum_csd(i) * weighted_sum_csd{i}.Entropy();
      % Method 2
      res = res + weighted_sum_csd{i}.NormalizedEntropy(prob_sum_csd(i));
    end
  end

  function res = GetPcompDenominator(obj, region)
    % Denominator: H(CSD) - H(alpha_Z)
    % Be careful which dimension is shared.
    res = obj.complete_split{region}{region}.Entropy();  % Observe the shared dimension.
    [dist_x, dist_y, dist_z] = MarginalDist(obj.dist{region}, obj.joint_to_margin);
    dist_margins = {dist_x, dist_y, dist_z};
    res = res - dist_margins{region}.Entropy();
  end

  function ceq = GetLagrangeConstraints(obj)
    % This function is only called after EvaluateInit
    % For every shape (i, j, k), we require lamX(i) + lamY(j) + lamZ(k) + lamSum == log(dist_max(i, j, k)) + 1
    ceq = cell(1, 3 * obj.num_shape);
    for r = 1 : 3
      for t = 1 : obj.num_shape
        cur_shape = obj.shapes(t, :);
        constr = exp(obj.lam_margin{r}{1}(cur_shape(1) + 1) + ...
                     obj.lam_margin{r}{2}(cur_shape(2) + 1) + ...
                     obj.lam_margin{r}{3}(cur_shape(3) + 1) + obj.lam_sum{r} - 1) - obj.dist_max{r}(t);
        ceq{(r - 1) * obj.num_shape + t} = constr;
      end
    end
  end
end
end
