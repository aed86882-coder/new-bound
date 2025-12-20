% This class records the IDs of optimizable parameters inside a part,
% together with several static information, like the set of possible splits,
% the joint_to_margin matrices, etc.
% This class only serves for level >= 3, not level 2.

% When the shape (i, j, k) contains zeros, we use another class "PartInfoZero".

% The evaluation consists of 4 steps. The current design is a bit redundant,
% but it may help if multiple high-level parts are pointing to the same low-level part.
%  - EvaluateInit: Initialize the part_frac to be 0.
%  - EvaluatePre: Propagate the part_frac from higher levels to lower levels.
%  - EvaluatePost: Compute the contribution to mat_size and num_block, and store inside the class.
%  - Some entity outside (class Workspace) will collect the contributions and finish the hashing.

% The following linear constraints are registered:
%  - split_dist: sum(split_dist) = 1;
%  - split_dist_max: sum(split_dist_max) = 1;  (redundant constraint)
%  - split_dist and split_dist_max share the same marginals.

classdef PartInfo < matlab.mixin.Copyable
properties
  level
  power
  sum_col
  sum_half
  part_id
  shape
  splits
  num_split
  joint_to_margin

  % Pointers to lower level. Each is a 1 by 3 cell array.
  left_idx
  right_idx
  left_ptr
  right_ptr

  % Optimizable variables. Many are 1 by 3 cell arrays.
  region_prop_id
  region_prop
  split_dist_id  % 1x3 of 1 by num_split
  split_dist
  split_dist_max_id  % the distribution with max entropy
  split_dist_max
  lam_margin_low  % integer array 1 by 3
  lam_margin_high
  lam_margin_id  % 1 by 3 cell array, each cell is 1 by (lam_margin_high(t) - lam_margin_low(t) + 1)
  lam_sum_id  % scalar
  lam_margin
  lam_sum

  % GVars
  mat_size_contribution  % Should be [0, 0, 0]. We only write contributions at (0, j, k) and do not propagate to upper levels.
  num_block_contribution  % 1x3 GVar
  part_frac
  hash_penalty_term
  complete_split
  complete_split_region
  p_comp  % Already multiplied by part_frac. Negative.

  % identifier: in the current design, it equals [id, region] of the parent part.
  % Parts with the same identifiers are merged (heuristically). Currently no parts are merged, so
  % the generated parts are fully consistent with the paper.
  identifier
end
methods
  function obj = PartInfo()
    % Do nothing
  end
  
  function obj = Build(obj, level, part_id, shape, identifier)
    global parts;
    obj.level = level;
    obj.power = 2 ^ (level - 1);
    obj.sum_col = 2 ^ level;
    obj.sum_half = obj.sum_col / 2;
    obj.part_id = part_id;
    obj.shape = shape;
    obj.splits = PrepareSplits(shape);
    obj.num_split = size(obj.splits, 1);
    obj.joint_to_margin = JointToMargin(obj.splits, obj.sum_half);
    obj.identifier = identifier;

    % Lagrange multiplier ranges
    obj.lam_margin_low = ones(1, 3) * Inf;
    obj.lam_margin_high = ones(1, 3) * -Inf;
    for i = 1 : obj.num_split
      for t = 1 : 3
        obj.lam_margin_low(t) = min(obj.lam_margin_low(t), obj.splits(i, t));
        obj.lam_margin_high(t) = max(obj.lam_margin_high(t), obj.splits(i, t));
      end
    end
    
    obj.RegisterVariablesAndLinearConstraints();
    
    % Recursively build low-level parts
    obj.left_idx = cell(1, 3);
    obj.right_idx = cell(1, 3);
    obj.left_ptr = cell(1, 3);
    obj.right_ptr = cell(1, 3);
    for r = 1 : 3  % r is the hashing region. Different regions contain different parts.
      obj.left_idx{r} = zeros(1, obj.num_split);
      obj.right_idx{r} = zeros(1, obj.num_split);
      obj.left_ptr{r} = cell(1, obj.num_split);
      obj.right_ptr{r} = cell(1, obj.num_split);
      for i = 1 : obj.num_split
        % Find/Create the left
        [left_id, left_ptr_cur, is_new] = FindOrCreatePart(level - 1, obj.splits(i, 1:3), [obj.part_id, r]);
        if is_new
          left_ptr_cur.Build(level - 1, left_id, obj.splits(i, 1:3), [obj.part_id, r]);  % The last argument is the identifier.
        end
        % Find/Create the right
        [right_id, right_ptr_cur, is_new] = FindOrCreatePart(level - 1, obj.splits(i, 4:6), [obj.part_id, r]);
        if is_new
          right_ptr_cur.Build(level - 1, right_id, obj.splits(i, 4:6), [obj.part_id, r]);
        end
        % Assign properties
        obj.left_idx{r}(i) = left_id;
        obj.right_idx{r}(i) = right_id;
        obj.left_ptr{r}{i} = left_ptr_cur;
        obj.right_ptr{r}{i} = right_ptr_cur;
      end
    end
  end

  function RegisterVariablesAndLinearConstraints(obj)
    global param_manager;
    % Register distributions
    obj.split_dist_id = cell(1, 3);
    obj.split_dist_max_id = cell(1, 3);
    for r = 1 : 3
      obj.split_dist_id{r} = param_manager.Register(obj.num_split, 0, 1, [0, 1 / obj.num_split]);
      obj.split_dist_max_id{r} = param_manager.Register(obj.num_split, 0, 1, [0, 1 / obj.num_split]);
    end
    % Region proportions
    obj.region_prop_id = param_manager.Register(3, 0, 1, [0, 1]);
    % Lagrange multipliers
    obj.lam_margin_id = cell(3, 3);
    obj.lam_sum_id = cell(1, 3);
    for r = 1 : 3
      for t = 1 : 3
        obj.lam_margin_id{r, t} = param_manager.Register(obj.lam_margin_high(t) - obj.lam_margin_low(t) + 1, -Inf, Inf, [-1 / 100, 1 / 100]);
      end
      obj.lam_sum_id{r} = param_manager.Register(1, -Inf, Inf, [-1 / 100, 1 / 100]);
    end
    
    % Add linear constraints
    for r = 1 : 3
      % sum(split_dist) == 1
      param_manager.AddLinearConstraintEq({obj.split_dist_id{r}, ones(obj.num_split, 1)}, 1);
      % split_dist and split_dist_max share marginals
      for t = 1 : 3
        A = obj.joint_to_margin{t};
        param_manager.AddLinearConstraintEq({obj.split_dist_id{r}, A; obj.split_dist_max_id{r}, -A}, zeros(1, obj.sum_half + 1));
      end
    end
    % sum of region_prop == 1
    param_manager.AddLinearConstraintEq({obj.region_prop_id, ones(3, 1)}, 1);
  end

  % Load distributions from the json file (i.e., Le Gall's parameters for square multiplication)
  function SetInitial(obj, json)
    % json is an array of {shape, dist} pairs.
    global param_manager;
    param_manager.SetSingleParam(obj.region_prop_id, [1/3, 1/3, 1/3]);
    for i = 1 : length(json)
      cur_shape = json(i).shape';
      cur_dist = json(i).dist';
      if isequal(obj.shape, cur_shape)
        for r = 1 : 3
          param_manager.SetSingleParam(obj.split_dist_id{r}, cur_dist);
          param_manager.SetSingleParam(obj.split_dist_max_id{r}, cur_dist);
        end
        % Get lambdas
        [lamX, lamY, lamZ, lamS] = GetLambda(cur_dist, obj.sum_half, obj.splits);
        lam_margins = {lamX, lamY, lamZ};
        for r = 1 : 3
          param_manager.SetSingleParam(obj.lam_sum_id{r}, lamS);
          for t = 1 : 3
            param_manager.SetSingleParam(obj.lam_margin_id{r, t}, lam_margins{t}(obj.lam_margin_low(t) + 1 : obj.lam_margin_high(t) + 1));
          end
        end
        return;
      end
    end
    % If not found, error
    error('Cannot find initial distribution for part %d with shape %d %d %d', obj.part_id, obj.shape(1), obj.shape(2), obj.shape(3));
  end

  function EvaluateInit(obj)
    global param_manager;
    obj.region_prop = param_manager.GetParam(obj.region_prop_id);
    obj.split_dist = cell(1, 3);
    obj.split_dist_max = cell(1, 3);
    obj.lam_sum = cell(1, 3);
    obj.lam_margin = cell(3, 3);
    obj.part_frac = GVar(param_manager.num_input, 0);  % Clear the part_frac value.
    for r = 1 : 3
      obj.split_dist{r} = param_manager.GetParam(obj.split_dist_id{r});
      obj.split_dist_max{r} = param_manager.GetParam(obj.split_dist_max_id{r});
      obj.lam_sum{r} = param_manager.GetParam(obj.lam_sum_id{r});
      for t = 1 : 3
        obj.lam_margin{r, t} = param_manager.GetParam(obj.lam_margin_id{r, t});
      end
    end
  end

  function EvaluatePre(obj)
    % The current part_frac is determined by higher levels.
    % Propagate it to lower levels.
    for i = 1 : obj.num_split
      for r = 1 : 3
        obj.left_ptr{r}{i}.part_frac = obj.left_ptr{r}{i}.part_frac + obj.part_frac * obj.split_dist{r}(i) * obj.region_prop(r);
        obj.right_ptr{r}{i}.part_frac = obj.right_ptr{r}{i}.part_frac + obj.part_frac * obj.split_dist{r}(i) * obj.region_prop(r);
      end
    end
  end

  function EvaluatePost(obj)
    global param_manager;
    % hash_penalty_term
    obj.hash_penalty_term = cell(1, 3);
    for r = 1 : 3
      obj.hash_penalty_term{r} = (obj.split_dist_max{r}.Entropy() - obj.split_dist{r}.Entropy()) * obj.part_frac * obj.region_prop(r);
    end
    
    % num_block
    obj.num_block_contribution = cell(1, 3);
    for r = 1 : 3
      [dist_x, dist_y, dist_z] = MarginalDist(obj.split_dist{r}, obj.joint_to_margin);
      obj.num_block_contribution{r} = obj.part_frac * obj.region_prop(r) * [dist_x.Entropy(), dist_y.Entropy(), dist_z.Entropy()];
    end
    
    % According to new definition, mat_size should be zeros.
    obj.mat_size_contribution = GVar(param_manager.num_input, [0, 0, 0]);

    % complete_split_region{r, 1:3} represents the complete split distribution for X, Y, Z in region r.
    % complete_split{1:3} is the weighted average over three regions.
    obj.complete_split_region = cell(3, 3);
    for r = 1 : 3
      for t = 1 : 3
        obj.complete_split_region{r, t} = GVar.Zeros(1, 3 ^ obj.power);
      end
      for i = 1 : obj.num_split
        for t = 1 : 3
          obj.complete_split_region{r, t} = obj.complete_split_region{r, t} + obj.split_dist{r}(i) * ConcatCSD(obj.left_ptr{r}{i}.complete_split{t}, obj.right_ptr{r}{i}.complete_split{t});
        end
      end
    end
    obj.complete_split = cell(1, 3);
    for t = 1 : 3
      obj.complete_split{t} = obj.complete_split_region{1, t} * obj.region_prop(1) + ...
                              obj.complete_split_region{2, t} * obj.region_prop(2) + ...
                              obj.complete_split_region{3, t} * obj.region_prop(3);
    end
    
    % p_comp
    obj.p_comp = cell(1, 3);
    for r = 1 : 3
      obj.p_comp{r} = (obj.GetPcompNumerator(r) - obj.GetPcompDenominator(r)) * obj.part_frac * obj.region_prop(r);
    end
  end

  function res = GetPcompNumerator(obj, region)
    % Numerator: the number of Z_{\hat K} \in Z_K that are compatible with X_I
    % Be careful which dimension is shared.
    global param_manager;
    res = GVar(param_manager.num_input, 0);
    % Slots for average complete split distribution tilde_alpha(+, +, k')
    weighted_sum_csd = cell(1, obj.sum_half + 1);
    for i = 1 : obj.sum_half + 1
      weighted_sum_csd{i} = GVar.Zeros(1, 3 ^ (obj.power / 2));  % We are focusing on the CSD of lower level.
    end
    prob_sum_csd = GVar.Zeros(1, obj.sum_half + 1);
    used_csd = zeros(1, obj.sum_half + 1);  % If some k' does not appear, we need to avoid 0/0 issue.
    
    for i = 1 : obj.num_split
      ptrs = {obj.left_ptr{region}{i}, obj.right_ptr{region}{i}};
      for left_or_right = 1 : 2
        ptr = ptrs{left_or_right};
        if ptr.shape(region) == 0
          continue;  % 0 has no different split methods and therefore does not contribute to p_comp. H(CSD of this part) == 0.
        end
        if min(ptr.shape) == 0
          % It is a component containing 0. Directly contribute to res.
          res = res + obj.split_dist{region}(i) * ptr.complete_split{region}.Entropy();
        else
          % Help compute average CSD. obj.split_dist{region}(i) fraction of ptr.complete_split_region{region, region} is appearing.
          weighted_sum_csd{ptr.shape(region) + 1} = weighted_sum_csd{ptr.shape(region) + 1} + obj.split_dist{region}(i) * ptr.complete_split{region};
          prob_sum_csd(ptr.shape(region) + 1) = prob_sum_csd(ptr.shape(region) + 1) + obj.split_dist{region}(i);
          used_csd(ptr.shape(region) + 1) = 1;
        end
      end
    end

    % Type-2 contribution.
    for i = 2 : obj.sum_half + 1  % 0 is skipped
      if used_csd(i) == 0
        continue;
      end
      res = res + weighted_sum_csd{i}.NormalizedEntropy(prob_sum_csd(i));
    end
  end

  function res = GetPcompDenominator(obj, region)
    % Denominator: H(CSD) - H(alpha_Z)
    % Be careful which dimension is shared.
    res = obj.complete_split_region{region, region}.Entropy();
    [dist_x, dist_y, dist_z] = MarginalDist(obj.split_dist{region}, obj.joint_to_margin);
    dist_margins = {dist_x, dist_y, dist_z};
    res = res - dist_margins{region}.Entropy();
  end

  function ceq = GetLagrangeConstraints(obj)
    % This function is only called after EvaluateInit
    % For every left-shape (i, j, k), we require lamX(i) + lamY(j) + lamZ(k) + lamSum == log(split_dist_max(i, j, k)) + 1
    ceq = cell(1, 3 * obj.num_split);
    for r = 1 : 3
      for t = 1 : obj.num_split
        left_shape = obj.splits(t, 1 : 3);
        constr = exp(obj.lam_margin{r, 1}(left_shape(1) - obj.lam_margin_low(1) + 1) + ...
                     obj.lam_margin{r, 2}(left_shape(2) - obj.lam_margin_low(2) + 1) + ...
                     obj.lam_margin{r, 3}(left_shape(3) - obj.lam_margin_low(3) + 1) + obj.lam_sum{r} - 1) - obj.split_dist_max{r}(t);
        ceq{(r - 1) * obj.num_split + t} = constr;
      end
    end
  end
end
methods (Static)
  % PartInfoLv2 is used for level-2 constituent tensors.
  % PartInfoZero if used for level-(>= 3) but the shape (i, j, k) contains zeros.
  % The above class PartInfo is used otherwise.
  function obj = CreateInstance(level, shape)
    % This function only selects the right class to create, but does not build the object.
    if level == 2
      obj = PartInfoLv2();
    elseif min(shape) == 0
      obj = PartInfoZero();
    else
      obj = PartInfo();
    end
  end
end
end
