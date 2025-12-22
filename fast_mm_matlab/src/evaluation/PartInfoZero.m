% This is similar to PartInfo for level >= 3, but specialized for (i, j, k) containing zeros.
% In this case, many variables are not used. We do not register for them.
% A static method in PartInfo will decide which class to create, PartInfo or PartInfoZero, or even PartInfoLv2.

classdef PartInfoZero < matlab.mixin.Copyable
properties
  level
  power
  sum_col
  sum_half
  part_id
  shape
  zero_dim  % Which dimension is zero.
  nonzero_dim_1
  nonzero_dim_2

  % Optimizable vars.
  complete_split  % Output 1x3 cell array of CSD. We give individual IDs to the optimizable ones within them.
  complete_split_id  % 1x3 cell array of scalar row vectors of IDs in param_manager. -1 represents "always zero". -2 represents "always one". Different elements can share the same ID meaning that they are always equal.

  % Outputs.
  % num_var_in_comp
  base_mat_size  % [1, 0, 0] or its permutations, depends on which dimension is zero.
  mat_size_contribution  % 1x3 GVar
  num_block_contribution  % Always 1x3 of [0, 0, 0]. It should not be used.
  part_frac
  hash_penalty_term  % Always 1x3 cell array of 0. It should not be used.
  p_comp  % Always 1x3 cell array of 0.

  identifier
end
methods
  function obj = PartInfoZero()
    % Do nothing.
  end

  % Override Build: do not recursively build low-level parts.
  function obj = Build(obj, level, part_id, shape, identifier)
    global param_manager;
    obj.level = level;
    obj.power = 2 ^ (level - 1);
    obj.sum_col = 2 ^ level;
    obj.sum_half = obj.sum_col / 2;
    obj.part_id = part_id;
    obj.shape = shape;
    obj.identifier = identifier;

    % Rename the dimensions and store information.
    if shape(1) == 0
      obj.zero_dim = 1;
      obj.nonzero_dim_1 = 2;
      obj.nonzero_dim_2 = 3;
      obj.base_mat_size = [0, 0, 1];
    elseif shape(2) == 0
      obj.zero_dim = 2;
      obj.nonzero_dim_1 = 1;
      obj.nonzero_dim_2 = 3;
      obj.base_mat_size = [1, 0, 0];
    else
      obj.zero_dim = 3;
      obj.nonzero_dim_1 = 1;
      obj.nonzero_dim_2 = 2;
      obj.base_mat_size = [0, 1, 0];
    end
    % base_mat_size needs to multiply with part_frac and inner_prod_size.

    % Complete split distributions.
    obj.complete_split_id = cell(1, 3);
    for t = 1 : 3
      obj.complete_split_id{t} = ones(1, 3 ^ obj.power) * (-1);  % represents always zero
    end
    obj.complete_split_id{obj.zero_dim}(1) = -2;  % represents always one
    for id = 1 : 3 ^ obj.power
      arr = DecodeCSD(id, obj.power);
      if sum(arr) ~= obj.shape(obj.nonzero_dim_1)
        continue;
      end
      obj.complete_split_id{obj.nonzero_dim_1}(id) = param_manager.Register(1, 0, 1, [0, 1 / 100]);
      arr_opposite = ones(1, obj.power) * 2 - arr;
      id_opposite = EncodeCSD(arr_opposite);
      obj.complete_split_id{obj.nonzero_dim_2}(id_opposite) = obj.complete_split_id{obj.nonzero_dim_1}(id);
    end

    % Register linear constraint: sum of complete split distributions is 1.
    lincon_terms = {};
    for id = 1 : 3 ^ obj.power
      if obj.complete_split_id{obj.nonzero_dim_1}(id) == -1  % not possible to be -2
        continue;
      end
      lincon_terms{end + 1, 1} = obj.complete_split_id{obj.nonzero_dim_1}(id);
      lincon_terms{end, 2} = 1;
    end
    param_manager.AddLinearConstraintEq(lincon_terms, 1);
    
    % Placeholder variables
    obj.num_block_contribution = {[0, 0, 0], [0, 0, 0], [0, 0, 0]};
    obj.hash_penalty_term = {0, 0, 0};
    obj.p_comp = {0, 0, 0};
  end

  % Use heuristics to set initial parameters. We simply choose the complete split distribution that
  % maximizes the number of entries in the result inner product tensor.
  % CVX required for this function, and the user may select a default CVX solver before calling.
  function SetInitial(obj, ~)
    global q;
    cvx_begin quiet
      variables csd(1, 3 ^ obj.power);
      inner_prod_size = sum(entr(csd));
      for id = 1 : 3 ^ obj.power
        arr = DecodeCSD(id, obj.power);
        inner_prod_size = inner_prod_size + csd(id) * sum(arr == 1) * log(q);
      end
      maximize inner_prod_size;
      subject to
        sum(csd) == 1;
        csd >= 0;
        for id = 1 : 3 ^ obj.power
          arr = DecodeCSD(id, obj.power);
          if sum(arr) ~= obj.shape(obj.nonzero_dim_1)
            csd(id) == 0;
          end
        end
    cvx_end
    csd = max(csd, 0);
    csd = csd / sum(csd);
    global param_manager;
    for id = 1 : 3 ^ obj.power
      arr = DecodeCSD(id, obj.power);
      if sum(arr) ~= obj.shape(obj.nonzero_dim_1)
        continue;
      end
      param_manager.SetSingleParam(obj.complete_split_id{obj.nonzero_dim_1}(id), csd(id));
    end
  end

  % Override the evaluation process.
  function EvaluateInit(obj)
    global param_manager;
    obj.part_frac = GVar(param_manager.num_input, 0);
    obj.complete_split = cell(1, 3);
    for t = 1 : 3
      obj.complete_split{t} = GVar.Zeros(1, 3 ^ obj.power);
      for id = 1 : 3 ^ obj.power
        if obj.complete_split_id{t}(id) == -1
          obj.complete_split{t}(id) = 0;
        elseif obj.complete_split_id{t}(id) == -2
          obj.complete_split{t}(id) = 1;
        else
          obj.complete_split{t}(id) = param_manager.GetParam(obj.complete_split_id{t}(id));
        end
      end
    end
  end

  function EvaluatePre(obj)
    % Do nothing. We do not split, so do not propagate the fraction.
  end

  function EvaluatePost(obj)
    % Compute matrix size contribution. Should be multiplied by obj.part_frac.
    % Should equal to 2^entropy * q^(number of one), sum over CSD.
    global q;
    
    % First, without part_frac.
    inner_prod_size = obj.complete_split{obj.nonzero_dim_1}.Entropy();
    for id = 1 : 3 ^ obj.power
      arr = DecodeCSD(id, obj.power);
      inner_prod_size = inner_prod_size + obj.complete_split{obj.nonzero_dim_1}(id) * sum(arr == 1) * log(q);
    end

    % Combine with part_frac and base_mat_size.
    obj.mat_size_contribution = obj.part_frac * inner_prod_size * obj.base_mat_size;
  end
end
end
