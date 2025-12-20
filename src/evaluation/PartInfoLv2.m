% Same as PartInfo, but only for level 2.
% We do not explicitly construct level-1 parts, but instead use closed-forms at level 2.
% One parameter is created for 112 and 022 parts.

% For all of the components, we let "complete_split{1 : 3}" represent the complete split distributions on X, Y, Z dimensions, respectively.
% Each of them is a 1x9 GVar array.

% UPD: Adjusted to asymmetric hashing version. Waiting for correctness check.

classdef PartInfoLv2 < matlab.mixin.Copyable
properties
  level
  power
  sum_col
  sum_half
  part_id
  shape
  shape_type  % '112' or '022'
  rotate_num  % The current shape is obtained by shifting '112' or '022' towards the left by rotate_num times.

  % Optimizable variables
  split_0_id  % Probability of splitting into 0+2 or 2+0
  split_0

  % GVars
  mat_size_contribution  % 1x3 GVar
  num_block_contribution  % 1x3 GVar
  part_frac  % Decided by upper levels
  hash_penalty_term  % Placeholder, always equal 0.
  complete_split
  p_comp  % Placeholder, always 0.

  identifier
end
methods
  function obj = PartInfoLv2()
    % Do nothing
  end
  
  function obj = Build(obj, level, part_id, shape, identifier)
    if level ~= 2
      error('PartInfoLv2: level should be 2.');
    end
    obj.level = level;  % Should be 2.
    obj.power = 2 ^ (level - 1);
    obj.sum_col = 2 ^ level;
    obj.sum_half = obj.sum_col / 2;
    obj.part_id = part_id;
    obj.shape = shape;
    obj.identifier = identifier;

    if max(shape) == 2 && min(shape) == 0
      obj.shape_type = '022';
      standard_shape = [0, 2, 2];
    elseif max(shape) == 2
      obj.shape_type = '112';
      standard_shape = [1, 1, 2];
    elseif max(shape) == 3
      % 013 or 031
      if isequal(shape, [0, 1, 3]) || isequal(shape, [1, 3, 0]) || isequal(shape, [3, 0, 1])
        obj.shape_type = '013';
        standard_shape = [0, 1, 3];
      else
        obj.shape_type = '031';
        standard_shape = [0, 3, 1];
      end
    else
      obj.shape_type = '004';
      standard_shape = [0, 0, 4];
    end
    obj.rotate_num = 0;
    while ~isequal(shape, standard_shape)
      shape = [shape(3), shape(1), shape(2)];  % Rotate to the right.
      obj.rotate_num = obj.rotate_num + 1;
    end
    
    if strcmp(obj.shape_type, '112') || strcmp(obj.shape_type, '022')
      obj.RegisterVariables();
    end

    % Placeholder
    obj.hash_penalty_term = 0;
    obj.p_comp = 0;
  end

  function RegisterVariables(obj)
    global param_manager;
    obj.split_0_id = param_manager.Register(1, 0, 0.5, [0, 1 / 100]);
  end

  function SetInitial(obj, split_0_initial)
    global param_manager;
    if strcmp(obj.shape_type, '112') || strcmp(obj.shape_type, '022')
      param_manager.SetSingleParam(obj.split_0_id, split_0_initial);
    end
  end

  function EvaluateInit(obj)
    global param_manager;
    if strcmp(obj.shape_type, '112') || strcmp(obj.shape_type, '022')
      obj.split_0 = param_manager.GetParam(obj.split_0_id);
    end
    obj.part_frac = GVar(param_manager.num_input, 0);
  end

  % The first stage of evaluation, called from upper levels to lower levels
  function EvaluatePre(obj)
    % Do nothing.
    % part_frac is already computed by upper levels.
  end

  % There are very few level-2 shapes, so we implement closed-forms for them.
  % The analysis for T_{1,1,2} (and its rotations) is by the laser method, while for other
  % constituent tensors we directly write the tensor power (with complete split distribution
  % constraints) as an inner product tensor.
  function EvaluatePost(obj)
    global param_manager q;
    obj.complete_split = cell(1, 3);
    if strcmp(obj.shape_type, '022')
      obj.num_block_contribution = GVar(param_manager.num_input, [0, 0, 0]);
      % The whole thing is an inner product tensor <1, 1, m>
      inner_prod_size = Entropy([obj.split_0, obj.split_0, 1 - 2 * obj.split_0]) + 2 * log(q) * (1 - 2 * obj.split_0);
      obj.mat_size_contribution = [0, 0, inner_prod_size];
      % Complete split distributions
      obj.complete_split{1} = GVar.Zeros(1, 9);
      obj.complete_split{1}(EncodeCSD([0, 0])) = 1;
      obj.complete_split{2} = GVar.Zeros(1, 9);
      obj.complete_split{2}(EncodeCSD([0, 2])) = obj.split_0;
      obj.complete_split{2}(EncodeCSD([2, 0])) = obj.split_0;
      obj.complete_split{2}(EncodeCSD([1, 1])) = 1 - 2 * obj.split_0;
      obj.complete_split{3} = obj.complete_split{2};
    elseif strcmp(obj.shape_type, '112')
      obj.num_block_contribution = [log(2), log(2), Entropy([obj.split_0, obj.split_0, 1 - 2 * obj.split_0])];
      % (1, 1, 0) -> <1, q, 1>; (1, 0, 1) -> <q, 1, 1>; (0, 1, 1) -> <1, 1, q>
      obj.mat_size_contribution = 2 * obj.split_0 * [0, log(q), 0] + (1 - 2 * obj.split_0) * [log(q), 0, log(q)];
      % Complete split distributions
      obj.complete_split{1} = GVar.Zeros(1, 9);
      obj.complete_split{1}(EncodeCSD([0, 1])) = 0.5;
      obj.complete_split{1}(EncodeCSD([1, 0])) = 0.5;
      obj.complete_split{2} = obj.complete_split{1};
      obj.complete_split{3} = GVar.Zeros(1, 9);
      obj.complete_split{3}(EncodeCSD([0, 2])) = obj.split_0;
      obj.complete_split{3}(EncodeCSD([2, 0])) = obj.split_0;
      obj.complete_split{3}(EncodeCSD([1, 1])) = 1 - 2 * obj.split_0;
    elseif strcmp(obj.shape_type, '013') || strcmp(obj.shape_type, '031')
      obj.num_block_contribution = GVar(param_manager.num_input, [0, 0, 0]);
      inner_prod_size = log(2) + log(q);
      obj.mat_size_contribution = GVar(param_manager.num_input, [0, 0, inner_prod_size]);
      % Complete split distributions
      obj.complete_split{1} = GVar.Zeros(1, 9);
      obj.complete_split{1}(EncodeCSD([0, 0])) = 1;
      obj.complete_split{2} = GVar.Zeros(1, 9);
      obj.complete_split{3} = GVar.Zeros(1, 9);
      if strcmp(obj.shape_type, '013')
        obj.complete_split{2}(EncodeCSD([0, 1])) = 0.5;
        obj.complete_split{2}(EncodeCSD([1, 0])) = 0.5;
        obj.complete_split{3}(EncodeCSD([1, 2])) = 0.5;
        obj.complete_split{3}(EncodeCSD([2, 1])) = 0.5;
      else  % 031
        obj.complete_split{2}(EncodeCSD([1, 2])) = 0.5;
        obj.complete_split{2}(EncodeCSD([2, 1])) = 0.5;
        obj.complete_split{3}(EncodeCSD([0, 1])) = 0.5;
        obj.complete_split{3}(EncodeCSD([1, 0])) = 0.5;
      end
    else  % 004
      obj.num_block_contribution = GVar(param_manager.num_input, [0, 0, 0]);
      obj.mat_size_contribution = GVar(param_manager.num_input, [0, 0, 0]);
      % Complete split distributions
      obj.complete_split{1} = GVar.Zeros(1, 9);
      obj.complete_split{1}(EncodeCSD([0, 0])) = 1;
      obj.complete_split{2} = obj.complete_split{1};
      obj.complete_split{3} = GVar.Zeros(1, 9);
      obj.complete_split{3}(EncodeCSD([2, 2])) = 1;
    end
    % Rotate and multiply by part_frac.
    obj.num_block_contribution = Rot3(obj.num_block_contribution, obj.rotate_num) * obj.part_frac;
    obj.mat_size_contribution = Rot3(obj.mat_size_contribution, obj.rotate_num) * obj.part_frac;
    obj.complete_split = Rot3c(obj.complete_split, obj.rotate_num);
  end
end
end
