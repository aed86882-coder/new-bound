% The class ParamManager manages all optimizable parameters. We only have one instance of this class as global variable.
% Before the optimization starts, we need to simulate the constraint evaluation process and register the parameters.
% Each registered parameter can be a scalar or a row vector.
% When registering, this class returns an ID for that parameter (group). That ID will be stored by other entities.
% When the optimization starts, the evaluator will call this class for GVars indicating these variables.

% Known issues:
%  - The PackResults() function might be slow.

% Note:
%  - Mostly, we do not run the optimization process from a random initial point. However, the functionality of
%    generating a random point in the parameter space still remains, because we need to evaluate at random points
%    to see the sparsity structure of the matrix of gradients.

classdef ParamManager < handle
properties
  num_group
  num_input       % i.e., the number of optimizable parameters.
  group_startpos  % Array of starting positions. 1 by num_group.
  group_size
  lb              % Concatenated array of lower bounds. 1 by num_input.
  ub
  cur_x           % Current value of the parameters. 1 by num_input.
  initializer     % Initializer for the parameters. 1 by num_input cell-array.
  lin_A
  lin_b
  lin_Aeq
  lin_beq
end
methods
  % Constructor
  function obj = ParamManager()
    obj.num_group = 0;
    obj.num_input = 0;
    obj.group_startpos = [];
    obj.group_size = [];
    obj.lb = [];
    obj.ub = [];
    obj.initializer = {};
    obj.cur_x = [];
    obj.lin_A = sparse(0, 0);
    obj.lin_b = sparse(0, 0);
    obj.lin_Aeq = sparse(0, 0);
    obj.lin_beq = sparse(0, 0);
  end

  % Register a parameter. Return the ID of the group.
  % lb and ub can be omitted, in which case [-Inf, Inf] is applied.
  % lb and ub can be single element or match the size of the parameter.
  function group_id = Register(obj, param_len, lb, ub, initer)
    if nargin < 3
      lb = -Inf;
    end
    if nargin < 4
      ub = Inf;
    end
    if isscalar(lb)
      lb = repmat(lb, 1, param_len);
    end
    if isscalar(ub)
      ub = repmat(ub, 1, param_len);
    end
    obj.num_group = obj.num_group + 1;
    group_id = obj.num_group;
    obj.group_startpos(group_id) = obj.num_input + 1;
    obj.group_size(group_id) = param_len;
    obj.num_input = obj.num_input + param_len;
    obj.lb = [obj.lb, lb];
    obj.ub = [obj.ub, ub];

    % Set initializer
    % Supported types:
    %  - [l, r]: uniform distribution over [l, r]
    %  - a function handle (function pointer): the function should take the size of the parameter as input and return 1 by n row vector.
    if nargin < 5
      initer = [0, 0.01];
    end
    if isa(initer, 'function_handle')
      obj.initializer{group_id} = initer;
    else
      obj.initializer{group_id} = @(sz) initer(1) + (initer(2) - initer(1)) * rand(1, sz);
    end
  end

  % Specify the current values of the parameters.
  function SetValue(obj, x)
    obj.cur_x = min(max(x, obj.lb), obj.ub);
  end

  % Specify the current values for a single parameter group.
  function SetSingleParam(obj, group_id, value)
    obj.cur_x(obj.group_startpos(group_id) : obj.group_startpos(group_id) + obj.group_size(group_id) - 1) = value;
  end

  % Random initialization.
  function x = RandomInit(obj)
    x = zeros(1, obj.num_input);
    for i = 1 : obj.num_group
      x(obj.group_startpos(i) : obj.group_startpos(i) + obj.group_size(i) - 1) = obj.initializer{i}(obj.group_size(i));
    end
    obj.cur_x = x;
  end

  % Get GVar for a parameter.
  function gvar = GetParam(obj, group_id)
    cur_start_pos = obj.group_startpos(group_id);
    gvar = GVar(obj.num_input, obj.cur_x(cur_start_pos : cur_start_pos + obj.group_size(group_id) - 1), 'startpos', cur_start_pos);
  end

  % Pack results into desired format.
  function [y, dy] = PackResults(obj, y_gvar)
    % y_gvar is a 1-d cell array of GVars.
    % y is a column vector storing constraint values.
    % dy is (# of constraints) by (# of parameters) matrix storing the gradients of the constraints.
    % The matrices might be very large. The current implementation is not too slow, but maybe could be improved further.
    y = [];
    dy = sparse(0, obj.num_input);
    for i = 1 : length(y_gvar)
      num_new_rows = length(y_gvar{i}.value);
      y(end + 1 : end + num_new_rows, 1) = y_gvar{i}.value;
      dy = [dy; y_gvar{i}.grad'];
    end
  end

  % Linear Constraints
  % A_entries is a k by 2 cell array, each row is {group_id, coeff_matrix},
  %   where coeff_matrix is n by r (n is the size of the parameter group). r is the same for all rows in the cell array, and b is a row vector of length r.
  % It is supported to add constraints when some unrelated variables are not registered yet.
  function AddLinearConstraint(obj, A_entries, b)
    row_start = size(obj.lin_A, 1) + 1;
    r = length(b);
    obj.lin_A(row_start + r - 1, obj.num_input) = 0;  % resize the matrix
    obj.lin_b(row_start : row_start + r - 1, 1) = b';
    for i = 1 : size(A_entries, 1)
      group_id = A_entries{i, 1};
      coeff_matrix = A_entries{i, 2};
      obj.lin_A(row_start : row_start + r - 1, obj.group_startpos(group_id) : obj.group_startpos(group_id) + obj.group_size(group_id) - 1) = coeff_matrix';
    end
  end

  % Just the same as AddLinearConstraint, but Aeq & beq.
  function AddLinearConstraintEq(obj, A_entries, b)
    row_start = size(obj.lin_Aeq, 1) + 1;
    r = length(b);
    obj.lin_Aeq(row_start + r - 1, obj.num_input) = 0;  % resize the matrix
    obj.lin_beq(row_start : row_start + r - 1, 1) = b';
    for i = 1 : size(A_entries, 1)
      group_id = A_entries{i, 1};
      coeff_matrix = A_entries{i, 2};
      obj.lin_Aeq(row_start : row_start + r - 1, obj.group_startpos(group_id) : obj.group_startpos(group_id) + obj.group_size(group_id) - 1) = coeff_matrix';
    end
  end

  % Read the linear constraints.
  function [A, b, Aeq, beq] = GetLinearConstraints(obj)
    % The column number of A and Aeq might be smaller than num_input. Fix it.
    if size(obj.lin_A, 1) > 0 && size(obj.lin_A, 2) < obj.num_input
      obj.lin_A(end, obj.num_input) = 0;
    end
    if size(obj.lin_Aeq, 1) > 0 && size(obj.lin_Aeq, 2) < obj.num_input
      obj.lin_Aeq(end, obj.num_input) = 0;
    end
    % Set return values
    A = obj.lin_A;
    b = obj.lin_b;
    Aeq = obj.lin_Aeq;
    beq = obj.lin_beq;
    % Remove empty rows
    non_empty_rows = any(A, 2);
    A = A(non_empty_rows, :);
    b = b(non_empty_rows, :);
    non_empty_rows = any(Aeq, 2);
    Aeq = Aeq(non_empty_rows, :);
    beq = beq(non_empty_rows, :);
    % Save back
    obj.lin_A = A;
    obj.lin_b = b;
    obj.lin_Aeq = Aeq;
    obj.lin_beq = beq;
  end

  function Perturb(obj, perturb_size)
    obj.cur_x = obj.cur_x + perturb_size * randn(1, obj.num_input);
    obj.cur_x = min(max(obj.cur_x, obj.lb), obj.ub);
  end
end
end
