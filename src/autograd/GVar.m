% Create a class named 'GVar'. It maintains a scalar (or a row vector) of variables with gradient information. Inside the class, we store:
%  - num_input: the number of input variables. It decides the size of the gradient matrix.
%  - value: the value of the variable. It is 1 * n for an arbitrary n >= 1.
%  - grad: the gradient of the variable. It is num_input * n.
% When initializing the class, we need to specify the number of input variables and maybe the value of the
%   variable. If the latter is not specified, we will initialize it to [0]. Note that we also need to know
%   the size (n) of the variable, but it is already included in the size of the value.
% Support basic operations: +, -, *, .*, ./, Entropy (e-based). The gradient is automatically computed.

% Known issues:
%  - (a + b).grad does not work. Need to first assign the result to a new GVar, say c = a + b, then use c.grad.
%  - ./ operator is not accurate when the denominator is below eps (around 1e-16).
%  - The extreme case handling in Entropy() will probably slow down the program.
%  - Entropy() is the e-based entropy instead of 2.
%  - Does not support / operator. Use ./ instead.
%  - No longer support log() or log2(). Not needed in the current implementation.

% Note:
%  - The corner case handling hyperparameters (like eps) cannot be too extreme (like 1e100),
%      otherwise the optimizer does not work. Default value of eps == 1e-16 works well.
%  - In some of the functions we used approximated or inaccurate partial derivatives,
%      but we tried the best to make the function value precise, so the correctness of the verification
%      program still holds.

classdef GVar
properties
  num_input
  value
  grad
end  
methods
  % Constructor 1: GVar(num_input). value = 0, grad = zero vector.
  % Constructor 2: GVar(num_input, value). grad = zero vector/matrix.
  % Constructor 3: GVar(num_input, value, 'pos'/'startpos'/'grad', param). grad is generated in three ways:
  %   - 'pos': param is 1 * n vector. Gradient of the i-th entry equals to double(1:num_input == param(i))'.
  %   - 'startpos': param is a scalar. Gradient of the i-th entry equals to double(1:num_input == param + i - 1)'.
  %   - 'grad': param is num_input * n matrix. Gradient equals param.
  function obj = GVar(num_input, value, grad_type, grad_param)
    if nargin < 2
      value = 0;
    end
    obj.num_input = num_input;
    obj.value = value;
    obj.grad = sparse(num_input, length(value));
    if nargin == 4
      if strcmp(grad_type, 'pos')
        for i = 1 : length(grad_param)
          obj.grad(grad_param(i), i) = 1;
        end
      elseif strcmp(grad_type, 'startpos')
        for i = 1 : length(value)
          obj.grad(grad_param + i - 1, i) = 1;
        end
      elseif strcmp(grad_type, 'grad')
        obj.grad = grad_param;
      else
        error('Unknown grad_type');
      end
    end
  end

  % Override the assign operator. When assigning a scalar/row vector, we will assign the value and set the
  %   gradient to zero; when assigning a GVar, we will assign the value and the gradient.
  % a(i) = b: assign scalar to an entry. b can be a scalar or a GVar of length 1.
  % a(:) = b: assign row vector to the whole vector. We also change the size.
  % a(i:j) = b: assign row vector to a subvector. (Did not implement on purpose, but seems to work.)
  function obj = subsasgn(obj, S, B)
    if length(S) ~= 1 || ~strcmp(S(1).type, '()') || length(S.subs) ~= 1
      obj = builtin('subsasgn', obj, S, B);
      return;
    end
    
    if isnumeric(B)
      B_val = B;
      B_grad = sparse(obj.num_input, size(B, 2));
    elseif isa(B, 'GVar')
      B_val = B.value;
      B_grad = B.grad;
    else
      error('Unsupported subsasgn type');
    end

    % if S.subs{1} == ':'
    if ischar(S.subs{1}) && strcmp(S.subs{1}, ':')
      obj.value = B_val;
      obj.grad = B_grad;
    else
      idx = S.subs{1};
      obj.value(idx) = B_val;
      obj.grad(:, idx) = B_grad;
    end
  end

  % Override the subsref operator. Supported cases:
  % a(i): get the i-th entry. Return a scalar GVar.
  function varargout = subsref(obj, S)
    if length(S) ~= 1 || ~strcmp(S(1).type, '()') || length(S.subs) ~= 1
      [varargout{1:nargout}] = builtin('subsref', obj, S);
      return;
    end
    
    idx = S.subs{1};
    varargout{1} = GVar(obj.num_input, obj.value(idx), 'grad', obj.grad(:, idx));
  end

  % Override the horzcat operator. Support [a, b, c, ..., x], where each of the elements can be a scalar/row vector or a GVar.
  function obj = horzcat(varargin)
    obj = varargin{1};
    for i = 2 : length(varargin)
      % Concatenate obj with varargin{i}.
      % If both are non-GVars
      if ~isa(obj, 'GVar') && ~isa(varargin{i}, 'GVar')
        obj = [obj, varargin{i}];
      else
        % Convert both to GVar
        if ~isa(obj, 'GVar')
          obj = GVar(varargin{i}.num_input, obj);
        end
        if ~isa(varargin{i}, 'GVar')
          varargin{i} = GVar(obj.num_input, varargin{i});
        end
        obj = GVar(obj.num_input, [obj.value, varargin{i}.value], 'grad', [obj.grad, varargin{i}.grad]);
      end
    end
  end

  % Override the plus operator. Both sides can be GVar or scalar/row vector.
  function obj = plus(obj1, obj2)
    if ~isa(obj1, 'GVar')
      obj1 = GVar(obj2.num_input, obj1);
    end
    if ~isa(obj2, 'GVar')
      obj2 = GVar(obj1.num_input, obj2);
    end
    obj = GVar(obj1.num_input, obj1.value + obj2.value, 'grad', obj1.grad + obj2.grad);
  end

  % Override the minus operator. Both sides can be GVar or scalar/row vector.
  function obj = minus(obj1, obj2)
    if ~isa(obj1, 'GVar')
      obj1 = GVar(obj2.num_input, obj1);
    end
    if ~isa(obj2, 'GVar')
      obj2 = GVar(obj1.num_input, obj2);
    end
    obj = GVar(obj1.num_input, obj1.value - obj2.value, 'grad', obj1.grad - obj2.grad);
  end

  % Override the .* operator. Both sides can be GVar or scalar/row vector. Supported cases:
  % (1 by n) .* (1 by n), (1 by n) .* scalar, and scalar .* (1 by n).
  function obj = times(obj1, obj2)
    if ~isa(obj1, 'GVar')
      obj1 = GVar(obj2.num_input, obj1);
    end
    if ~isa(obj2, 'GVar')
      obj2 = GVar(obj1.num_input, obj2);
    end
    obj = GVar(obj1.num_input, obj1.value .* obj2.value, 'grad', obj1.grad .* obj2.value + obj1.value .* obj2.grad);
    % Note: MATLAB matrices support (n by m) .* (1 by m), broadcasting the second one.
  end

  % Override the ./ operator. Both sides can be GVar or scalar/row vector.
  function obj = rdivide(obj1, obj2)
    if ~isa(obj1, 'GVar')
      obj1 = GVar(obj2.num_input, obj1);
    end
    if ~isa(obj2, 'GVar')
      obj2 = GVar(obj1.num_input, obj2);
    end
    starting_point = eps;
    obj = GVar(obj1.num_input, obj1.value ./ max(obj2.value, starting_point), 'grad', ...
               (obj1.grad .* obj2.value - obj1.value .* obj2.grad) ./ max((obj2.value .^ 2), starting_point .^ 2));
  end

  % Override the * operator. Supported cases for a * b:
  % one of a and b is a scalar, and the other is a GVar: same as .*;
  % a is a GVar (1 by n), and b is a matrix (n by m): return a GVar of 1 by m.
  function obj = mtimes(obj1, obj2)
    % See if the first case applies.
    if all(size(obj1) == 1) || all(size(obj2) == 1)
      obj = times(obj1, obj2);
      return;
    end
    % The second case.
    if ~isa(obj1, 'GVar')
      error('Unexpected GVar.mtimes case');
    end
    % The second parameter (matrix) does not have gradients.
    obj = GVar(obj1.num_input, obj1.value * obj2, 'grad', obj1.grad * obj2);
  end

  % Override the exp operator.
  function res = exp(obj)
    res = GVar(obj.num_input, exp(obj.value), 'grad', obj.grad .* exp(obj.value));
  end

  % Entropy(a): sum of -a_i .* ln(a_i).
  % If some entry a_i is less than eps (a predetermined hyperparameter),
  % we take log(eps) instead of log(a_i) to avoid infinite derivative.
  % Even for entries close to zero, the function value (instead of the derivative) is always accurate.
  function res = Entropy(obj)
    eps_entr = eps;  % roughly 1e-16, dominated by the function evaluation's float-number error.
    res = GVar(obj.num_input, sum(-obj.value .* log(obj.value + (obj.value <= 0))), 'grad', ...
               obj.grad * (-log(max(obj.value, eps_entr)) - 1)');
  end

  % NormalizedEntropy(a, p): returns p * Entropy(a / p). 'a' is a vector with sum p, while p can be very small.
  % It equals the sum of -a_i .* ln(a_i / p).
  % p should always be sum(a_i), but we still pass in this parameter to avoid repetitive calculations (and hence errors).
  % Again, when any entry is close to zero, the function value is accurate
  % but the derivative will compute log(eps) instead of the precise formula.
  % This if for avoiding NaN or Inf in the derivative.
  function res = NormalizedEntropy(obj, p)
    eps_entr = eps;
    if p.value <= 0
      % Corner case, special derivatives.
      % Actually, at the point p == 0 (thus all a_i == 0), the function NormalizedEntropy() is not differentiable.
      % In this case, we overestimate the partial derivatives in order to help escape this point (if possible).
      res = GVar(obj.num_input, 0, 'grad', sum(obj.grad, 2) .* log(size(obj.value, 2)));
    else
      res = GVar(obj.num_input, sum(-obj.value .* log(obj.value ./ p.value + (obj.value <= 0))), 'grad', ...
                 obj.grad * (-log(max(obj.value ./ p.value, eps_entr)))');  % Inaccurate when obj.value is close to zero.
    end
  end

  % Override size() function. Return the size of the value.
  function s = size(obj, dim)
    if nargin == 1
      s = size(obj.value);
    else
      s = size(obj.value, dim);
    end
  end

  % Override length() function. Return the length of the value.
  function l = length(obj)
    l = length(obj.value);
  end
end
methods (Static)
  % We can use GVar.Convert([1, 2, 3]) to initialize a GVar with zero gradient.
  function obj = Convert(value)
    global param_manager;
    obj = GVar(param_manager.num_input, value);
  end

  % We can use GVar.Zeros(1, 100) to initialize a 1 by 100 zero vector (with zero gradient).
  function obj = Zeros(varargin)
    global param_manager;
    obj = GVar(param_manager.num_input, zeros(varargin{:}));
  end
end
end
