% The class Workspace is the outermost structure that stores optimizable variables.
% It represents the process of hashing in all levels (including the global one).
% For every level, it adds up num_block of all parts, and compute the number of retained
% triples. This is done by creating an optimizable variable instead of using min().

% Also, it manages all parts and the global stage.
% When Build() is called, it creates instances for parts and global stage recursively.
% When Evaluate() is called, it calls EvaluateInit(), EvaluatePre(), EvaluatePost() for the global stage & all parts.
% The returned value of Evaluate() is ALL constraints of the system. It consists two cell arrays [c, ceq] of GVar. The convertion will be done outside.

classdef Workspace < handle
properties
  num_retain_comp_id  % 1x3 cell array of 1 by max_level cell-array
  num_retain_comp  % 1x3 cell array of 1 by max_level cell-array of GVar
  num_retain_comp_lasteval
  num_retain_glob_id
  num_retain_glob  % scalar GVar
  num_retain_glob_lasteval
  single_mat_size_id
  single_mat_size  % should equal min(mat_size(1), mat_size(2), mat_size(3) / K)
  single_mat_size_lasteval
  value  % final value in Schonhage's inequality.
  omega_id
  omega
  K_id
  K

  % Auxiliary entries for performance
  num_constr_c
  num_constr_ceq
end
methods
  function obj = Workspace()
    % Do nothing.
  end

  function obj = Build(obj, max_level_)
    % Initialize the object and register the variables.
    global max_level param_manager parts globstage expinfo;
    max_level = max_level_;
    param_manager = ParamManager();  % Initialize the parameter manager.
    obj.num_retain_comp_id = cell(1, 3);
    obj.num_retain_glob_id = cell(1, 3);
    for r = 1 : 3
      obj.num_retain_comp_id{r} = cell(1, max_level);
      for lv = 2 : max_level
        obj.num_retain_comp_id{r}{lv} = param_manager.Register(1, 0, Inf, [0, 1 / 100]);
      end
      obj.num_retain_glob_id{r} = param_manager.Register(1, 0, Inf, [0, 1 / 100]);
    end
    obj.single_mat_size_id = param_manager.Register(1, 0, Inf, [0, 1 / 100]);
    
    % Register objectives.
    if strcmp(expinfo.obj_mode, 'alpha')
      % We restrict omega == 2. A negligible error of 1e-9 is allowed due to floating-number error.
      goal_eps = 1e-9;
      obj.omega_id = param_manager.Register(1, 2, 2 + goal_eps);
    else
      obj.omega_id = param_manager.Register(1, 0, Inf);
    end
    if strcmp(expinfo.obj_mode, 'omega')
      % We restrict K == expinfo.K.
      obj.K_id = param_manager.Register(1, expinfo.K, expinfo.K);
    else
      obj.K_id = param_manager.Register(1, 0, Inf);
    end
    if strcmp(expinfo.obj_mode, 'mu')
      % We restrict omega(K) <= 1 + 2 * K.
      param_manager.AddLinearConstraint({obj.omega_id, 1; obj.K_id, -2}, 1);
    end
    
    obj.num_constr_c = 0;
    obj.num_constr_ceq = 0;
    % Recursively build things.
    parts = cell(1, max_level);
    globstage = GlobalStage();
    globstage.Build();
  end

  function SetInitial(obj)
    % This function only sets the auxiliary variables in the black box.
    % Before calling, the user should set K and omega to desired values.
    global param_manager max_level;
    obj.Evaluate();  % Get evaluated auxiliary variables
    for r = 1 : 3
      for l = 2 : max_level
        param_manager.SetSingleParam(obj.num_retain_comp_id{r}{l}, obj.num_retain_comp_lasteval{r}{l});
      end
      param_manager.SetSingleParam(obj.num_retain_glob_id{r}, obj.num_retain_glob_lasteval{r});
    end
    param_manager.SetSingleParam(obj.single_mat_size_id, obj.single_mat_size_lasteval);
  end

  function [c, ceq] = Evaluate(obj)
    % Initialization Step: extract the values of the optimizable variables.
    global max_level param_manager parts globstage q;
    obj.num_retain_comp = cell(1, 3);
    obj.num_retain_glob = cell(1, 3);
    obj.num_retain_comp_lasteval = cell(1, 3);
    obj.num_retain_glob_lasteval = cell(1, 3);
    for r = 1 : 3
      obj.num_retain_comp{r} = cell(1, max_level);
      obj.num_retain_comp_lasteval{r} = cell(1, max_level);
      for l = 2 : max_level
        obj.num_retain_comp{r}{l} = param_manager.GetParam(obj.num_retain_comp_id{r}{l});
      end
      obj.num_retain_glob{r} = param_manager.GetParam(obj.num_retain_glob_id{r});
    end
    obj.single_mat_size = param_manager.GetParam(obj.single_mat_size_id);
    obj.omega = param_manager.GetParam(obj.omega_id);
    obj.K = param_manager.GetParam(obj.K_id);
    
    % Initialization for parts
    globstage.EvaluateInit();
    for l = 1 : max_level
      for i = 1 : length(parts{l})
        parts{l}{i}.EvaluateInit();
      end
    end

    % Propagate the fractions of parts
    globstage.EvaluatePre();
    for l = max_level : -1 : 1
      for i = 1 : length(parts{l})
        parts{l}{i}.EvaluatePre();
      end
    end

    % Evaluate the intermediate results
    for l = 1 : max_level
      for i = 1 : length(parts{l})
        parts{l}{i}.EvaluatePost();
      end
    end
    globstage.EvaluatePost();

    % Collect the nonlinear constraints.
    c = cell(1, obj.num_constr_c);
    ceq = cell(1, obj.num_constr_ceq);
    len_c = 0;
    len_ceq = 0;

    function AddC(c_now)
      for tmp_addc = 1 : length(c_now)
        len_c = len_c + 1;
        c{len_c} = c_now{tmp_addc};
      end
    end

    function AddCeq(ceq_now)
      for tmp_addceq = 1 : length(ceq_now)
        len_ceq = len_ceq + 1;
        ceq{len_ceq} = ceq_now{tmp_addceq};
      end
    end

    % Lagrange constraints
    for l = 1 : max_level
      for i = 1 : length(parts{l})
        % If is PartInfoLv2 or PartInfoZero, does not have Lagrange constraints.
        if isa(parts{l}{i}, 'PartInfo')
          ceq_now = parts{l}{i}.GetLagrangeConstraints();
          AddCeq(ceq_now);
        end
      end
    end
    ceq_now = globstage.GetLagrangeConstraints();
    AddCeq(ceq_now);

    % Asymmetric hashing: num_of_retain constraints
    for l = 3 : max_level  % Asymmetric hashing starts from level-3
      for r = 1 : 3
        % Component hashing from level-l to level-(l-1), region r
        num_block = GVar(param_manager.num_input, [0, 0, 0]);
        hash_penalty = GVar(param_manager.num_input, 0);
        p_comp = GVar(param_manager.num_input, 0);
        for i = 1 : length(parts{l})
          num_block = num_block + parts{l}{i}.num_block_contribution{r};  % The frac_part and region_prop within the part are already considered.
          hash_penalty = hash_penalty + parts{l}{i}.hash_penalty_term{r};
          p_comp = p_comp + parts{l}{i}.p_comp{r};
        end
        obj.num_retain_comp_lasteval{r}{l} = Inf;
        for t = 1 : 3
          if t ~= r
            AddC({obj.num_retain_comp{r}{l} - num_block(t) + hash_penalty});
            foo = num_block(t) - hash_penalty;
            obj.num_retain_comp_lasteval{r}{l} = min(obj.num_retain_comp_lasteval{r}{l}, foo.value);
          else
            AddC({obj.num_retain_comp{r}{l} - num_block(t) + p_comp});
            foo = num_block(t) - p_comp;
            obj.num_retain_comp_lasteval{r}{l} = min(obj.num_retain_comp_lasteval{r}{l}, foo.value);
          end
        end
      end
    end
    % Symmetric hashing at level 2
    % Note that num_block_contribution of level-2 is NOT a 1x3 cell array.
    % (It is a 1x3 GVar indicating the contribution to N_X, N_Y, and N_Z).
    num_block = GVar(param_manager.num_input, [0, 0, 0]);
    for i = 1 : length(parts{2})
      num_block = num_block + parts{2}{i}.num_block_contribution;
    end
    % obj.num_retain_comp{1}{2} (region 1, level 2) is utilized to represent the min() relationship.
    % {2}{2} and {3}{2} (i.e., regions 2 & 3) are omitted and always set to zero.
    for t = 1 : 3
      AddC({obj.num_retain_comp{1}{2} - num_block(t)});
    end
    AddCeq({obj.num_retain_comp{2}{2}});
    AddCeq({obj.num_retain_comp{3}{2}});
    obj.num_retain_comp_lasteval{1}{2} = min(num_block.value);
    obj.num_retain_comp_lasteval{2}{2} = 0;
    obj.num_retain_comp_lasteval{3}{2} = 0;
    
    % Global hashing from CW power to max_level
    for r = 1 : 3
      num_block = globstage.num_block{r};
      hash_penalty = globstage.hash_penalty_term{r};
      p_comp = globstage.p_comp{r};
      obj.num_retain_glob_lasteval{r} = Inf;
      for t = 1 : 3
        if t ~= r
          AddC({obj.num_retain_glob{r} - num_block(t) + hash_penalty});
          foo = num_block(t) - hash_penalty;
          obj.num_retain_glob_lasteval{r} = min(obj.num_retain_glob_lasteval{r}, foo.value);
        else
          AddC({obj.num_retain_glob{r} - num_block(t) + p_comp});
          foo = num_block(t) - p_comp;
          obj.num_retain_glob_lasteval{r} = min(obj.num_retain_glob_lasteval{r}, foo.value);
        end
      end
    end

    % Single Matrix Size constraints
    % See what is the largest <a, a, a^K> we can obtain.
    mat_size = globstage.mat_size;
    AddC({obj.single_mat_size - mat_size(1), ...
          obj.single_mat_size - mat_size(2), ...
          obj.single_mat_size - mat_size(3) ./ obj.K});
    obj.single_mat_size_lasteval = min([mat_size.value(1), mat_size.value(2), mat_size.value(3) ./ obj.K.value]);

    % value satisfies Schonhage
    % value equals the sum of num_retain plus mat_size * omega
    obj.value = obj.single_mat_size * obj.omega;
    for r = 1 : 3
      obj.value = obj.value + obj.num_retain_glob{r};  % Have considered region_frac.
    end
    for l = 2 : max_level
      for r = 1 : 3
        obj.value = obj.value + obj.num_retain_comp{r}{l};
      end
    end
    AddC({log(q + 2) * (2 .^ (max_level - 1)) - obj.value});  % value >= power * log(q + 2)

    % Set back the auxiliary variables
    % len_c and len_ceq are unnecessary auxiliary variables but could avoid dynamic allocation.
    if length(c) > len_c
      c = c(1 : len_c);
    end
    if length(ceq) > len_ceq
      ceq = ceq(1 : len_ceq);
    end
    obj.num_constr_c = len_c;
    obj.num_constr_ceq = len_ceq;
  end

  function omega_pos = GetOmegaPos(obj)
    % [Hacky function] Return the position of omega in the param_manager.
    global param_manager;
    omega_pos = param_manager.group_startpos(obj.omega_id);
  end

  function K_pos = GetKPos(obj)
    % [Hacky function] Return the position of K in the param_manager.
    global param_manager;
    K_pos = param_manager.group_startpos(obj.K_id);
  end
end
end
