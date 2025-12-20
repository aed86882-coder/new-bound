% Compute a "reasonable" initial point. This is called after building the workspace.
% Initialize with Le Gall's distributions for square matrix multiplication (q = 5, 4-th power).

function InitialPoint()
  global workspace param_manager parts globstage q max_level expinfo;
  
  % Read Le Gall's parameters of K = 1
  fid = fopen('primitive_param_legall.json');
  % Read json file
  json = jsondecode(fscanf(fid, '%c'));
  fclose(fid);

  % Hardcoded: how likely the 2 in 112 (resp. 022) will split into 0+2 (or 2+0).
  split_112 = 0.021007529944960156;
  split_022 = 1 / (q^2 + 2);
  
  % Set initial points for level-2 parts
  for i = 1 : length(parts{2})
    if strcmp(parts{2}{i}.shape_type, '022')
      parts{2}{i}.SetInitial(split_022);
    elseif strcmp(parts{2}{i}.shape_type, '112')
      parts{2}{i}.SetInitial(split_112);
    end
  end

  % Set initial points for other parts. Use the json data.
  % The Lagrange multipliers will also be set accordingly.
  for l = 3 : max_level
    for i = 1 : length(parts{l})
      parts{l}{i}.SetInitial(json);
    end
  end

  % Set initial point for global stage
  globstage.SetInitial(json);

  % Set initial for the workspace. The omega for this parameter set is hardcoded.
  param_manager.SetSingleParam(workspace.omega_id, 2.372937 * max(1, expinfo.K));
  % Initializing the workspace does not require the json; it basically evaluates once and
  % set all the auxiliary variables for min() to correct values.
  workspace.SetInitial();
end
