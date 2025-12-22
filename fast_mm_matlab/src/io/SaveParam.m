% Save the whole environment.

function SaveParam(filename, x)
  if nargin < 1
    filename = 'best.mat';
  end
  global param_manager;
  if nargin < 2
    x = param_manager.cur_x';
  end

  file = struct;
  file.params = x';
  save(filename, '-struct', 'file');
end
