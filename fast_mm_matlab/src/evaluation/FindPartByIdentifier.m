% Find a part in the given level that matches "shape" and "identifier".
% If not found, return -1; otherwise, return the ID of the part.

function id = FindPartByIdentifier(level, shape, identifier)
  global parts;
  id = -1;
  for i = 1 : length(parts{level})
    if isequal(parts{level}{i}.shape, shape) && isequal(parts{level}{i}.identifier, identifier)
      id = i;
      return;
    end
  end
  % return -1 if not found
end
