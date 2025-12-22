% Find a part in the given level that matches "shape" and "identifier".
% If not found, create one and return.
% Currently, "identifier" consists of the ID of the parent part and the hashing region (1 to 3).

function [idx, ptr, is_new] = FindOrCreatePart(level, shape, identifier)
  global parts;
  idx = FindPartByIdentifier(level, shape, identifier);
  if idx ~= -1
    ptr = parts{level}{idx};
    is_new = false;
    return;
  end
  idx = length(parts{level}) + 1;
  parts{level}{idx} = PartInfo.CreateInstance(level, shape);
  ptr = parts{level}{idx};
  is_new = true;
end
