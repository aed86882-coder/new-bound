% Input the id (from 1 to 3^power) and output an index array (i_1, ..., i_{power})
% that indicates an entry of the complete split distribution.

function arr = DecodeCSD(id, power)
  arr = zeros(1, power);
  id = id - 1;  % base-3 index starting from 0
  for i = power : -1 : 1
    arr(i) = mod(id, 3);
    id = floor(id / 3);
  end
end
