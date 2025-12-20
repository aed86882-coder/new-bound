% Encode an index array (i_1, ..., i_{power}) to a number in [1, 3^power].

function id = EncodeCSD(arr, ~)
  id = 0;
  for i = 1 : length(arr)
    id = id * 3 + arr(i);
  end
  id = id + 1;  % base-3 index starting from 0
end
