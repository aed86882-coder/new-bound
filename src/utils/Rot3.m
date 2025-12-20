% Rotate a 3-element vector to the left (by n times).

function x = Rot3(x, n)
  if nargin < 2
    n = 1;
  end
  n = mod(n, 3);
  switch n
  case 0
    return;
  case 1
    x = [x(2), x(3), x(1)];
    return;
  case 2
    x = [x(3), x(1), x(2)];
    return;
  end
end
