% For debug only. Print an array to the screen.

function PrintArray(a, name)
  if nargin == 2
    fprintf("%s: ", name);
  end
  for i = 1 : length(a)
    fprintf("%.10f ", a(i));
  end
  fprintf("\n");
end
