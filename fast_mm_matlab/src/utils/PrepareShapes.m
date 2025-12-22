% Given a level, list all shapes (i, j, k) where i + j + k = 2^{level}.
% Return a n*3 matrix.

function shapes = PrepareShapes(level)
  shapes = [];
  sum_col = 2 ^ level;
  for i = 0 : sum_col
    for j = 0 : sum_col - i
      k = sum_col - i - j;
      shapes = [shapes; i, j, k];
    end
  end
end
