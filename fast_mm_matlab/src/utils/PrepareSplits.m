% Similar to PrepareShapes. Given a shape (i, j, k) where i + j + k = 2^{level},
% list all methods to split (i, j, k) into two level-{level - 1} shapes.
% Return a n*6 matrix.

function splits = PrepareSplits(shape)
  splits = [];
  sum_col = sum(shape);
  sum_half = sum_col / 2;
  for i = 0 : min(sum_half, shape(1))
    for j = 0 : min(sum_half - i, shape(2))
      k = sum_half - i - j;
      if k > shape(3)
        continue;
      end
      splits = [splits; i, j, k, shape(1) - i, shape(2) - j, shape(3) - k];
    end
  end
end
