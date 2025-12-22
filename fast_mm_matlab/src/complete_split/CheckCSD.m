% Check if every non-zero entry of csd (a complete split distribution) has index-sum k.
% This is only for debug use.

function CheckCSD(csd, power, k)
  if isa(csd, 'GVar')
    csd = csd.value;
  end
  for id = 1 : 3 ^ power
    if csd(id) ~= 0
      if sum(DecodeCSD(id, power)) ~= k
        error("CSD's sum is not %d", k);
      end
    end
  end
end
