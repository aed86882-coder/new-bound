% Fix alph with some precision issue.

function alph = FixAlpha(alph)
  alph = max(alph, 0);
  alph = alph / sum(alph);
end
