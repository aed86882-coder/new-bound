% Verify the provided sets of parameters.

AddPath;
VerifyOmega(5, 1.00, 'data/K100_2.37155181.mat');
VerifyAlpha(5, 'data/alpha_0.32133405.mat');
VerifyMu(5, 'data/mu_0.52766067.mat');

verify_all = true;
if verify_all
  VerifyOmega(5, 0.33, 'data/K33_2.00009991.mat');
  VerifyOmega(5, 0.34, 'data/K34_2.00059911.mat');
  VerifyOmega(5, 0.35, 'data/K35_2.00136202.mat');
  VerifyOmega(5, 0.40, 'data/K40_2.00954023.mat');
  VerifyOmega(5, 0.45, 'data/K45_2.02378783.mat');
  VerifyOmega(5, 0.50, 'data/K50_2.04299314.mat');
  VerifyOmega(5, 0.55, 'data/K55_2.06613393.mat');
  VerifyOmega(5, 0.60, 'data/K60_2.09263062.mat');
  VerifyOmega(5, 0.65, 'data/K65_2.12173331.mat');
  VerifyOmega(5, 0.70, 'data/K70_2.15304805.mat');
  VerifyOmega(5, 0.75, 'data/K75_2.18620911.mat');
  VerifyOmega(5, 0.80, 'data/K80_2.22092806.mat');
  VerifyOmega(5, 0.85, 'data/K85_2.25698323.mat');
  VerifyOmega(5, 0.90, 'data/K90_2.29420832.mat');
  VerifyOmega(5, 0.95, 'data/K95_2.33243927.mat');
  VerifyOmega(5, 1.00, 'data/K100_2.37155181.mat');
  VerifyOmega(5, 1.10, 'data/K110_2.45205576.mat');
  VerifyOmega(5, 1.20, 'data/K120_2.53506362.mat');
  VerifyOmega(5, 1.50, 'data/K150_2.79494019.mat');
  VerifyOmega(5, 2.00, 'data/K200_3.25038482.mat');
  VerifyOmega(5, 2.50, 'data/K250_3.72046737.mat');
  VerifyOmega(5, 3.00, 'data/K300_4.19880890.mat');
end
