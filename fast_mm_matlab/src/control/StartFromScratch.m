% Start from the initial point generated from "primitive_param_legall.txt"
% (i.e., from Le Gall's analysis of CW_5^4 for square matrix multiplication)
% and run the climber controller.

function StartFromScratch(q_, K_, exp_name)
  SnInit(q_, K_, exp_name);  % Build the environment
  InitialPoint();  % Get the initial point
  SnContinue();
  KeepClimb();
end

%{
Use example:
  AddPath;
  experiment_name = 'K100';
  StartFromScratch(5, 1.00, experiment_name);
%}
