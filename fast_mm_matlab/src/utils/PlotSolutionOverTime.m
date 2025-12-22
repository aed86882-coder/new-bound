% Auxiliary function used for visualizing the convergence procedure. Mainly for debugging.
% Known issue: only support log files for non-refining mode.
% It displays a figure where the horizontal axis is time while the vertical axis is the best
% known solution till that time.

function PlotSolutionOverTime(log_file_path)
  % Read the log file
  log_file = fopen(log_file_path, 'r');
  log_data = textscan(log_file, '%s', 'Delimiter', '\n');
  log_data = log_data{1};
  fclose(log_file);

  % Extract the best solution and time data
  best_solution = [];
  time = [];
  for i = 1:length(log_data)
    if contains(log_data{i}, 'New best solution:')
      best_solution(end + 1) = str2double(extractAfter(log_data{i}, 'New best solution: '));
      extr = extractBetween(log_data{i}, '[', ']');
      time(end + 1) = datenum(datetime(extr{1}, 'InputFormat', 'MM-dd HH:mm:ss'));
    end
  end

  % Subtract the first element from all the elements in the time array
  time = (time - time(1)) * 24 + 1;

  % Plot the curve
  plot(time, best_solution);
  xlabel('Time');
  ylabel('Best Solution');
  title('Best Solution over Time');
end
