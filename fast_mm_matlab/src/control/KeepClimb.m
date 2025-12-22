% Repeatedly execute ClimbStep until a local minimum is reported or stopped manually.

function KeepClimb()
  global sn_output expinfo;
  exp_name = expinfo.exp_name;

  % If manual_stop sign is set, skip this function.
  fid = fopen(strcat(exp_name, '/manual_stop.txt'));
  tline = fgetl(fid);
  fclose(fid);
  if strcmp(tline, '1')
    return;
  end

  % Keep trying perturbation.
  while true
    WriteLog('New Climbing Step');
    ClimbStep();

    fid = fopen(strcat(exp_name, '/manual_stop.txt'));
    tline = fgetl(fid);
    fclose(fid);

    if strcmp(tline, '1') || sn_output.INFO == 1 || sn_output.INFO == 3
      break;
    end
  end
end
