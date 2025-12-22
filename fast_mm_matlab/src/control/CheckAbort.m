% Check whether we should stop SNOPT. This is called every major iteration.
% It also reads expinfo.is_refine_mode to decide its behavior.
% First, if "manual_stop.txt" contains 1, then stop.
% Second, in non-refine mode, if no updates within 1 hour, then stop.

function [iAbort] = CheckAbort(itn, nMajor, nMinor, condZHZ, obj, merit, step, ...
  primalInf, dualInf, maxViol, maxViolRel, ...
  x, xlow, xupp, xmul, xstate, ...
  F, Flow, Fupp, Fmul, Fstate)

  % Called every major iteration
  % Use iAbort to stop SNOPT (if iAbort == 0, continue; else stop)
  iAbort = 0;

  global expinfo;
  exp_name = expinfo.exp_name;

  % open "manual_stop.txt" and read the first line
  fid = fopen(strcat(exp_name, '/manual_stop.txt'));
  tline = fgetl(fid);
  fclose(fid);

  % if the first line is "1", then stop SNOPT
  if strcmp(tline, '1')
    iAbort = 1;
    WriteLog('Manual stop.');
  end

  % If no updates within 1 hour, stop. This is disabled in the refine mode.
  if ~expinfo.is_refine_mode && expinfo.TimeElapsed() > 3600
    iAbort = 1;
    WriteLog('No updates within 1 hour.');
  end
end
