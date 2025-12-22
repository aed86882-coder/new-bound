% Write a new line of log to expname/log.txt and screen.

function WriteLog(str)
  global expinfo;
  fid = fopen(strcat(expinfo.exp_name, '/log.txt'), 'a');
  timestamp = datestr(now, 'mm-dd HH:MM:SS');
  fprintf(fid, '[%s] %s\n', timestamp, str);
  fprintf('[%s] %s\n', timestamp, str);
  fclose(fid);
end
