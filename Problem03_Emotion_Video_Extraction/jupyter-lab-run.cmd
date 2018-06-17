@ECHO OFF
%~d0%
CD %~dp0
REM %~dp0 will give you the full path to the batch file's directory (static)
REM %cd% will give you the current working directory (variable)
REM %~dp0  drive+path command file, %~p0 only path
SET PYTHONROOT=C:/Anaconda3
SET PYTHONPATH=%PYTHONPATH%
SET PATH=%PYTHONROOT%;%PYTHONROOT%/Scripts;%PATH%;C:\Program Files\NVIDIA Corporation\NVSMI
SET CMD1=activate root
SET CMD2=jupyter notebook --config='jupyter-lab-config.json'
REM cmd.exe /K "%CMD1%"
REM cmd.exe /K "%CMD1% && %CMD2%"
cmd.exe /K "%CMD2%"
REM cmd.exe