@ECHO OFF
%~d0%
CD %~dp0
SET PYTHONROOT=C:/Anaconda3/
SET PYTHONPATH=%PYTHONPATH%
SET PATH=%PYTHONROOT%;%PYTHONROOT%/Scripts;%PATH%;C:\Program Files\NVIDIA Corporation\NVSMI
python main.py --path ./data/out1.mp4 --out_dir ./data --detection opencv --max_length 5 --min_length 2 --no_face_detect 5 --focus_face 3 --no_focus_face 5 --show_win True --show_scale 0.5
pause