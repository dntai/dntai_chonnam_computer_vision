(*) Install Anaconda and Python Packages as below:

conda create --name video_emotion python=3.5
activate video_emotion
python -m pip install --upgrade pip
conda install dlib
conda install -c aaronzs tensorflow-gpu 
conda install opencv
pip install moviepy

(*) Run main.bat containing command:
python main.py --path ./data/out1.mp4 --out_dir ./data --detection opencv --max_length 5 --min_length 2 --no_face_detect 5 --focus_face 3 --no_focus_face 5 --show_win False

--max_length      5 (maximum time to cut)
--min_length      2 (minimum time to cut)
--no_face_detect  5 (number of frame to cut because of no face detect)
--focus_face      3 (number of faces in a frame to calculate for emotion)
--no_focus_face   5 (number of frame to cut because of focust face is lost)