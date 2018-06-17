import argparse, cv2, numpy as np, sys, os
from moviepy.editor import *

src_zip = os.path.join(os.path.split(__file__)[0],'prlab.zip')
if os.path.exists(src_zip) == True and src_zip not in sys.path:
    print(src_zip)
    sys.path.insert(0, src_zip)
		
from prlab.detection.dlib.detection import DlibDetection
from prlab.detection.dlib.detection import DlibGpuDetection
from prlab.detection.opencv.detection import OpenCVDetection
from prlab.detection.mtcnn_tf.detection import MtcnnFaceDetection
from prlab.utils.video import VideoReader
from prlab.emotion.mlcnn.detection import MLCNNEmotionDetection
from prlab.projects.FaceTracking.HungarianFaceTracking import HungarianMatching

class FaceEmotionReader(VideoReader):
    def __init__(self, path = "", out_dir = '.', detection = 'mtcnn', extra = {}, **kwargs):
        super(FaceEmotionReader, self).__init__(source = 'video', path = path, extra = extra, **kwargs)
        
        self.detection = detection
        if self.detection == 'mtcnn':
            print('Mtcnn Face Detection')
            self.detector = MtcnnFaceDetection.getDetector()
        elif self.detection == 'opencv':
            print('OpenCV Face Detection')
            self.detector = OpenCVDetection.getDetector()
        elif self.detection == 'dlib': 
            print('Dlib Face Detection')
            self.detector = DlibDetection.getDetector()
        else:
            self.detector = MtcnnFaceDetection.getDetector()

        self.emotion_detector = MLCNNEmotionDetection.getDetector()
        self.emotion_labels   = self.emotion_detector.labels
        self.params["show_info"] = "FPS: {fps:.0f} - {timestamp_sec: .2f} (s) - {num_faces: d} Face(s){split}"
        self.params["show_prop"] = {'pos': (20, 50), 'font_scale': 1.2, 'thickness': 2}
        # self.params["debug"] = True
        self.faces = []
        self.emotions = []
        self.matching = HungarianMatching()
        self.initOnce = False
        
        self.split_positions = []
        self.cur_start =  0.0
        self.cur_end   = -1.0
        self.is_splitting_process = False
        self.num_split = 0
        
        self.max_length= int(self.params["max_length"]) # msec - 2s
        self.min_length= int(self.params["min_length"]) # msec - 1s
        self.max_num_frame_no_faces = int(self.params["no_face_detect"])
        self.max_check_faces = int(self.params["focus_face"])
        self.max_check_no_faces = int(self.params["no_focus_face"])
        
        
        self.out_dir   = os.path.abspath(out_dir)
        self.base_name = os.path.splitext(os.path.basename(self.params["path"]))[0]
        self.log_file_path = os.path.join(self.out_dir, self.base_name + ".log.txt")
        
    # __init__
    
    def OnInit(self, args):
        self.clip = VideoFileClip(self.params["path"])
        self.log_file = open(self.log_file_path,'at')
        self.log_file.writelines("Video: %s"%(self.params["path"]))
        wait = input("PRESS ENTER TO CONTINUE.")

    def OnExit(self, args):
        self.clip.close()
        self.log_file.close()
        
    def OnProcess(self, args):
        # TODO
        self.args = args
        if self.args.get("split") is None:
            self.args.update({"split" : ''})
        self.args.update({"timestamp_sec": args["timestamp"] / 1000.0})
        self.cur_time   = args["timestamp"] / 1000.0
        self.frame      = args["frame"]
        self.frame_rgb  = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                
        self.key_bboxes = []
        if self.detection == 'opencv':
            self.bboxes, _  = self.detector.detect(self.frame)
        else:
            self.bboxes, _  = self.detector.detect(self.frame_rgb)

        args.update({"num_faces": len(self.bboxes)})
        
        self.detect_face(self.frame_rgb, self.bboxes)
        self.update_key_faces()
        
        self.update_split_position()
        
        self.draw_bbox(self.frame, self.bboxes)
        
        super(FaceEmotionReader, self).OnProcess(args)
        
    # OnProcess
    
    def update_split_position(self):
        # Start Split (after previous end, and there are faces in scence)
        if self.is_splitting_process == False and len(self.bboxes)>0:
            self.cur_start = self.cur_time
            self.num_split = self.num_split + 1
            self.cnt_num_frame_no_faces = 0
            self.cnt_check_no_faces = 0
            
            self.emotion_vote = np.zeros(len(self.emotion_labels))
            
            self.key_track_id = []
            for i in range(min(len(self.key_faces), self.max_check_faces)):
                self.key_track_id.append(self.key_faces[i][1])
            
            self.is_splitting_process = True
        # if
        
        if self.is_splitting_process == True:
            if len(self.bboxes)==0:
                self.cnt_num_frame_no_faces = self.cnt_num_frame_no_faces + 1
            
            ff = False
            for i in range(len(self.key_bboxes)):
                if self.key_bboxes[i] in self.key_track_id:
                    ff = True
                    break
            if ff == False:    
                self.cnt_check_no_faces = self.cnt_check_no_faces + 1
            
            
            for i in range(min(3, len(self.key_faces))):
                idx = self.key_faces[i][2]
                self.emotion_vote[self.emotions[idx]] = self.emotion_vote[self.emotions[idx]] + 1
            self.emotion_sub_video = self.emotion_labels[np.argmax(self.emotion_vote)]
            
            self.cur_end = self.cur_time

            self.args["split"] = "[{start: .2f}-{dur:.2f}][{num:d},{noface:d},{nocheck:d}]{emotion:s}".format_map(
                {'start': self.cur_start, 
                 'dur': (self.cur_end - self.cur_start), 
                 'num': self.num_split,
                 'noface' : self.cnt_num_frame_no_faces, 
                 'nocheck': self.cnt_check_no_faces,
                 'emotion': self.emotion_sub_video}
            )
        # if
        
        # End Split (after previous end, and there are faces in scence)
        flag = False
        if self.is_splitting_process == True:
            if self.cur_time - self.cur_start>self.max_length: # over-time
                flag = True
            # if
            if self.cur_time - self.cur_start>=self.min_length and self.cur_time - self.cur_start<=self.max_length: # in-time
                if self.cnt_num_frame_no_faces > self.max_num_frame_no_faces: # over no detec faces
                    flag = True
                if self.cnt_check_no_faces > self.max_check_no_faces: # over no detec faces
                    flag = True
            # if 
        # if
        
        if flag == True:
            self.cur_end = self.cur_time
            self.args["split"] = ""
            self.is_splitting_process = False
            self.split_positions.append((self.cur_start, self.cur_end))
            
            print("Split %d: From [%.2f] to [%.2f], Time: %.2f, Emotion: %s"%(self.num_split, self.cur_start, 
                        self.cur_end, self.cur_end - self.cur_start, self.emotion_sub_video))
            self.log_file.writelines("Split %d: From [%.2f] to [%.2f], Time: %.2f, Emotion: %s"%(self.num_split, self.cur_start, 
                        self.cur_end, self.cur_end - self.cur_start, self.emotion_sub_video))
            
            subclip= self.clip.subclip(self.cur_start, self.cur_end)
            subclip.write_videofile(os.path.join(self.out_dir, self.base_name + ".{:05d}.{:s}.mp4".format(self.num_split, self.emotion_sub_video)), progress_bar=False)
            
            flag = False
        # if
        
        pass
        
    def update_key_faces(self):
        self.key_faces = []
        if len(self.key_bboxes)>0:
            key_cmp   = []
            key_id    = list(range(len(self.key_bboxes)))
            for key_bbox in self.key_bboxes:
                bbox = self.matching.feature_map[key_bbox]["bboxes"][-1]
                key_cmp.append(bbox[2] * bbox[3])
            self.key_faces = list(zip(key_cmp, self.key_bboxes, key_id))
            self.key_faces.sort(reverse=True)
        # if
    # def update_key_faces
        
    def detect_face(self, image, bboxes, **kwargs):
        self.faces = []
        if len(bboxes)>0:
            for i, d in enumerate(bboxes):
                (x, y, w, h) = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
                self.faces.append(image[y:y+w,x:x+w])
        self.emotions = []
        if len(bboxes)>0:
            self.emotions = self.emotion_detector.detect(self.faces)
        
        if self.initOnce == False and len(bboxes)>0:
            self.key_bboxes = self.matching.start(self.frame_rgb, bboxes)
            self.initOnce = True
        
        if self.initOnce == True:
            self.key_bboxes = self.matching.update(self.frame_rgb, bboxes)
        
        
    def draw_bbox(self, image, bboxes, **kwargs):
        nn = len(bboxes)
        if len(bboxes)>0:
            for i, d in enumerate(bboxes):
                (x, y, w, h) = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
                cv2.rectangle(image, (x, y),(x + w, y + h),(0,255,0),2)
                if (self.key_bboxes[i] in self.key_track_id):
                    self.box_text(image, '![%d (%s)]'%(self.key_bboxes[i], self.emotion_labels[self.emotions[i]]), (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2, (0, 0, 255))
                else:
                    self.box_text(image, '%d (%s)'%(self.key_bboxes[i], self.emotion_labels[self.emotions[i]]), (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2, (0, 255, 0))
                
    def box_text(self, image, text, point, font_face, font_scale, text_color, thickness, box_color):
        size, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
        (x, y) = point
        
        text_width = size[0]
        text_height = size[1]
        margin = 5
        cv2.rectangle(image, (x - margin, y - text_height - baseline - margin), (x + text_width + margin, y + margin), box_color, cv2.FILLED)
        cv2.putText(image, text, (x, y - baseline), font_face, font_scale, text_color, thickness)

# FaceEmotionReader

def parse_args(params = ''):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='.', help='input file')
    parser.add_argument('--out_dir', default='.', help='output dir')
    
    parser.add_argument('--show_scale', default= 0.5, help='window scale')
    parser.add_argument('--show_win',   default=True, help='show window')
    parser.add_argument('--verbose',    default=0, help='display log file')
    
    if params is None:
        return parser.parse_known_args()
    else:
        return parser.parse_known_args(params)

def main(**kwargs):
    args = {}
    args.update(kwargs)
    testvideo = FaceEmotionReader(path = args["path"], out_dir = args["out_dir"], detection = args["detection"], extra = args)
    testvideo.process()
# def main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Emotion Extract')
    parser.add_argument('--path', default='.', help='input file')
    parser.add_argument('--out_dir', default='.', help='output dir')
    parser.add_argument('--detection', default='dlib', help='face detection (mtcnn, dlib, opencv)')
    
    parser.add_argument('--max_length'     , default=5, help='Maximum subvideo to clip (second)')
    parser.add_argument('--min_length'     , default=2, help='Minimum subvideo to clip (second)')
    parser.add_argument('--no_face_detect' , default=5, help='Number of frames with no face detection')
    parser.add_argument('--focus_face'     , default=3, help='Number of face to focus to track and get emotion')
    parser.add_argument('--no_focus_face'  , default=5, help='Number of frame to lost tracking with faces')
    
    parser.add_argument('--show_scale', default= 0.5, help='window scale')
    parser.add_argument('--show_win',   default=True, help='show window')
    parser.add_argument('--verbose',    default=0, help='display log file')
    parser.add_argument('--debug',      default=False, help='step by step')

    args = parser.parse_args()

    main(path = args.path, out_dir = args.out_dir, detection = args.detection,
         show_scale = float(args.show_scale), show_win = bool(args.show_win), verbose = int(args.verbose),
         max_length = float(args.max_length), min_length = float(args.min_length),
         no_face_detect = int(args.no_face_detect), focus_face = int(args.focus_face),
         no_focus_face  = int(args.no_focus_face), debug = bool(args.debug))
# main