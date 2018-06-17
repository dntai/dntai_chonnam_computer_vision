from .HungarianMatching import HungarianMatching
from prlab.detection.opencv.detection import OpenCVDetection
import cv2, os

DATA_DIR = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'data')

def test_tracker():
    tracker = HungarianMatching()
    detector= OpenCVDetection.getDetector()        
    show_video(detector = detector, tracker = tracker, source = 'video', path = os.path.join(DATA_DIR,'prlab_demo.avi'), extension = '.jpg')
    pass    
# test_tracker

def show_video(extra = {}, **kwargs):
    defaults = {'source': 'camera',     # [camera, video, files]
                'path': 0,              # device
                'extension': '.jpg',    # extension if source from files
                'debug'    : False ,    # step by step
                'delay'    : 5,         # waitkey time
                'detector' : None, 
                'tracker'  : None, 
    }
    args = defaults
    args.update(kwargs)
    args.update(extra)
    print(args['path'])
    # Load video or image sequence
    if args['source'] =='camera':
        video_capture = cv2.VideoCapture(int(args['path']))
        flag, frame = video_capture.retrieve()
        name = 'Camera [%d]'%(int(args['path']))
    elif args['source'] =='video':
        video_capture = cv2.VideoCapture(args["path"])
        flag, frame = video_capture.read()
        name = 'Video [%s]'%(args['path'])
    elif args['source'] =='files':
        files = [os.path.join(args["path"], x) for x in os.listdir(args["path"]) if x. endswith(args["extension"])]
        files.sort()
        name = "Files[%s]" % (args["path"])
        if len(files) > 0:
            frame = cv2.imread(files[0])
            flag  = True
    # if Load video

    if flag == False or frame is None:
        return False

    # image information
    height, width, channels = frame.shape

    print("[info] starting to read a sequence ...")

    # initialize
    frame_idx = 0
    frame_cnt = 0
    win_name  = name

    # tracking
    initOnce = False

    tracker  = args["tracker"]
    detector = args["detector"]

    # loop processing video
    while(True):
        # read frame
        if args["source"] == 'camera' or args["source"] == 'video':
            flag, frame = video_capture.read()
        elif args["source"] == 'files':
            if frame_idx >= len(files):
                break
            frame = cv2.imread(files[frame_idx])
        if frame is None:
            break
        frame_idx = frame_idx + 1
        frame_cnt = frame_cnt + 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_bgr = frame

        # process frame
        if tracker is not None and detector is not None:
            bboxes, _ = detector.detect(frame_bgr)
            if initOnce == True: # Something Else
                tracker.update(frame_rgb, bboxes)
            # if Something Else

            if initOnce == False: # First Time
                initOnce = True
                tracker.start(frame_rgb, bboxes)
            # if first time
            tracker.draw_bboxes(frame)
        # if

        # show frame
        cv2.imshow(win_name, frame);

        # next frame
        key = -1
        if args['debug'] == True:
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(args['delay'])

        if key == 27:
            break
    # end while

    cv2.destroyWindow(win_name)
    print("[info] stopping to read a sequence ...")
# test_tracker

def box_text(image, text, point, font_face, font_scale, text_color, thickness, box_color, margin = 5):
    size, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    (x, y) = point
    text_width = size[0]
    text_height = size[1]
    cv2.rectangle(image, (x - margin, y - text_height - baseline - margin), 
                  (x + text_width + margin, y + margin), box_color, cv2.FILLED)
    cv2.putText(image, text, (x, y - baseline), font_face, font_scale, text_color, thickness)
# box_text

def draw_bbox(self, image, bboxes, **kwargs):
    if len(bboxes)>0:
        for i, d in enumerate(bboxes):
            (x, y, w, h) = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
            cv2.rectangle(image, (x, y),(x + w, y + h), (0,255,0), 2)
            box_text(image, '%d'%(i), (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2, (0, 255, 0), 5)
# draw_bbox
