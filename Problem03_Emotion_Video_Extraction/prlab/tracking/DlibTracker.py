import cv2, dlib

class DlibTracker(object):
    """description of class"""
    def __init__(self):
        self.tracker = tracker = dlib.correlation_tracker()
        self.success = False
        self.bbox    = [0, 0, 0, 0]
        pass
    
    def init(self, frame, bbox):
        rect = dlib.rectangle(int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        self.tracker.start_track(frame, rect)
        self.success = True
        return self.success
        
    def update(self, frame):
        self.tracker.update(frame)
        bbox = self.tracker.get_position()
        self.success = True
        self.bbox = (int(bbox.left()), int(bbox.top()), int(bbox.right() - bbox.left()), int(bbox.bottom() - bbox.top()))
        return self.success, self.bbox

    def drawbbox(self, frame):
        if self.success == True:
            (x, y, w, h) = (int(self.bbox[0]), int(self.bbox[1]), int(self.bbox[2]), int(self.bbox[3]))
            cv2.rectangle(frame, (x, y),(x + w, y + h),(0,255,0),2)
   
    def getbbox(self):
        return self.bbox
   
    def state(self):
        return self.success
