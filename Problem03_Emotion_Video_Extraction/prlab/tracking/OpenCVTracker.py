import os, cv2
from enum import Enum

class OpenCVTrackerType(Enum):
    BOOSTING = 0
    MIL = 1
    KCF = 2
    TLD = 3
    MEDIANFLOW = 4    
    GOTURN = 5
    MOSSE = 6

class OpenCVTracker(object):
    """description of class"""
    def __init__(self, trackertype):# ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        self.trackerOpenCV = None
        if trackertype == OpenCVTrackerType.BOOSTING:
            self.trackerOpenCV = cv2.TrackerBoosting_create()
        elif trackertype == OpenCVTrackerType.MIL:    
            self.trackerOpenCV = cv2.TrackerMIL_create()
        elif trackertype == OpenCVTrackerType.KCF:    
            self.trackerOpenCV = cv2.TrackerKCF_create()
        elif trackertype == OpenCVTrackerType.TLD:    
            self.trackerOpenCV = cv2.TrackerTLD_create()
        elif trackertype == OpenCVTrackerType.MEDIANFLOW:    
            # self.trackerOpenCV = cv2.Tracker_create("MEDIANFLOW")
            self.trackerOpenCV = cv2.TrackerMedianFlow_create()
        elif trackertype == OpenCVTrackerType.GOTURN:    
            self.trackerOpenCV = cv2.TrackerGOTURN_create()
        elif trackertype == OpenCVTrackerType.MOSSE:    
            self.trackerOpenCV = cv2.TrackerMOSSE_create()

        self.success = False
        self.bbox    = [0, 0, 0, 0]
        pass
    def init(self, frame, bbox):
        self.success = self.trackerOpenCV.init(frame, (bbox[0],bbox[1],bbox[2],bbox[3]))
        return self.success
        
    def update(self, frame):
        self.success, self.bbox = self.trackerOpenCV.update(frame) 
        return self.success, self.bbox

    def drawbbox(self, frame):
        if self.success == True:
            (x, y, w, h) = (int(self.bbox[0]), int(self.bbox[1]), int(self.bbox[2]), int(self.bbox[3]))
            cv2.rectangle(frame, (x, y),(x + w, y + h),(0,255,0),2)
   
    def getbbox(self):
        return self.bbox
   
    def state(self):
        return self.success