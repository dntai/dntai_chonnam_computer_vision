import os
import cv2
from enum import Enum

class OpenCVDetection(object):
    detector = None
    def __init__(self, model_path = None):
        if not model_path:
            model_path,_ = os.path.split(os.path.realpath(__file__))
        self.detector = cv2.CascadeClassifier(os.path.join(model_path,"model", "haarcascade_frontalface_alt.xml"))
        return

    # Singleton
    @staticmethod
    def getDetector():
        if OpenCVDetection.detector == None:
            OpenCVDetection.detector = OpenCVDetection()
        return OpenCVDetection.detector

    def detect(self, image): # (image bgr), (x, y, w, h)
        dets = self.detector.detectMultiScale(image, 1.3, 5)
        bboxes = []
        for i, d in enumerate(dets):
            bboxes.append([d[0], d[1], d[2], d[3]]);
        return  bboxes, None

    def draw_bbox(self, image, bboxes, **kwargs):
        if len(bboxes)>0:
            for i, d in enumerate(bboxes):
                (x, y, w, h) = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
                cv2.rectangle(image, (x, y),(x + w, y + h),(0,255,0),2)
        return
# OpenCVDetection