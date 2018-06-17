import os, dlib, cv2

class DlibDetection(object):
    detector = None
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector() 
        return

    # Singleton
    @staticmethod
    def getDetector():
        if DlibDetection.detector == None:
            DlibDetection.detector = DlibDetection()
        return DlibDetection.detector

    def detect(self, image):
        dets = self.detector(image, 1)
        bboxes = []
        for i, d in enumerate(dets):
            bboxes.append([d.left(), d.top(), d.right() - d.left(), d.bottom() - d.top()]);
        return  bboxes, None

    def draw_bbox(self, image, bboxes, **kwargs):
        if len(bboxes)>0:
            for i, d in enumerate(bboxes):
                (x, y, w, h) = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
                cv2.rectangle(image, (x, y),(x + w, y + h),(0,255,0),2)
        return
# DlibDetection

class DlibGpuDetection(object):
    detector = None
    def __init__(self, model_path = None):
        if not model_path:
            model_path,_ = os.path.split(os.path.realpath(__file__))
        self.detector = dlib.cnn_face_detection_model_v1(os.path.join(model_path, "model", "mmod_human_face_detector.dat"))
        return

    # Singleton
    @staticmethod
    def getDetector():
        if DlibGpuDetection.detector == None:
            DlibGpuDetection.detector = DlibGpuDetection(None)
        return DlibGpuDetection.detector

    def detect(self, image):
        dets = self.detector(image, 1)
        bboxes = []
        for i, d in enumerate(dets):
            bboxes.append([d.rect.left(), d.rect.top(), d.rect.right() - d.rect.left(),d.rect.bottom() - d.rect.top(), d.confidence]);
        return  bboxes, None

    def draw_bbox(self, image, bboxes, **kwargs):
        if len(bboxes)>0:
            for i, d in enumerate(bboxes):
                (x, y, w, h) = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
                cv2.rectangle(image, (x, y),(x + w, y + h),(0,255,0),2)
        return
# DlibGpuDetection

