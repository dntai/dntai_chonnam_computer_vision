# https://github.com/davidsandberg/facenet/blob/e4958e3b36d811e56882891a1836de0922944dbb/contributed/face.py

import numpy as np, cv2
import tensorflow as tf
from . import detect_face

class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None
# Face

class MtcnnFaceDetection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]	 # three steps's threshold
    factor = 0.709	# scale factor

    detector = None
    gpu_memory_fraction = 0.3

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    # Singleton
    @staticmethod
    def getDetector(face_crop_size=160, face_crop_margin=32):
        if MtcnnFaceDetection.detector == None:
            MtcnnFaceDetection.detector = MtcnnFaceDetection(face_crop_size, face_crop_margin)
        return MtcnnFaceDetection.detector

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MtcnnFaceDetection.gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []

        bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

            faces.append(face)

        return faces

    def detect(self, image):
        bounding_boxes, points = detect_face.detect_face(image, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        bbox = []
        prob = []
        for bb in bounding_boxes:
            x1 = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            y1 = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            x2 = np.minimum(bb[2] + self.face_crop_margin / 2, image.shape[1])
            y2 = np.minimum(bb[3] + self.face_crop_margin / 2, image.shape[0])
            bbox.append((x1, y1, x2 - x1, y2 - y1))
            prob.append(bb[4])

        return bbox, (prob, points)

    def draw_bbox(self, image, bboxes, **kwargs):
        if len(bboxes)>0:
            for i, d in enumerate(bboxes):
                (x, y, w, h) = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
                cv2.rectangle(image, (x, y),(x + w, y + h),(0,255,0),2)
        
        points = []
        if kwargs.get("points") is not None:
            points = kwargs.get("points")
            
        if len(points) > 0:
            pts = [(0,5),(1,6),(2,7),(3,8),(4,9)]
            n_faces  = len(points[0]) # col = num faces, row = features = 10
            for f in range(n_faces):
                for r in range(0, 5):
                    cv2.circle(image, (points[pts[r][0], f], points[pts[r][1], f]), 3, (0, 0, 255), 2)
        # if

        if kwargs.get("prob") is not None:
            prob = kwargs.get("prob")
            for f in range(n_faces):
                cv2.putText(image, "%.2f"%(prob[f]), (int(bboxes[f][0]), int(bboxes[f][1])), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
        # if

        return
# MtcnnFaceDetection