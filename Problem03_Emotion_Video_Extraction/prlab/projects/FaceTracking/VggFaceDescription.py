from prlab.contrib.keras_vggface import VGGFace, utils
import cv2, os, numpy as np

module_dir     = os.path.abspath(os.path.split(__file__)[0])
data_dir       = os.path.join(module_dir, 'data')

predictor_path = os.path.join(module_dir, 'model', 'shape_predictor_5_face_landmarks.dat')

class VggFaceDescription(object):
    
    descriptor = {}

    def __init__(self, model = 'vgg16', **kwargs):      
        self.args = {
            'model'   : model, # resnet50, senet50
            'version' : -1, # vgg16(1), resnet50 (2), senet50 (2)
            'margin'  : 20,
        }
        self.args.update(kwargs)
        if model == 'vgg16':
            self.args['version'] = 1
        else:
            self.args['version'] = 2  
        self.model   = VGGFace(include_top=False, input_shape=(224, 224, 3), model=self.args['model'], pooling = "avg")
        
    # __init__

    def predict(self, image, bbox):
        image_face, new_box = crop_face(image, bbox, margin = self.args['margin'], size = 224)
        fvecs      = calculate_feature(image_face, self.model, self.args['version'])
        return fvecs, image_face, new_box
    # calculate

    def append_feature(self, fvecs, batch_fvecs):
        fvecs = np.append(fvecs, batch_fvecs, axis=0)
        return fvecs

    def mean_feature(self, fvecs):
        return np.array(fvecs).sum(axis=0) / len(fvecs)
        return fvecs

    @staticmethod
    def getDescriptor(model = 'vgg16', **kwargs):
        if VggFaceDescription.descriptor.get(model) is None:
            VggFaceDescription.descriptor[model] = VggFaceDescription(model = model, **kwargs)
        return VggFaceDescription.descriptor[model]
# VggFaceDescription  

def calculate_feature(image, model, version):
    if image.shape[0] != 224 and image.shape[1] != 224:
        image  =  cv2.resize(image, (224, 224))    
    if len(image.shape) == 2:
        image  = image.reshape(224, 224, 1)        
    if image.shape[2] == 1:
        image  = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)            
    if len(image.shape) == 3:
        images = np.expand_dims(image, axis=0)   
    images = utils.preprocess_input(images, version=version)  # 1 (VGG16), 2 (RESNET50 or SENET50)
    fvecs  = model.predict(images)
    return fvecs

def crop_face(imgarray, section, margin=20, size=224):
    """
    :param imgarray: full image
    :param section: face detected area (x, y, w, h)
    :param margin: add some margin to the face detected area to include a full head
    :param size: the result image resolution with be (size x size)
    :return: resized image in numpy array with shape (size x size x 3)
    """
    img_h, img_w, _ = imgarray.shape
    if section is None:
        section = [0, 0, img_w, img_h, None]
    x = section[0]
    y = section[1]
    w = section[2]
    h = section[3]
    x_a = int(x - margin)
    y_a = int(y - margin)
    x_b = int(x + w + margin)
    y_b = int(y + h + margin)
    if x_a < 0:
        x_b = min(x_b - x_a, img_w - 1)
        x_a = 0
    if y_a < 0:
        y_b = min(y_b - y_a, img_h - 1)
        y_a = 0
    if x_b > img_w:
        x_a = max(x_a - (x_b - img_w), 0)
        x_b = img_w
    if y_b > img_h:
        y_a = max(y_a - (y_b - img_h), 0)
        y_b = img_h
    cropped = imgarray[y_a: y_b, x_a: x_b]
    resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img, dtype=np.float64)
    return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)
# def

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    (x, y) = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


def identify_face(faces_map, features, threshold=(0.5, 1, 2, 3)): # faces_map = { "id":string, "features": feature face, ... }
    distances = []
    keys = []
    for key in faces_map:
        face_features = faces_map[key].get("features")
        distance = (10000, 1000, 1000, 1000)
        for feature in face_features:
            fvecs = np.array(feature).sum(axis=0) / len(face_features)
            d0 = sp.spatial.distance.cosine(feature, features)
            d1 = sp.spatial.distance.euclidean(feature, features)
            d2 = sp.spatial.distance.correlation(feature, features)
            d3 = sp.spatial.distance.cosine(feature, features)
            di = (d0, d1, d2, d3)
            if distance>di:
                distance = di
        #fvecs = np.array(face_features).sum(axis=0) / len(face_features)
        #distance = sp.spatial.distance.correlation(fvecs, features)
        # print(distance)
        distances.append(distance)
        keys.append(key)
    min_distance_value = min(distances)
    min_distance_index = distances.index(min_distance_value)
    if min_distance_value[0] < threshold[0] or min_distance_value[1] < threshold[1] or min_distance_value[2] < threshold[2] or min_distance_value[3] < threshold[3]:
        return (keys[min_distance_index], min_distance_value)
    else:
        return (-1, min_distance_value)
#def

def distance_face(faces_map, features): # faces_map = { "id":string, "features": feature face, ... }
    distances = {}
    keys = []
    for key in faces_map:
        face_features = faces_map[key].get("features")
        distance = 1000 # (10000, 1000, 1000, 1000)
#         for feature in face_features:
#             fvecs = np.array(feature).sum(axis=0) / len(face_features)
#             d0 = sp.spatial.distance.cosine(feature, features)
#             d1 = sp.spatial.distance.euclidean(feature, features)
#             d2 = sp.spatial.distance.correlation(feature, features)
#             d3 = sp.spatial.distance.cosine(feature, features)
#             di = d0 # 0 (d0, d1, d2, d3)
#             if distance>di:
#                 distance = di
        fvecs = np.array(face_features).sum(axis=0) / len(face_features)
        distance = sp.spatial.distance.cosine(fvecs, features)
        distances[key] = distance
    return distances
#def

def location_face(faces_map, bbox):
    distances = {}
    keys = []
    ssum = 0
    for key in faces_map:
        face_bbox = faces_map[key].get("bbox")
        face_show = faces_map[key].get("show")
        
        if face_show == True:
            face_center = (face_bbox[0] + face_bbox[2]/2.0, face_bbox[1] + face_bbox[3]/2.0)
            bbox_center = (bbox[0] + bbox[2]/2.0, bbox[1] + bbox[3]/2.0)
            distance    = math.sqrt((bbox_center[0] - face_center[0])*(bbox_center[0] - face_center[0]) + (bbox_center[1] - face_center[1])*(bbox_center[1] - face_center[1]))
            
            ssum = ssum + distance
        else:
            distance = 10000
        distances[key] = distance
    return (distances, ssum)