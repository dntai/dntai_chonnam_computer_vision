import scipy, cv2, os, numpy as np
from scipy.optimize import linear_sum_assignment
from .VggFaceDescription import VggFaceDescription


class HungarianMatching(object):
    def __init__(self):
        # key --> {"key": cnt, 'images': images[idx], "features": fvecs[idx], "bbox": face, "cur_active": True, 'num': n_idx}
        self.feature_map = {}
        self.feature_idx = 0
        self.feature_size= 15
        self.descriptor  = VggFaceDescription.getDescriptor()
        self.infinity  = 100000.0
        self.threshold = 0.3
        pass
    
    def start(self, image, bboxes):
        self.feature_map = {}
        self.feature_idx = 0
        key_bboxes = []
        for bbox in bboxes:
            fvecs, image_face, new_box = self.descriptor.predict(image, bbox)
            image_bbox = image[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2])]
            self.feature_map[self.feature_idx] = {'key': self.feature_idx, 'features': fvecs, 'images' : [image_bbox], 'bboxes': [bbox], 'active':True, 'num': 1}
            key_bboxes.append(self.feature_idx)
            self.feature_idx = self.feature_idx + 1                
        return key_bboxes
    # start
    

    # new bboxes (row) = old_boxes (cols)
    # cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    # from scipy.optimize import linear_sum_assignment
    # row_ind, col_ind = linear_sum_assignment(cost)
    #     [0] [1] [2]
    # [0]  4   1   3
    # [1]  2   0   5
    # [2]  3   2   2
    # row_ind --> array([0, 1, 2], dtype=int64)
    # col_ind --> array([1, 0, 2], dtype=int64)
    # (0,1),(1,0),(2,2)

    # new bboxes (row) < old_boxes (cols)
    # cost = np.array([[4, 1, 3], [2, 0, 5]])
    # from scipy.optimize import linear_sum_assignment
    # row_ind, col_ind = linear_sum_assignment(cost)
    #     [0] [1] [2]
    # [0]  4   1   3
    # [1]  2   0   5
    # row_ind --> array([0, 1], dtype=int64)
    # col_ind --> array([1, 0], dtype=int64)
    # (0,1),(1,0)

     # new bboxes (row) > old_boxes (cols)
    # cost = np.array([[4, 1], [2, 0], [3, 2]])
    # from scipy.optimize import linear_sum_assignment
    # row_ind, col_ind = linear_sum_assignment(cost)
    #     [0] [1]
    # [0]  4   1
    # [1]  2   0
    # [2]  3   2
    # row_ind --> array([0, 1], dtype=int64)
    # col_ind --> array([1, 0], dtype=int64)
    # (0,1),(1,0)
    def update(self, image, bboxes):
        
        # create weight matrix
        feature_map_keys   = list(self.feature_map.keys()) # matrix map idx --> keys        
        distance_matrix_feature_map = [] # distance matrix for feature map to apply Hungarian Alg.
        cur_features       = []
        cur_images_box     = []         
        cur_batch_fvecs    = []
        for idx, bbox in enumerate(bboxes):
            batch_fvecs, image_face, new_box = self.descriptor.predict(image, bbox)
            image_bbox = image[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2])]
            cur_images_box.append(image_bbox)
            
            # feature of bboxes
            feature = np.array(batch_fvecs).sum(axis=0) / len(batch_fvecs) # mean batch_fvecs
            cur_features.append(feature)
            cur_batch_fvecs.append(batch_fvecs)
        
            # calculate distance of current bbox to all feature map
            feature_map_distances = distance_face(self.feature_map, feature, self.threshold, self.infinity)                       
            
            # build distance_matrix_feature_maps with index linear
            distance_feature_map = []
            for i in range(len(feature_map_keys)): # create a row feature map distance of bbox to all bboxes
                distance_feature_map.append(feature_map_distances[feature_map_keys[i]])
            # bboxes > current feature map (expand array) 
            # --> easy to detect new box has map with old box? --> d[row][col] == inf --> row not match old or col outside keys
            for i in range(len(bboxes) - len(feature_map_keys)):  
                distance_feature_map.append(self.infinity)
            distance_matrix_feature_map.append(distance_feature_map) # row: new bboxes, col: old_bboxes (feature maps)
        # for

        # Hungarian method
        if len(bboxes)>0:
            row_ind, col_ind = linear_sum_assignment(distance_matrix_feature_map)

        # Assign old feature --> no active
        for key in feature_map_keys:
            self.feature_map[key]["active"] = False

        # Matching
        key_bboxes = []
        num_features = len(feature_map_keys)
        for idx, bbox in enumerate(bboxes):
            # Matched
            if col_ind[idx]<num_features and distance_matrix_feature_map[idx][col_ind[idx]] < self.infinity:
                matched_key = feature_map_keys[col_ind[idx]]
                fvecs       = self.feature_map[matched_key]["features"]
                # delete if over size
                if self.feature_map[matched_key]["num"] + 1 > self.feature_size:
                    fvecs = np.delete(fvecs, 0, axis = 0)
                    self.feature_map[matched_key]["num"] = self.feature_map[matched_key]["num"]  - 1
                    self.feature_map[matched_key]["images"].pop(0)
                    self.feature_map[matched_key]["bboxes"].pop(0)
                fvecs = np.append(fvecs, cur_batch_fvecs[idx], axis=0)
                self.feature_map[matched_key]["features"] = fvecs
                self.feature_map[matched_key]["images"].append(cur_images_box[idx])
                self.feature_map[matched_key]["bboxes"].append(bbox)
                self.feature_map[matched_key]["active"] = True
                self.feature_map[matched_key]["num"] = self.feature_map[matched_key]["num"] + 1
                key_bboxes.append(matched_key)
            else: # Unmatched
                self.feature_map[self.feature_idx] = {'key': self.feature_idx, 'features': cur_batch_fvecs[idx], 'images' : [cur_images_box[idx]], 'bboxes': [bbox], 'active':True, 'num': 1}
                key_bboxes.append(self.feature_idx)
                self.feature_idx = self.feature_idx + 1
                pass
            # if
        # for
        return key_bboxes
    # update
    
    def draw_bboxes(self, image):
        for key in self.feature_map.keys():
            cur_feature = self.feature_map[key]
            if cur_feature["active"] == True:
                bbox = cur_feature["bboxes"][-1]
                draw_bbox_text(image, bbox, "%d"%(key))
            # if
        pass
    # draw_bboxes
    
# class HungarianMatching

def distance_face(feature_map, feature, threshold, infinity_value): # faces_map = { "key": string, "features": features, ... }
    distances = {}
    keys = []
    for key in feature_map.keys():
        key_features = feature_map[key]["features"] # feature at key
        key_fvecs = np.array(key_features).sum(axis=0) / len(key_features) # mean features
        distance  = scipy.spatial.distance.cosine(key_fvecs, feature)
        if distance <= threshold:
            distances[key] = distance
        else:
            distances[key] = infinity_value
    return distances
#def    

def box_text(image, text, point, font_face, font_scale, text_color, thickness, box_color, margin = 5):
    size, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    (x, y) = point
    text_width = size[0]
    text_height = size[1]
    cv2.rectangle(image, (x - margin, y - text_height - baseline - margin), 
                  (x + text_width + margin, y + margin), box_color, cv2.FILLED)
    cv2.putText(image, text, (x, y - baseline), font_face, font_scale, text_color, thickness)
# box_text

def draw_bbox_text(image, bbox, text):
    (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    cv2.rectangle(image, (x, y),(x + w, y + h), (0,255,0), 2)
    box_text(image, text, (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2, (0, 255, 0), 5)
# draw_bbox
