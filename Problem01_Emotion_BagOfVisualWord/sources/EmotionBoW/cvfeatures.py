# https://github.com/jonasrothfuss/videofeatures/blob/master/videofeatures/CVFeatures.py

from .basefeatures import BaseFeatureExtractor

class SIFTFeatureExtractor(BaseFeatureExtractor):
    # todo: documentation
    def __init__(self, n_descriptors=5):
        self.n_descriptors = n_descriptors
    # __init__

    def computeFeatures(self, video):
        """
        Computes SIFT features for a single video.
        :param video: a video of shape (n_frames, width, height, channel)
        :return: the features, shape ()
        """
        descriptor_array = []
        for i in range(video.shape[0]):
            frame = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY).astype('uint8')
            _, descriptors = cv2.xfeatures2d.SIFT_create(nfeatures=self.n_descriptors).detectAndCompute(frame, None)

            if descriptors is not None:
                if descriptors.shape[0] < self.n_descriptors:
                    descriptors = np.concatenate([descriptors, np.zeros((self.n_descriptors - descriptors.shape[0], 128))], axis=0)
                else:
                    descriptors = descriptors[:self.n_descriptors]
            else:
                descriptors = np.zeros((self.n_descriptors, 128))

            assert descriptors.shape == (self.n_descriptors, 128)
            descriptor_array.append(descriptors)
        features = np.concatenate(descriptor_array, axis=0)
        return features

class SURFFeatureExtractor(BaseFeatureExtractor):
    # todo: documentation
    def __init__(self, n_descriptors=5):
        self.n_descriptors = n_descriptors

    def computeFeatures(self, video):
        """
        Computes SURF features for a single video.
        :param video: a video of shape (n_frames, width, height, channel)
        :return: the features, shape ()
        """
        descriptor_array = []
        for i in range(video.shape[0]):
      frame = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY).astype('uint8')
      _, descriptors = cv2.xfeatures2d.SURF_create().detectAndCompute(frame, None)

      # make sure that descriptors have shape (n_descriptor, 64)
      if descriptors is not None:
        if descriptors.shape[0] < self.n_descriptors:
          descriptors = np.concatenate([descriptors, np.zeros((self.n_descriptors - descriptors.shape[0], 64))],
                                       axis=0)
        else:
          descriptors = descriptors[:self.n_descriptors]
      else:
        descriptors = np.zeros((self.n_descriptors, 64))

      assert descriptors.shape == (self.n_descriptors, 64)
      descriptor_array.append(descriptors)

    return np.concatenate(descriptor_array, axis=0)