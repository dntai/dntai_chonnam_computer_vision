
class BaseFeatureExtractor:
    def computeFeatures(self, frames_batch): # (frames, width, height, channel)
        raise NotImplementedError
# BaseFeatureExtractor