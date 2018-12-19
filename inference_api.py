import cv2
import numpy as np
from tensorflow.contrib.predictor.saved_model_predictor import SavedModelPredictor
from utils.image_util import decode_labels


class ExportedModel:
    """
    This is an easy-to-use API for predicting the output based on an input
    """

    def __init__(self, filename, image_size):
        """
        Args:
        ----
        :param filename: Filename of the exported model
        :param image_size: Image size on which the model was trained
        """
        self.__image_size = image_size
        self.__predictor = self.load_exported_model(filename)

    def load_exported_model(self, filename):
        return SavedModelPredictor(filename)

    def predict(self, rgb_input):
        """
        This method takes NHWC input and then performs the prediction using the trained model
        ----
        Args:
        ----
        rgb_input: input to the trained model
        ----
        Return:
        ----
        rgb_input: after being resized to the specified size
        predictions: predicted classes [0,1,2,3, etc.]
        predictions_decoded: predictions with their colors decoded. See label_colours in image_util.py.
        """
        rgb_input = cv2.resize(rgb_input, (self.__image_size[1], self.__image_size[0]), interpolation=cv2.INTER_LINEAR)
        predictions = self.__predictor({'input': np.expand_dims(rgb_input, axis=0)})
        # Decode the classes into an RGB image
        predictions_decoded = decode_labels(predictions['output'])[0]
        return rgb_input, predictions, predictions_decoded
