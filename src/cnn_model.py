import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
from constants import MODEL_PATH
import matplotlib.pyplot as plt

class CNN:

    # Constructor
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)

    # Prediction of the preprocessed image
    def predict(self, preprocessed_img):
        prediction = self.model.predict(preprocessed_img)
        return prediction

    # Outputs text with the prediction
    def prediction_output(self, prediction):
        final_pred = np.argmax(prediction, axis = 1)
        if final_pred == 0:
            title = "Office"
        else:
            title = "Screenshot"
        return title
        
    # Plot the input image
    def image_plot(self, img_path, title):
        img_plot = image.load_img(img_path)
        plt.imshow(img_plot)
        plt.title(title)
        plt.show()