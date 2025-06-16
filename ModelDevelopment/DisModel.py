import tensorflow as tf
import numpy as np

class DisModel:
    def __init__(self, scan, mask):
        self.scan = scan
        self.mask = mask

    def getPredictions(self):
        dispModel = tf.keras.models.load_model('D2IM_trained.h5')
        predictions = dispModel.predict([self.scan, self.mask])
        return predictions

    def getPredictedW(self):
        plot_num = self.scan.shape[0]
        predictions = self.getPredictions()

        predicted_W = []
        for i in range(plot_num):  # Loop through each sample
            resized_img = predictions[2][i].reshape((20, 20))
            predicted_W.append(resized_img)

        predicted_W = np.array(predicted_W)
        return predicted_W