import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Original model for displacement.
class DisplacementModel:
    def __init__(self, scan, bd_mask, be_mask):
        self.scan = scan
        self.bd_mask = bd_mask
        self.be_mask = be_mask
        self.predictions = self.__get_predictions()
        self.standardized_pezz = self.__get_standardized_pezz()

    def __get_predictions(self):
        path = r"C:\Users\kv7169h\PythonProjects\D2IM-Strain\_3_Main\D2IM_trained.h5"
        displacement_model = tf.keras.models.load_model(path)
        # displacement_model.summary()
        predictions = displacement_model.predict([self.scan, self.bd_mask])
        return predictions

    # Unused function
    def get_predicted_w(self):
        plot_num = self.scan.shape[0]

        predicted_w = []
        for i in range(plot_num):  # Loop through each sample
            resized_img = self.predictions[2][i].reshape((20, 20))
            predicted_w.append(resized_img)

        predicted_w = np.array(predicted_w)
        return predicted_w

    def __calculate_strain(self, displacement_w, dx):
        dw_dz = np.gradient(displacement_w, dx, axis=0)
        ezz = dw_dz * 1e6
        # ezz = dw_dz
        return ezz

    def __get_standardized_pezz(self):
        vs = 39  # real voxel size um
        ns = 50  # Node spacing

        plot_num = self.scan.shape[0]
        predicted_ezz = []
        for i in range(plot_num):  # Loop through each sample
            resized_img = self.predictions[2][i].reshape(20, 20)

            ezz = self.__calculate_strain(resized_img, ns)

            predicted_ezz.append(ezz)

        predicted_ezz = np.array(predicted_ezz)
        predicted_ezz = np.where(self.be_mask, predicted_ezz, 0.0)

        global_mean = np.mean(predicted_ezz)
        global_std = np.std(predicted_ezz)
        standardized_pezz = np.where(self.be_mask, (predicted_ezz - global_mean) / global_std, 0.0)

        return standardized_pezz

    def visualise(self, example_index):
        plt.imshow(self.standardized_pezz[example_index], cmap='coolwarm')
        plt.title(f"Example {example_index + 1} of ezz strain from Displacement")
        plt.colorbar()
        plt.show()