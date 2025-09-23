import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

class Strain:
    def __init__(self, be_mask, w_input_images) -> None:
        self.be_mask = be_mask
        self.w_input_images = w_input_images
        self.standardized_ezz = self.__get_standardized_ezz()

    def __calculate_strain(self, displacement_w, dx):
        dw_dz = np.gradient(displacement_w, dx, axis=0)
        ezz = dw_dz * 1e6
        # ezz = dw_dz
        return ezz

    def __get_standardized_ezz(self):
        # vs = 39  # real voxel size um
        ns = 50  # Node spacing

        strain_data = []
        for img in self.w_input_images:
            strain = self.__calculate_strain(img, ns)
            strain_data.append(strain)

        output_ims_w = np.array(self.w_input_images)
        target_ezz = np.array(strain_data)
        target_ezz = np.where(self.be_mask, target_ezz, 0.0)

        # Standardisation
        global_mean = np.mean(target_ezz)
        global_std = np.std(target_ezz)
        standardized_ezz = np.where(self.be_mask, (target_ezz - global_mean) / global_std, 0.0)

        # Log Normalisation
        # log_normalized_ezz = np.where(self.be_mask, np.log(target_ezz), 0.0)
        # # Normalise to [0, 1] if you want
        # log_vals = log_normalized_ezz
        # log_min = np.min(log_vals[self.be_mask])
        # log_max = np.max(log_vals[self.be_mask])
        # log_normalized_scaled = np.where(self.be_mask, (log_vals - log_min) / (log_vals - log_max), 0.0)

        return target_ezz

    def visualise(self, example_index, filenames):
        plt.imshow(self.standardized_ezz[example_index], cmap='coolwarm')
        plt.title(f"Example {example_index + 1} of Strain Image {filenames[example_index]}")
        plt.colorbar()
        plt.show()

    def visualiseW(self, example_index, filenames):
        plt.imshow(self.w_input_images[example_index], cmap='coolwarm')
        plt.title(f"Example {example_index + 1} of W Image {filenames[example_index]}")
        plt.colorbar()
        plt.show()