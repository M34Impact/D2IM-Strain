import numpy as np

class Strain:
    def __init__(self, scan, mask, predictions) -> None:
        self.scan = scan
        self.mask = mask
        self.predictions = predictions

    def calculate_strain(displacement_w, dx):
        dw_dz = np.gradient(displacement_w, dx, axis=(0))
        ezz = dw_dz * 1e6
        # ezz = dw_dz
        return ezz

    def getStandardizedPezz(self):
        vs = 39  # real voxel size um
        ns = 50  # Node spacing
        mask2 = binary_erosion(mask, iterations=2)

        plot_num = self.scan.shape[0]
        predicted_ezz = []
        for i in range(plot_num):  # Loop through each sample
            resized_img = self.predictions[2][i].reshape(20, 20)

            ezz = self.calculate_strain(resized_img, ns)

            predicted_ezz.append(ezz)

        predicted_ezz = np.array(predicted_ezz)

        predicted_ezz = np.where(mask2, predicted_ezz, 0.0)

        standardized_pezz = np.where(mask2, (predicted_ezz - global_mean) / global_std, 0.0)