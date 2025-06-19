from scipy.ndimage import binary_dilation, binary_erosion
import numpy as np
import matplotlib.pyplot as plt

class Masking:
    def __init__(self, masks_scan) -> None:
        self.maskArray = np.array(masks_scan, dtype=bool)
        self.bd_mask = binary_dilation(self.maskArray)
        self.be_mask = binary_erosion(self.bd_mask, iterations=2)

    def get_binary_dilation_mask(self):
        return self.bd_mask

    def get_binary_erosion_mask(self):
        return self.be_mask

    def visualise(self, example_index: int):
        plt.imshow(self.bd_mask[example_index], cmap='gray')
        plt.title(f"Example {example_index + 1} of Mask Image")
        plt.colorbar()
        plt.show()