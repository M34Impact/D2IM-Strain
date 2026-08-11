from scipy.ndimage import binary_dilation, binary_erosion
import numpy as np
import matplotlib.pyplot as plt

# Figure 1 panel typography. Matches TrainingAnalysis; titles are bold, as these
# panels sit beneath the hand-drawn panel A, whose headings are bold.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
PANEL_FIGSIZE = (5, 3.75)
TITLE_SIZE = 20
LABEL_SIZE = 18
TICK_SIZE = 16

class Masking:
    def __init__(self, masks_scan) -> None:
        self.maskArray = np.array(masks_scan, dtype=bool)
        self.bd_mask = binary_dilation(self.maskArray)
        self.be_mask = binary_erosion(self.bd_mask, iterations=2)

    def get_binary_dilation_mask(self):
        return self.bd_mask

    def get_binary_erosion_mask(self):
        return self.be_mask

    def visualise(self, example_index: int, filenames):
        plt.figure(figsize=PANEL_FIGSIZE)
        plt.imshow(self.bd_mask[example_index], cmap='gray')
        plt.title(f"Example {example_index + 1} of Mask Image {filenames[example_index]}",
                  fontsize=TITLE_SIZE, fontweight='bold')
        plt.xticks(fontsize=TICK_SIZE)
        plt.yticks(fontsize=TICK_SIZE)
        plt.colorbar().ax.tick_params(labelsize=TICK_SIZE)
        plt.tight_layout()
        plt.show()