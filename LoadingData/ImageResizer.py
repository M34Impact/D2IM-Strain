from matplotlib import pyplot as plt
from LoadingData.FolderImageLoader import FolderImageLoader
from pathlib import Path
import re

# Figure 1 panel typography. Matches TrainingAnalysis; titles are bold, as these
# panels sit beneath the hand-drawn panel A, whose headings are bold.
plt.rcParams['font.family'] = ['Arial', 'Helvetica', 'DejaVu Sans']
PANEL_FIGSIZE = (5, 3.75)
TITLE_SIZE = 20
LABEL_SIZE = 18
TICK_SIZE = 16

# Resolved from this file rather than the working directory, so the data is found
# whether the pipeline is launched from the project root or from Main/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ImageResizer:
    trainPath = str(PROJECT_ROOT / 'Data' / 'Input' / 'Scan')
    maskPath = str(PROJECT_ROOT / 'Data' / 'Input' / 'Mask')
    testPathW = str(PROJECT_ROOT / 'Data' / 'Target' / 'W')
    folderPaths = [trainPath, maskPath, testPathW]
    allLoaders = []

    def get_resized_loaders(self):
        # Load all images
        for folderPath in self.folderPaths:
            loader = FolderImageLoader(folderPath)
            summary = loader.get_summary()
            print(f"\nSummary: {summary}")
            self.allLoaders.append(loader)

        # Resize all images
        for i in range(len(self.allLoaders)):
            if self.allLoaders[i].get_images():
                if i == 0:  # Scan
                    resized_count = self.allLoaders[i].resize_all(256, 256, maintain_aspect=False)
                else:
                    resized_count = self.allLoaders[i].resize_all(20, 20, maintain_aspect=False)
                print(f"Resized {resized_count} images (maintaining aspect ratio)")

        return self.allLoaders

    def visualise(self, example_index):
        loader = self.allLoaders[0].images[example_index]
        plt.figure(figsize=PANEL_FIGSIZE)
        plt.imshow(loader.image, cmap='gray')
        plt.title(f"Example {example_index + 1} of Input Image {loader.metadata['filename']}",
                  fontsize=TITLE_SIZE, fontweight='bold')
        plt.xticks(fontsize=TICK_SIZE)
        plt.yticks(fontsize=TICK_SIZE)
        plt.colorbar().ax.tick_params(labelsize=TICK_SIZE)
        plt.tight_layout()
        plt.show()