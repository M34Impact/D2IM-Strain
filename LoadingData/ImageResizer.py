from LoadingData.FolderImageLoader import FolderImageLoader

class ImageResizer:
    trainPath = '../data/Input/Scan'
    maskPath = '../data/Input/Mask'
    testPathW = '../data/Target/W'
    folderPaths = [trainPath, maskPath, testPathW]
    allLoaders = []

    def get_resized_loaders(self):
        # Load all images
        for folderPath in self.folderPaths:
            loader = FolderImageLoader(folderPath, recursive=False)
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

        # Print resized images
        for loader in self.allLoaders:
            print(loader.print_details())
        return self.allLoaders