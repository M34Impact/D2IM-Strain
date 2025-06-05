from LoadingData.FolderImageLoader import FolderImageLoader

if __name__ == "__main__":
    print("=== Folder Image Loader Started ===")

    trainPath = '../data/Input/Scan'
    maskPath = '../data/Input/Mask'
    testPathW = '../data/Target/W'
    folderPaths = [trainPath, maskPath, testPathW]
    allLoaders = []

    # Load all images
    for folderPath in folderPaths:
        loader = FolderImageLoader(folderPath, recursive=False)
        summary = loader.get_summary()
        print(f"\nSummary: {summary}")
        allLoaders.append(loader)

    # Resize all images
    for i in range(len(allLoaders)):
        if allLoaders[i].get_images():
            if i == 0: # Scan
                resized_count = allLoaders[i].resize_all(256, 256, maintain_aspect=True)
            else:
                resized_count = allLoaders[i].resize_all(20, 20, maintain_aspect=True)
            print(f"Resized {resized_count} images (maintaining aspect ratio)")

            # Save all processed images -- Optional
            # output_folder = "./processed_images"
            # saved_count = loader.save_all(output_folder, prefix="processed_", format="jpg", quality=90)
            # print(f"Saved {saved_count} processed images to {output_folder}")

    # Print resized images
    for loader in allLoaders:
        print(loader.print_details())