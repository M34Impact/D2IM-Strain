import os
import glob
from typing import Optional, Tuple, List, Union, Dict

from _0_LoadingData.ImageLoader import ImageLoader
import matplotlib.pyplot as plt


class FolderImageLoader:
    def __init__(self, folder_path: str, recursive: bool = False):
        self.folder_path = folder_path
        self.recursive = recursive
        self.images = []
        self.failed_files = []
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
        self.load_all_images()

    def load_all_images(self) -> None:
        if not os.path.exists(self.folder_path):
            print(f"Folder not found: {self.folder_path}")
            return

        if not os.path.isdir(self.folder_path):
            print(f"Path is not a directory: {self.folder_path}")
            return

        # Get all image files
        image_files = self._get_image_files()

        print(f"Found {len(image_files)} image files in {self.folder_path}")

        # Load each image
        for file_path in image_files:
            try:
                loader = ImageLoader(file_path)
                if loader.image is not None:
                    self.images.append(loader)
                    print(f"✓ Loaded: {os.path.basename(file_path)}")
                else:
                    self.failed_files.append(file_path)
                    print(f"✗ Failed: {os.path.basename(file_path)}")
            except Exception as e:
                self.failed_files.append(file_path)
                print(f"✗ Error loading {os.path.basename(file_path)}: {str(e)}")

        print(f"\nSummary: {len(self.images)} loaded, {len(self.failed_files)} failed")

    def _get_image_files(self) -> List[str]:
        image_files = []

        if self.recursive:
            # Search recursively
            for ext in self.supported_extensions:
                pattern = os.path.join(self.folder_path, '**', f'*{ext}')
                image_files.extend(glob.glob(pattern, recursive=True))
                # Also search for uppercase extensions
                pattern = os.path.join(self.folder_path, '**', f'*{ext.upper()}')
                image_files.extend(glob.glob(pattern, recursive=True))
        else:
            # Search only in the main folder
            for ext in self.supported_extensions:
                pattern = os.path.join(self.folder_path, f'*{ext}')
                image_files.extend(glob.glob(pattern))
                # Also search for uppercase extensions
                pattern = os.path.join(self.folder_path, f'*{ext.upper()}')
                image_files.extend(glob.glob(pattern))

        # Remove duplicates and sort
        image_files = sorted(list(set(image_files)))
        return image_files

    def get_images(self) -> List[ImageLoader]:
        return self.images

    def get_failed_files(self) -> List[str]:
        return self.failed_files

    def get_summary(self) -> Dict:
        if not self.images:
            return {"total_images": 0, "message": "No images loaded"}

        formats = {}
        total_size = 0
        total_pixels = 0

        for img in self.images:
            info = img.get_info()
            fmt = info.get('format', 'Unknown')
            formats[fmt] = formats.get(fmt, 0) + 1
            total_size += info.get('file_size_bytes', 0)
            total_pixels += info.get('width', 0) * info.get('height', 0)

        return {
            "total_images": len(self.images),
            "failed_images": len(self.failed_files),
            "formats": formats,
            "total_file_size_mb": round(total_size / (1024 * 1024), 2),
            "total_pixels": total_pixels,
            "folder_path": self.folder_path,
            "recursive_search": self.recursive
        }

    def filter_by_size(self, min_width: int = 0, min_height: int = 0,
                       max_width: int = float('inf'), max_height: int = float('inf')) -> List[ImageLoader]:
        filtered = []
        for img in self.images:
            info = img.get_info()
            w, h = info['width'], info['height']
            if min_width <= w <= max_width and min_height <= h <= max_height:
                filtered.append(img)
        return filtered

    def filter_by_format(self, formats: List[str]) -> List[ImageLoader]:
        formats = [fmt.upper() for fmt in formats]
        filtered = []
        for img in self.images:
            if img.get_info().get('format', '').upper() in formats:
                filtered.append(img)
        return filtered

    def resize_all(self, width: int, height: int, maintain_aspect: bool = True) -> int:
        success_count = 0
        for img in self.images:
            if img.resize(width, height, maintain_aspect):
                success_count += 1
        return success_count

    def save_all(self, output_folder: str, prefix: str = "", suffix: str = "",
                 format: str = None, quality: int = 95) -> int:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        success_count = 0
        for img in self.images:
            try:
                filename = img.get_info()['filename']
                name, ext = os.path.splitext(filename)

                # Apply format change if specified
                if format:
                    ext = f".{format.lower()}"

                # Create new filename with prefix/suffix
                new_filename = f"{prefix}{name}{suffix}{ext}"
                output_path = os.path.join(output_folder, new_filename)

                if img.save(output_path, quality):
                    success_count += 1
                    print(f"✓ Saved: {new_filename}")
                else:
                    print(f"✗ Failed to save: {new_filename}")

            except Exception as e:
                print(f"✗ Error saving {img.get_info()['filename']}: {str(e)}")

        return success_count

    def print_details(self) -> None:
        """Print detailed information about all loaded images."""
        print(f"\n{'=' * 60}")
        print(f"FOLDER IMAGE LOADER DETAILS")
        print(f"{'=' * 60}")
        print(f"Folder: {self.folder_path}")
        print(f"Recursive: {self.recursive}")
        print(f"Total Images: {len(self.images)}")
        print(f"Failed Files: {len(self.failed_files)}")

        if self.images:
            print(f"\n{'Image Details:':<20} {'Format':<8} {'Size':<12} {'File Size':<10}")
            print("-" * 60)
            for img in self.images:
                info = img.get_info()
                size_str = f"{info['width']}x{info['height']}"
                file_size = f"{info['file_size_bytes'] / 1024:.1f}KB"
                print(f"{info['filename']:<20} {info['format']:<8} {size_str:<12} {file_size:<10}")

        if self.failed_files:
            print(f"\nFailed Files:")
            for failed in self.failed_files:
                print(f"  ✗ {os.path.basename(failed)}")

    @staticmethod
    def visualise(sample):
        plt.imshow(sample, cmap='gray')
        plt.title(f"Example 40 of Input Image")
        plt.colorbar()
        plt.show()