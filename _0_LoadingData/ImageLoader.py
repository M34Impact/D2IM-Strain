import os
from PIL import Image, ImageEnhance
import numpy as np
import tiffile as tiff
from scipy.ndimage import zoom

# Single File
class ImageLoader:
    def __init__(self, file_path: str):
        self.img = None
        self.img_array = None
        self.file_path = file_path
        self.image = None
        self.original_image = None
        self.metadata = {}
        self.load_image()

    def load_image(self) -> bool:
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"Image file not found: {self.file_path}")

            self.img = tiff.imread(self.file_path)
            self.img[np.isnan(self.img)] = 0
            self.image = Image.fromarray(self.img)

            self.original_image = self.image.copy()

            # Store metadata
            self.metadata = {
                'filename': os.path.basename(self.file_path),
                'filepath': self.file_path,
                'format': self.image.format,
                'mode': self.image.mode,
                'size': self.image.size,
                'width': self.image.width,
                'height': self.image.height,
                'file_size_bytes': os.path.getsize(self.file_path)
            }

            return True

        except Exception as e:
            print(f"Error loading image {self.file_path}: {str(e)}")
            return False

    def get_info(self) -> dict:
        """Get image metadata."""
        return self.metadata.copy()

    def resize(self, width: int, height: int, maintain_aspect: bool = True) -> bool:
        """Resize the image."""
        if not self.image:
            return False

        try:
            if maintain_aspect: # Not called
                self.image.thumbnail((width, height), Image.Resampling.NEAREST)
            else:
                self.image = zoom(self.img, (width/self.img.shape[0], height/self.img.shape[1]), mode='nearest', order=0)
            self.metadata['size'] = self.image.size
            self.metadata['width'] = self.image.width
            self.metadata['height'] = self.image.height
            return True
        except Exception as e:
            print(f"Error resizing image: {str(e)}")
            return False

    def save(self, output_path: str, quality: int = 95) -> bool:
        """Save the image to a file."""
        if not self.image:
            return False

        try:
            ext = os.path.splitext(output_path)[1].lower()
            if ext in ['.jpg', '.jpeg']:
                self.image.save(output_path, 'JPEG', quality=quality)
            elif ext == '.png':
                self.image.save(output_path, 'PNG')
            else:
                self.image.save(output_path)
            return True
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return False