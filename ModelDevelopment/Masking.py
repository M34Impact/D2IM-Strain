from scipy.ndimage import binary_dilation, binary_erosion

class Masking:
    def __init__(self, scan, mask, predictions) -> None:
        self.scan = scan
        self.mask = mask