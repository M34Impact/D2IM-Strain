from _0_LoadingData.FolderImageLoader import FolderImageLoader
from _0_LoadingData.ImageResizer import ImageResizer
from _1_DataProcessing.DisplacementModel import DisplacementModel
from _1_DataProcessing.Masking import Masking
from _1_DataProcessing.Strain import Strain
from _2_Training.DataSpilt import DataSplit
from _2_Training.ScanModel import ScanModel
import numpy as np

if __name__ == "__main__":
    # Resize
    resizer = ImageResizer()
    allLoaders = resizer.get_resized_loaders()

    # Plot sample input scan
    resizer.visualise(example_index=69)

    # Plot sample mask
    mask_array = [image_loader.image for image_loader in allLoaders[1].images]
    mask_obj = Masking(mask_array)
    mask_filename_array = [image_loader.metadata['filename'] for image_loader in allLoaders[1].images]
    mask_obj.visualise(example_index=69, filenames=mask_filename_array)

    # Strain Calculation and Visualisation
    input_w_array = [image_loader.image for image_loader in allLoaders[2].images]
    be_mask = mask_obj.get_binary_erosion_mask()
    strain_obj = Strain(be_mask, input_w_array)
    strain_filename_array = [image_loader.metadata['filename'] for image_loader in allLoaders[2].images]
    strain_obj.visualise(example_index=69, filenames=strain_filename_array)

    # Strain Calculation from Original D2IM Model -- Not used
    # scan_array = [image_loader.image for image_loader in allLoaders[0].images]
    # bd_mask = mask_obj.get_binary_dilation_mask()
    # dp_model = DisplacementModel(scan_array, bd_mask, be_mask)
    # dp_model.visualise(example_index=69)
    # Data Splitting - Predicted Strains
    # pezz_split = DataSplit(dp_model.standardized_pezz)
    # pezz_train, pezz_val, pezz_test = pezz_split.split_data()

    # Data Splitting - Strain
    # strain_np_array = np.array(strain_obj.standardized_ezz)
    # strain_split = DataSplit(strain_np_array)
    strain_split = DataSplit(strain_obj.standardized_ezz)
    strain_train, strain_val, strain_test = strain_split.split_data()

    # Data Splitting - Scans
    scan_array = [image_loader.image for image_loader in allLoaders[0].images]
    # scan_np_array = np.array(scan_array)
    # scan_split = DataSplit(scan_np_array)
    scan_split = DataSplit(scan_array)
    scan_train, scan_val, scan_test = scan_split.split_data()

    # Data Splitting - Masks
    # mask_np_array = np.array(be_mask)
    # mask_split = DataSplit(mask_np_array)
    mask_split = DataSplit(be_mask)
    mask_train, mask_val, mask_test = mask_split.split_data()

    # Training CNN
    scan_model = ScanModel(scan_train, scan_val, mask_train, mask_val, strain_train, strain_val)
    scan_model.train()
