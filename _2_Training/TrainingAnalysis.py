import tensorflow as tf
from matplotlib import pyplot as plt, gridspec
import os
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr
import numpy as np
import tiffile as tiff
from sklearn.model_selection import train_test_split

# Coloured correlation analysis between predicted and observed values
class TrainingAnalysis:
    def __init__(self, scan_train, scan_val, scan_test,
                 mask_train, mask_val, mask_test,
                 strain_train, strain_val, strain_test,
                 strain_pred_train, strain_pred_val, strain_pred_test,
                 global_mean, global_std) -> None:
        self.scan_train = scan_train
        self.scan_val = scan_val
        self.scan_test = scan_test
        self.mask_train = mask_train
        self.mask_val = mask_val
        self.mask_test = mask_test
        self.strain_train = strain_train
        self.strain_val = strain_val
        self.strain_test = strain_test
        self.pezz_train = strain_pred_train
        self.pezz_val = strain_pred_val
        self.pezz_test = strain_pred_test
        self.input_data_test1 = None
        self.predictions = None
        self.global_mean = global_mean
        self.global_std = global_std

    def calculate_loss(self):
        path = r"C:\Users\kv7169h\PythonProjects\D2IM-Strain\_3_Main\M1_best.h5"
        model = tf.keras.models.load_model(path)
        # model.summary()

        num_channels = 1
        input_data1 = self.scan_train.reshape(self.scan_train.shape[0], self.scan_train.shape[1],
                                              self.scan_train.shape[2], num_channels)
        input_data_val1 = self.scan_val.reshape(self.scan_val.shape[0], self.scan_val.shape[1], self.scan_val.shape[2],
                                                num_channels)
        input_data2 = self.mask_train.reshape(self.mask_train.shape[0], self.mask_train.shape[1],
                                              self.mask_train.shape[2], num_channels)
        input_data_val2 = self.mask_val.reshape(self.mask_val.shape[0], self.mask_val.shape[1], self.mask_val.shape[2],
                                                num_channels)
        target_data_val_ezz = self.strain_val.reshape(self.strain_val.shape[0], -1)
        target_data_ezz = self.strain_train.reshape(self.strain_train.shape[0], -1)
        results = model.evaluate([input_data1, input_data2], [target_data_ezz])
        print("train loss, train acc:", results)

        results = model.evaluate([input_data_val1, input_data_val2], [target_data_val_ezz])
        print("val loss, val acc:", results)

        # Get test data into correct shape
        # TODO: - Redefine input_test_1 and similarly for others
        self.input_data_test1 = self.scan_test.reshape(self.scan_test.shape[0], self.scan_test.shape[1],
                                                  self.scan_test.shape[2], num_channels)
        input_data_test2 = self.mask_test.reshape(self.mask_test.shape[0], self.mask_test.shape[1],
                                                  self.mask_test.shape[2], num_channels)

        target_data_test_ezz = self.strain_test.reshape(self.strain_test.shape[0], -1)

        # Evaluate on test data
        results = model.evaluate([self.input_data_test1, input_data_test2], [target_data_test_ezz])
        print("test loss, test acc:", results)

        # Make predictions using test data
        self.predictions = model.predict([self.input_data_test1, input_data_test2])


    # Create a coloured correlation analysis between predicted and observed values
    def visualiseCorrelation(self):
        # Reimport scan data to get mappings to label each image with the vertebra
        # Define directory paths

        trainPath = r"C:\Users\kv7169h\PythonProjects\D2IM-Strain\data\Input\Scan"
        maskPath = r"C:\Users\kv7169h\PythonProjects\D2IM-Strain\data\Input\Mask"
        testPathW = r"C:\Users\kv7169h\PythonProjects\D2IM-Strain\data\Target\W"

        # Define function to import images
        def import_images_label(folder_path, name):
            image_data_list = []
            file_path = os.path.join(folder_path, name)
            img = tiff.imread(file_path)
            img[np.isnan(img)] = 0
            # Extract the identifier from the file name
            identifier = name.split("_")[0]  # Assuming the identifier is before the first underscore
            image_data_list.append((img, identifier))
            return image_data_list

        # Get the list of files in testPathU (this was chosen generically as files in all directories have the same name)
        test_w_files = set([f for f in os.listdir(testPathW) if os.path.isfile(os.path.join(testPathW, f))])

        bone_data = [[], [], [], [], []]
        for file in os.listdir(trainPath):
            if file in test_w_files:
                bone_data[0].extend(import_images_label(trainPath, file))

        ident_only = [data[1] for data in bone_data[0]]

        RS = 3623
        ident_NT, ident_test = train_test_split(ident_only, test_size=0.1, random_state=RS, shuffle=True)
        ident_test = np.array(ident_test)

        # Define the mappings
        # Slice numbers obtained using: np.unique(ident_test, return_counts=True)
        mapping = {
            'S1': "Intact 1 (6 Slices)",
            'S2': "Anterior Lesion 1 (3 Slices)",
            'S3': "Intact 2 (0 slices)",
            'S4': "Lateral Lesion 1 (2 Slices)",
            'S5': "Intact 3 (1 Slices)",
            'S6': "Lateral Lesion 2 (2 Slices)",
            'S7': "Intact 4 (2 Slices)",
            'S8': "Anterior Lesion 2 (5 Slices)",
            'S9': "Intact 5 (3 Slices)",
            'S10': "Anterior Lesion 3 (2 Slices)"
        }

        # Create a new variable with the updated strings based on the mappings
        new_ident_test = np.vectorize(mapping.get)(ident_test)

        ident_test_3d = new_ident_test.reshape((26, 1, 1))  # Reshape to 26x1x1

        # Now, use broadcasting to repeat the values along the other dimensions
        ident_test_3d = np.broadcast_to(ident_test_3d, (26, 20, 20))

        # Create a function to calculate correlation and plot the data with color-coded points
        def plot_correlation_with_colors(predicted_data, target_data, labels, title, title2):
            # Flatten the data if they are not already 1D arrays
            predicted_data = predicted_data.reshape(-1)
            target_data = target_data.reshape(-1)
            labels = labels.reshape(-1)

            # Create a filter to exclude data points where either value is 0
            non_zero_filter = (predicted_data != 0) & (target_data != 0)

            # Apply the filter to all datasets
            predicted_data = predicted_data[non_zero_filter]
            target_data = target_data[non_zero_filter]
            labels = labels[non_zero_filter]

            # Calculate the correlation coefficient
            correlation_coefficient, _ = pearsonr(predicted_data, target_data)

            # Calculate the coefficients for the line of best fit (linear regression)
            coefficients = np.polyfit(predicted_data, target_data, 1)

            # Create the linear regression line using the coefficients
            line_of_best_fit = np.poly1d(coefficients)

            # Create a colormap based on the number of unique labels
            num_labels = len(np.unique(labels))
            color_map = plt.cm.tab10

            # Create a scatter plot with color-coded points
            for label in np.unique(labels):
                label_indices = labels == label
                plt.scatter(predicted_data[label_indices], target_data[label_indices], label=label, alpha=0.3, s=12)
                plt.tick_params(labelsize=20)

            # Plot the linear regression line
            plt.plot(predicted_data, line_of_best_fit(predicted_data), color='black')

            plt.title(f'Correlation: $R^2 =$ {correlation_coefficient:.2f}', fontsize=30)
            plt.xlabel(f'Predicted {title2}', fontsize=25)
            plt.ylabel(f'Measured {title}', fontsize=25)
            plt.grid(True)

        vs = 39

        # Extract the data
        predicted_data_D2IM = self.pezz_test * vs
        predicted_data_D2IM_str = (self.predictions + self.global_mean) * self.global_std * vs
        predicted_data_D2IM_str = (self.predictions) * vs
        target_data_ezz = self.strain_test * vs

        # Create a figure with three subplots
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        # Plot correlations for U displacement
        plt.sca(axs[0])
        plot_correlation_with_colors(predicted_data_D2IM, target_data_ezz, ident_test_3d, '$ezz$',
                                     'Derived $\overline{ezz}$')

        # Plot correlations for V displacement
        plt.sca(axs[1])
        plot_correlation_with_colors(predicted_data_D2IM_str, target_data_ezz, ident_test_3d, '$ezz$',
                                     'Directly $\overline{ezz}$')

        # # Get the handles and labels for the legend from the first plot
        # handles, labels = axs[0,0].get_legend_handles_labels()
        # # Create the legend
        # legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.665, 0.2), title='Vertebra', title_fontsize=30, fontsize=25, scatterpoints=1)

        # # Increase the size of the legend markers and adjust their opacity
        # for handle in legend.legendHandles:
        #     handle.set_sizes([100])  # Increase the size of the points
        #     handle.set_alpha(1.0)   # Set the opacity of the points

        plt.tight_layout()
        # output_file = f"C:/Users/ps5375i/OneDrive - University of Greenwich/Documents/Research/Projects/Gianluca/Data/Outputs/2024/Correlations_coloured_largefont.jpg"  # Specify the output file name
        # plt.savefig(output_file, dpi=500, bbox_inches='tight')  # dpi controls the resolution (dots per inch)
        plt.show()

    def visualise(self):
        # Detailed plots with strain 4 cases used in paper
        # NOTE: the plot numbers do not currently match due to differnt data importing behaviour in kaggle

        plot_num = [3, 4, 9, 24]

        vs = 39  # real voxel size um
        ns = 50  # Node spacing

        for i in plot_num:  # Loop through each sample
            print("plot ", i)
            plt.figure(figsize=(25, 5))  # Adjust the figure size to accommodate three plots

            input_shape1 = (self.scan_train.shape[1], self.scan_train.shape[2], 1)
            input_shape2 = (self.mask_train.shape[1], self.mask_train.shape[2], 1)
            output_shape = (self.strain_train.shape[1], self.strain_train.shape[2])

            predicted_image = (np.flipud(self.predictions[i].reshape(output_shape)))
            target_image = np.flipud(self.strain_test[i])
            D2IM_strn = np.flipud(self.pezz_test[i])
            max_strerr_bar = np.max(np.abs(target_image - predicted_image))
            # min_d=np.array([target_image.min(),predicted_image.min()]).min()
            # max_d=np.array([target_image.max(),predicted_image.max()]).max()

            # # Load or generate the mask
            # msk = binary_erosion(input_data_test2[i].reshape(input_shape2[:2]), iterations = 2)

            # ezz_t = np.flipud(calculate_strain(np.flipud(target_image), ns*vs))
            # ezz_p = np.flipud(calculate_strain(np.flipud(predicted_image), ns*vs))
            # # Apply the mask to the strain field
            # ezz_t_i = np.where(msk, ezz_t.reshape(output_shape), 0.0)
            # ezz_p_i = np.where(msk, ezz_p.reshape(output_shape), 0.0)

            min_s = np.array([target_image.min(), predicted_image.min(), D2IM_strn.min()]).min()
            max_s = np.array([target_image.max(), predicted_image.max(), D2IM_strn.max()]).max()

            # Masking and value capping for clearer visualisations
            # if i == 4:
            #     threshold = -40000
            #     stress_mask = ezz_t_i < threshold
            #     ezz_t_i[stress_mask] = 0
            #     ezz_p_i[stress_mask] = 0
            #     min_s = -30000

            # if i == 24:
            #     threshold = -60000
            #     stress_mask = ezz_t_i < threshold
            #     ezz_t_i[stress_mask] = 0
            #     ezz_p_i[stress_mask] = 0

            # if i == 9:
            #     threshold1 = -129000
            #     threshold2 = -129500
            #     stress_mask = (ezz_t_i <= threshold1) & (ezz_t_i >= threshold2)
            #     ezz_t_i[stress_mask] = 0
            #     ezz_p_i[stress_mask] = 0

            min_s = np.array([target_image.min(), predicted_image.min()]).min()
            max_s = np.array([target_image.max(), predicted_image.max()]).max()

            # Create a grid of subplots
            gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1.065, 1.065])

            # Plot the input image
            ax3 = plt.subplot(gs[0, 0])  # Fourth subplot
            input_image = np.flipud(self.input_data_test1[i].reshape(input_shape1[:2]))
            im3 = ax3.imshow(input_image, cmap='gray', vmin=0, vmax=1)
            ax3.set_title("Input Image")
            ax3.axis('off')

            # Plot the ground truth target value
            ax1 = plt.subplot(gs[0, 1])  # First subplot
            im1 = ax1.imshow(target_image, cmap='plasma_r', vmin=min_s, vmax=max_s)
            ax1.set_title("Measured Strain ε$_{zz}$ (με))")
            ax1.axis('off')

            # Plot the predicted output
            ax2 = plt.subplot(gs[0, 2], sharey=ax1)  # Second subplot, sharing y-axis with the first subplot
            im2 = ax2.imshow(D2IM_strn, cmap='plasma_r', vmin=min_s, vmax=max_s)
            ax2.set_title("Derived Strain $ε_{zz}$ (με)")

            # Plot the predicted output
            ax2 = plt.subplot(gs[0, 3], sharey=ax1)  # Second subplot, sharing y-axis with the first subplot
            im2 = ax2.imshow(predicted_image, cmap='plasma_r', vmin=min_s, vmax=max_s)
            ax2.set_title("Predicted Strain $ε_{zz}$ (με)")

            # shared axes
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust the size and padding
            plt.colorbar(im2, cax=cax)  # Add colorbar to the right side
            cax.yaxis.set_ticks_position('right')  # Move the colorbar ticks to the left side
            ax2.axis('off')

            error = np.abs((target_image - predicted_image) / (target_image)) * 100
            error = np.nan_to_num(error, posinf=0, neginf=0)

            # Plot the displacement error
            ax3 = plt.subplot(gs[0, 4])  # First subplot
            im3 = ax3.imshow(error, cmap='jet', vmin=0, vmax=100)
            ax3.set_title("Strain Error (με): |ε$_{zz}-ε_{zz}$|")

            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust the size and padding
            plt.colorbar(im3, cax=cax)  # Add colorbar to the right side
            cax.yaxis.set_ticks_position('right')  # Move the colorbar ticks to the left side
            ax3.axis('off')

            plt.tight_layout()  # Ensure plots don't overlap
            plt.show()



