import tensorflow as tf
from matplotlib import pyplot as plt, gridspec
import os
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr
import numpy as np
import tiffile as tiff
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

from _2_Training.DataSpilt import DataSplit


# Coloured correlation analysis between predicted and observed values
class TrainingAnalysis:
    def __init__(self, scan_train, scan_val, scan_test,
                 mask_train, mask_val, mask_test,
                 strain_train, strain_val, strain_test,
                 strain_pred_train, strain_pred_val, strain_pred_test,
                 global_std, global_mean) -> None:
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
        self.global_std = global_std
        self.global_mean = global_mean

    # def dynamic_masked_mae(self, y_true, y_pred):
    #     # mask = tf.cast(y_true != 0, tf.float32)
    #     mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
    #     diff = tf.abs(y_true - y_pred) * mask
    #     return tf.reduce_sum(diff) / (tf.reduce_sum(mask) + 1e-8)

    def calculate_loss(self):
        path = r"C:\Users\kv7169h\PythonProjects\D2IM-Strain\_3_Main\M1_best.h5"
        # model = tf.keras.models.load_model(
        #     path,
        #     custom_objects={'dynamic_masked_mae': self.dynamic_masked_mae}
        # )
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
    def visualise_correlation(self):
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

        data_split = DataSplit(ident_only)
        ident_train, ident_val, ident_test = data_split.split_data()
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

        num_test_samples = self.predictions.shape[0]
        ident_test_3d = new_ident_test.reshape((num_test_samples, 1, 1))  # Reshape to 26x1x1

        # Now, use broadcasting to repeat the values along the other dimensions
        ident_test_3d = np.broadcast_to(ident_test_3d, (num_test_samples, 20, 20))

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
                plt.tick_params(labelsize=15)

            # Plot the linear regression line
            plt.plot(predicted_data, line_of_best_fit(predicted_data), color='black')

            plt.title(f'Correlation: $R^2 =$ {correlation_coefficient:.2f}', fontsize=30)
            plt.xlabel(f'Predicted {title2}', fontsize=25)
            plt.ylabel(f'Measured {title}', fontsize=25)
            plt.grid(True)

        vs = 39

        # Extract the data
        predicted_data_D2IM = self.pezz_test * vs
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

    def visualise_confusion_matrix(self):
        def create_confusion_matrix(predicted_data, target_data, threshold=10000, ax=None, title=""):
            # Flatten the data if they are not already 1D arrays
            predicted_data = predicted_data.reshape(-1)
            target_data = target_data.reshape(-1)

            # Classify as 1 if >= threshold, else 0
            predicted_classes = (abs(predicted_data) >= threshold).astype(int)
            target_classes = (abs(target_data) >= threshold).astype(int)

            # Generate confusion matrix
            cm = confusion_matrix(target_classes, predicted_classes)

            # Plot the confusion matrix in the specified subplot
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["Predicted <10000", "Predicted ≥10000"],
                        yticklabels=["Actual <10000", "Actual ≥10000"], ax=ax)
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Actual Class")
            ax.set_title(title)

        # Plot each confusion matrix with its own title
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # I need this step!
        predicted_data_D2IM = np.where(self.mask_test.reshape(26, 400),
                                       (self.pezz_test.reshape(26, 400) * self.global_std + self.global_mean), 0.0)
        target_data_ezz = np.where(self.mask_test.reshape(26, 400),
                                   (self.strain_test.reshape(26, 400) * self.global_std + self.global_mean), 0.0)
        predictions_model = np.where(self.mask_test.reshape(26, 400),
                                     (self.predictions * self.global_std + self.global_mean), 0.0)

        create_confusion_matrix(predicted_data_D2IM, target_data_ezz, ax=axs[0], title="Displacement Derived")
        create_confusion_matrix(predictions_model, target_data_ezz, ax=axs[1], title="Directly")

        plt.tight_layout()
        output_file = r"C:\Users\kv7169h\PythonProjects\D2IM-Strain\_4_Figure\confusion_matrix.jpg"  # Specify the output file name
        plt.savefig(output_file, dpi=500)  # dpi controls the resolution (dots per inch)
        plt.show()

    def visualise_4_plots(self):
        # Detailed plots with strain 4 cases used in paper
        # NOTE: the plot numbers do not currently match due to differnt data importing behaviour in kaggle

        plot_num = [3, 4, 9, 24]

        vs = 39  # real voxel size um
        ns = 50  # Node spacing

        for i in plot_num:  # Loop through each test sample
            # print(f"plot {i+1} of scan {scan_test_filenames[i]}")
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

            eps = 1e-8
            error2D = np.abs((target_image - predicted_image) / (target_image + eps)) * 100
            error2D = np.nan_to_num(error2D, posinf=0, neginf=0)

            # Plot the displacement error
            ax3 = plt.subplot(gs[0, 4])  # First subplot
            im3 = ax3.imshow(error2D, cmap='jet', vmin=0, vmax=100)
            ax3.set_title("Strain Error (με): |ε$_{zz}-ε_{zz}$|")

            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust the size and padding
            plt.colorbar(im3, cax=cax)  # Add colorbar to the right side
            cax.yaxis.set_ticks_position('right')  # Move the colorbar ticks to the left side
            ax3.axis('off')

            plt.tight_layout()  # Ensure plots don't overlap
            plt.show()

    # Create a coloured correlation analysis with legend
    def visualise_correlation_legend(self):
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

        data_split = DataSplit(ident_only)
        ident_train, ident_val, ident_test = data_split.split_data()
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

        num_test_samples = self.predictions.shape[0]
        ident_test_3d = new_ident_test.reshape((num_test_samples, 1, 1))  # Reshape to 26x1x1

        # Now, use broadcasting to repeat the values along the other dimensions
        ident_test_3d = np.broadcast_to(ident_test_3d, (num_test_samples, 20, 20))

        # Create a function to calculate correlation and plot the data with color-coded points
        def plot_correlation_with_colors(predicted_data, target_data, labels, title, title2, title3):
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
                plt.tick_params(labelsize=15)

            # Plot the linear regression line
            plt.plot(predicted_data, line_of_best_fit(predicted_data), color='black')

            plt.title(f'{title3}: $R^2 =$ {correlation_coefficient:.2f}', fontsize=30)
            plt.xlabel(f'Predicted {title2}', fontsize=25)
            plt.ylabel(f'Measured {title}', fontsize=25)
            plt.grid(True)

        # Extract the data
        predicted_data_D2IM = np.where(self.mask_test.reshape(26, 400),
                                       (self.pezz_test.reshape(26, 400) * self.global_std + self.global_mean), 0.0)
        target_data_ezz = np.where(self.mask_test.reshape(26, 400),
                                   (self.strain_test.reshape(26, 400) * self.global_std + self.global_mean), 0.0)
        predictions_model = np.where(self.mask_test.reshape(26, 400),
                                     (self.predictions * self.global_std + self.global_mean), 0.0)

        # Create a figure with three subplots
        fig, axs = plt.subplots(2, 2, figsize=(20, 18))

        # Plot correlations for U displacement
        plt.sca(axs[0, 0])
        plot_correlation_with_colors(predicted_data_D2IM, target_data_ezz, ident_test_3d, '$ezz$',
                                     'Derived $\overline{ezz}$', 'Displacement Derived')

        # Plot correlations for V displacement
        plt.sca(axs[0, 1])
        plot_correlation_with_colors(predictions_model, target_data_ezz, ident_test_3d, '$ezz$',
                                     'Directly $\overline{ezz}$', 'Directly')
        # Place for legend
        plt.sca(axs[1, 1])
        axs[1, 0].axis('off')
        axs[1, 1].axis('off')

        # # Get the handles and labels for the legend from the first plot
        handles, labels = axs[0,0].get_legend_handles_labels()
        # # Create the legend
        legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.15),
                            title='Vertebra', title_fontsize=30, fontsize=25, scatterpoints=1)

        # # Increase the size of the legend markers and adjust their opacity
        for handle in legend.legendHandles:
            handle.set_sizes([100])  # Increase the size of the points
            handle.set_alpha(1.0)   # Set the opacity of the points

        plt.tight_layout()
        output_file = r"C:\Users\kv7169h\PythonProjects\D2IM-Strain\_4_Figure\correlation_legend.jpg"
        plt.savefig(output_file, dpi=500, bbox_inches='tight')  # dpi controls the resolution (dots per inch)
        plt.show()

    def visualise_correlation_threshold(self):
        # Function to calculate correlation and plot the data with color-coded points
        def plot_correlation_with_colors(predicted_data, target_data, title, title2, title3):
            # Flatten the data if they are not already 1D arrays
            predicted_data = predicted_data.reshape(-1)
            target_data = target_data.reshape(-1)

            # Filter out data points where either value is 0
            non_zero_filter = (predicted_data != 0) & (target_data != 0)
            predicted_data = predicted_data[non_zero_filter]
            target_data = target_data[non_zero_filter]

            # Calculate correlation coefficient
            correlation_coefficient, _ = pearsonr(predicted_data, target_data)

            # Calculate line of best fit
            coefficients = np.polyfit(predicted_data, target_data, 1)
            line_of_best_fit = np.poly1d(coefficients)

            # Highlight points with target_data > 10,000
            highlight_filter = abs(target_data) > 10000
            colors = np.where(highlight_filter, 'red', 'blue')

            # Plot points with color highlighting
            plt.scatter(predicted_data, target_data, c=colors, alpha=0.5, s=12,
                        label='>10000' if np.any(highlight_filter) else '<=10000')
            plt.tick_params(labelsize=15)

            # Plot the line of best fit
            plt.plot(predicted_data, line_of_best_fit(predicted_data), color='black')

            # Plot title and labels
            plt.title(f'{title3}: $R^2 =$ {correlation_coefficient:.2f}', fontsize=30)
            plt.xlabel(f'Predicted {title2}', fontsize=25)
            plt.ylabel(f'Measured {title}', fontsize=25)
            plt.grid(True)

        # Extract the data
        predicted_data_D2IM = np.where(self.mask_test.reshape(26, 400),
                                           (self.pezz_test.reshape(26, 400) * self.global_std + self.global_mean), 0.0)
        target_data_ezz = np.where(self.mask_test.reshape(26, 400),
                                       (self.strain_test.reshape(26, 400) * self.global_std + self.global_mean), 0.0)
        predictions_model = np.where(self.mask_test.reshape(26, 400),
                                         (self.predictions * self.global_std + self.global_mean), 0.0)

        # Create a figure with three subplots
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))

        # Plot correlations for U displacement
        plt.sca(axs[0, 0])
        plot_correlation_with_colors(predicted_data_D2IM, target_data_ezz, '$ezz$',
                                     'Derived $\overline{ezz}$', 'Displacement Derived')

        # Plot correlations for V displacement
        plt.sca(axs[0, 1])
        plot_correlation_with_colors(predictions_model, target_data_ezz, '$ezz$',
                                     'Directly $\overline{ezz}$', 'Directly')
        # Place for legend
        plt.sca(axs[1, 1])
        axs[1, 0].axis('off')
        axs[1, 1].axis('off')

        # # Get the handles and labels for the legend from the first plot
        handles, labels = axs[0, 0].get_legend_handles_labels()
        # # Create the legend
        legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.52, 0.38),
                            title='Measured Value', title_fontsize=30, fontsize=25, scatterpoints=1)

        # # Increase the size of the legend markers and adjust their opacity
        for handle in legend.legendHandles:
            handle.set_sizes([100])  # Increase the size of the points
            handle.set_alpha(1.0)  # Set the opacity of the points

        plt.tight_layout()
        output_file = r"C:\Users\kv7169h\PythonProjects\D2IM-Strain\_4_Figure\correlation_10K.jpg"
        plt.savefig(output_file, dpi=500, bbox_inches='tight')  # dpi controls the resolution (dots per inch)
        plt.show()

    def visualise_box_plot(self):
        predicted_data_D2IM = np.where(self.mask_test.reshape(26, 400),
                                       (self.pezz_test.reshape(26, 400) * self.global_std + self.global_mean), 0.0)
        target_data_ezz = np.where(self.mask_test.reshape(26, 400),
                                   (self.strain_test.reshape(26, 400) * self.global_std + self.global_mean), 0.0)
        predictions_model = np.where(self.mask_test.reshape(26, 400),
                                     (self.predictions * self.global_std + self.global_mean), 0.0)

        # Relative error of displacement predictions

        # Ensure all arrays have the same shape
        predicted_data_1 = predicted_data_D2IM.reshape(-1)
        predicted_data_2 = predictions_model.reshape(-1)

        target_data_str = target_data_ezz.reshape(-1)

        # Create filters to exclude data points where either value is 0
        non_zero_filter_1 = (predicted_data_1 != 0) & (target_data_str != 0)
        non_zero_filter_2 = (predicted_data_2 != 0) & (target_data_str != 0)

        # Apply the filters to both datasets
        predicted_data_1 = predicted_data_1[non_zero_filter_1]
        predicted_data_2 = predicted_data_2[non_zero_filter_2]

        target_data_1 = target_data_str[non_zero_filter_1]
        target_data_2 = target_data_str[non_zero_filter_2]

        # Calculate relative errors
        relative_errors_1 = np.abs((predicted_data_1 - target_data_1) / target_data_1)
        relative_errors_2 = np.abs((predicted_data_2 - target_data_2) / target_data_2)

        # Create a box and whisker plot for all relative errors on the same plot
        fig, ax = plt.subplots(figsize=(8, 6))

        bp = ax.boxplot(
            [relative_errors_1 * 100,
             relative_errors_2 * 100],
            vert=True,
            showfliers=False,
            labels=['D2IM', 'Directly'],
            patch_artist=True,  # Color code the boxes
            medianprops={'color': 'black'}  # Set median line color to black
        )

        # Color code the boxes
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel('Strain Error (%)')
        ax.set_title('Relative Error of Strain Predictions without bone yield')

        plt.tight_layout()

        output_file = r"C:\Users\kv7169h\PythonProjects\D2IM-Strain\_4_Figure\box_error_without_bone_yield.jpg"  # Specify the output file name
        plt.savefig(output_file, dpi=500)  # dpi controls the resolution (dots per inch)
        plt.show()

    def visualise_box_plot2(self):
        predicted_data_D2IM = np.where(self.mask_test.reshape(26, 400),
                                       (self.pezz_test.reshape(26, 400) * self.global_std + self.global_mean), 0.0)
        target_data_ezz = np.where(self.mask_test.reshape(26, 400),
                                   (self.strain_test.reshape(26, 400) * self.global_std + self.global_mean), 0.0)
        predictions_model = np.where(self.mask_test.reshape(26, 400),
                                     (self.predictions * self.global_std + self.global_mean), 0.0)
        # Relative error of displacement predictions

        high_value_filter = abs(target_data_ezz) > 10000

        # Ensure all arrays have the same shape
        predicted_data_1 = predicted_data_D2IM[high_value_filter].reshape(-1)
        predicted_data_2 = predictions_model[high_value_filter].reshape(-1)

        target_data_str = target_data_ezz[high_value_filter].reshape(-1)

        # Create filters to exclude data points where either value is 0
        non_zero_filter_1 = (predicted_data_1 != 0) & (target_data_str != 0)
        non_zero_filter_2 = (predicted_data_2 != 0) & (target_data_str != 0)

        # Apply the filters to both datasets
        predicted_data_1 = predicted_data_1[non_zero_filter_1]
        predicted_data_2 = predicted_data_2[non_zero_filter_2]

        target_data_1 = target_data_str[non_zero_filter_1]
        target_data_2 = target_data_str[non_zero_filter_2]

        # Calculate relative errors
        relative_errors_1 = np.abs((predicted_data_1 - target_data_1) / target_data_1)
        relative_errors_2 = np.abs((predicted_data_2 - target_data_2) / target_data_2)

        # Create a box and whisker plot for all relative errors on the same plot
        fig, ax = plt.subplots(figsize=(8, 6))

        bp = ax.boxplot(
            [relative_errors_1 * 100,
             relative_errors_2 * 100],
            vert=True,
            showfliers=False,
            labels=['D2IM', 'Directly'],
            patch_artist=True,  # Color code the boxes
            medianprops={'color': 'black'}  # Set median line color to black
        )

        # Color code the boxes
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel('Strain Error (%)')
        ax.set_title('Relative Error of Strain Predictions with bone yield')

        plt.tight_layout()
        output_file = r"C:\Users\kv7169h\PythonProjects\D2IM-Strain\_4_Figure\box_error_with_bone_yield.jpg"
        plt.savefig(output_file, dpi=500)  # dpi controls the resolution (dots per inch)
        plt.show()

    # Detailed plots with strain 4 cases used in paper
    # NOTE: the plot numbers do not currently match due to differnt data importing behaviour in kaggle
    def visualise_strain(self):
        plot_num = [3, 4, 9, 24]

        vs = 39  # real voxel size um
        ns = 50  # Node spacing

        predicted_data_D2IM = np.where(self.mask_test.reshape(26, 400),
                                       (self.pezz_test.reshape(26, 400) * self.global_std + self.global_mean), 0.0)
        target_data_ezz = np.where(self.mask_test.reshape(26, 400),
                                   (self.strain_test.reshape(26, 400) * self.global_std + self.global_mean), 0.0)
        predictions_model = np.where(self.mask_test.reshape(26, 400),
                                     (self.predictions * self.global_std + self.global_mean), 0.0)

        input_shape1 = (self.scan_train.shape[1], self.scan_train.shape[2], 1)
        output_shape = (self.strain_train.shape[1], self.strain_train.shape[2])

        for i in plot_num:  # Loop through each sample
            print("plot ", i)
            plt.figure(figsize=(20, 10))  # Adjust the figure size to accommodate 2 plots

            predicted_image = (np.flipud(predicted_data_D2IM[i].reshape(output_shape)))
            target_image = np.flipud(target_data_ezz[i].reshape(output_shape))

            min_s = np.array([target_data_ezz[i].min(), predictions_model[i].min()]).min()
            max_s = np.array([target_data_ezz[i].max(), predictions_model[i].max()]).max()

            # Create a grid of subplots
            gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1.065, 1.065])

            # Plot the input image
            ax3 = plt.subplot(gs[0, 0])  # Fourth subplot
            input_image = np.flipud(self.input_data_test1[i].reshape(input_shape1[:2]))
            im3 = ax3.imshow(input_image, cmap='gray', vmin=0, vmax=1)
            ax3.set_title("Input Image")
            ax3.axis('off')

            # Plot the ground truth target value
            ax1 = plt.subplot(gs[0, 1])  # First subplot
            im1 = ax1.imshow(target_image, cmap='plasma_r', vmin=min_s, vmax=max_s)
            ax1.set_title("Measured Strain ε$_{zz}$ (με)")
            ax1.axis('off')

            # Plot the predicted output
            ax2 = plt.subplot(gs[0, 2], sharey=ax1)  # Second subplot, sharing y-axis with the first subplot
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
            ax3 = plt.subplot(gs[0, 3])  # First subplot
            # im3 = ax3.imshow((np.abs(ezz_t_i-ezz_p_i)/ezz_t_i)*100, cmap='jet')
            im3 = ax3.imshow(error, cmap='jet', vmin=0, vmax=100)
            # im3 = ax3.imshow(np.abs(ezz_t_i-ezz_p_i)/ezz_t_i, cmap='jet', vmin=0, vmax=max_strerr_bar)
            ax3.set_title("Strain Error (%): $|(ε_{zz}-\overline{ε}_{zz})/ε_{zz}$|")

            #### block 2

            predicted_image = (np.flipud(predictions_model[i].reshape(output_shape)))
            # Plot the input image
            ax3 = plt.subplot(gs[1, 0])  # Fourth subplot
            input_image = np.flipud(self.input_data_test1[i].reshape(input_shape1[:2]))
            im3 = ax3.imshow(input_image, cmap='gray', vmin=0, vmax=1)
            ax3.set_title("Input Image")
            ax3.axis('off')

            # Plot the ground truth target value
            ax1 = plt.subplot(gs[1, 1])  # First subplot
            im1 = ax1.imshow(target_image, cmap='plasma_r', vmin=min_s, vmax=max_s)
            ax1.set_title("Measured Strain ε$_{zz}$ (με)")
            ax1.axis('off')

            # Plot the predicted output
            ax2 = plt.subplot(gs[1, 2], sharey=ax1)  # Second subplot, sharing y-axis with the first subplot
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
            ax3 = plt.subplot(gs[1, 3])  # First subplot
            # im3 = ax3.imshow((np.abs(ezz_t_i-ezz_p_i)/ezz_t_i)*100, cmap='jet')
            im3 = ax3.imshow(error, cmap='jet', vmin=0, vmax=100)
            # im3 = ax3.imshow(np.abs(ezz_t_i-ezz_p_i)/ezz_t_i, cmap='jet', vmin=0, vmax=max_strerr_bar)
            ax3.set_title("Strain Error (%): $|(ε_{zz}-\overline{ε}_{zz})/ε_{zz}$|")
            ax3.axis('off')

            plt.tight_layout()  # Ensure plots don't overlap
            output_file = f"C:\\Users\kv7169h\PythonProjects\D2IM-Strain\_4_Figure\Strain_output_{i}.jpg"  # Specify the output file name
            plt.savefig(output_file, dpi=500)  # dpi controls the resolution (dots per inch)
            plt.show()

    def highlight_edges(self, ax, error, ezz_t_i, ezz_p_i, color, highlight):
        """
        Highlight specific regions in the error plot based on strain magnitudes

        Parameters:
        - ax: matplotlib axis object
        - error: error array to display
        - ezz_t_i: target strain values
        - ezz_p_i: predicted strain values
        - color: color for the highlight edges
        - highlight: 1 (low-low), 2 (high-low), 3 (high-high)
        """
        if highlight == 1:
            highlight_mask = (np.abs(ezz_p_i) < 10000) & (np.abs(ezz_t_i) < 10000) & (error > 0)
        elif highlight == 2:
            highlight_mask = (np.abs(ezz_p_i) > 10000) & (np.abs(ezz_t_i) < 10000) & (error > 0)
        elif highlight == 3:
            highlight_mask = (np.abs(ezz_p_i) > 10000) & (np.abs(ezz_t_i) > 10000) & (error > 0)

        # Apply transparency to the error plot outside of the highlight box
        im = ax.imshow(np.ma.masked_where(highlight_mask, error), cmap='jet', vmin=0, vmax=100, alpha=0.35)
        im = ax.imshow(np.ma.masked_where(~highlight_mask, error), cmap='jet', vmin=0, vmax=100, alpha=1.0)

        # Draw edges around highlighted regions
        for i in range(error.shape[0]):
            for j in range(error.shape[1]):
                if highlight_mask[i, j]:
                    if i == 0 or not highlight_mask[i - 1, j]:
                        ax.plot([j - 0.5, j + 0.5], [i - 0.5, i - 0.5], color=color, linewidth=7)
                    if i == error.shape[0] - 1 or not highlight_mask[i + 1, j]:
                        ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color=color, linewidth=7)
                    if j == 0 or not highlight_mask[i, j - 1]:
                        ax.plot([j - 0.5, j - 0.5], [i - 0.5, i + 0.5], color=color, linewidth=7)
                    if j == error.shape[1] - 1 or not highlight_mask[i, j + 1]:
                        ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color=color, linewidth=7)

        return im

    def visualise_strain_highlighted_comparison(self):
        """
        Visualize strain predictions with error highlighting
        - Row 1: D2IM predictions for 4 different scans
        - Row 2: Model predictions for the same 4 scans
        - Each cell shows the error with highlighting
        """
        plot_num = [24, 4, 3, 9]  # Four scans to display as columns

        # Prepare data
        predicted_data_D2IM = np.where(self.mask_test.reshape(26, 400),
                                       (self.pezz_test.reshape(26, 400) * self.global_std + self.global_mean), 0.0)
        target_data_ezz = np.where(self.mask_test.reshape(26, 400),
                                   (self.strain_test.reshape(26, 400) * self.global_std + self.global_mean), 0.0)
        predictions_model = np.where(self.mask_test.reshape(26, 400),
                                     (self.predictions * self.global_std + self.global_mean), 0.0)

        output_shape = (self.strain_train.shape[1], self.strain_train.shape[2])

        # Create figure with 2 rows x 4 columns
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1], hspace=0.05, wspace=0.05)

        im_list = []  # Store imshow objects for colorbar

        # ============================================================
        # ROW 1: D2IM Predictions (Derived Model)
        # ============================================================
        for col_idx, scan_idx in enumerate(plot_num):
            print(f"Processing D2IM - scan {scan_idx}")

            # Prepare images
            predicted_image_d2im = np.flipud(predicted_data_D2IM[scan_idx].reshape(output_shape))
            target_image = np.flipud(target_data_ezz[scan_idx].reshape(output_shape))

            # Calculate error
            error_d2im = np.abs((target_image - predicted_image_d2im) / target_image) * 100
            error_d2im = np.nan_to_num(error_d2im, posinf=0, neginf=0)

            # Create subplot
            ax = plt.subplot(gs[0, col_idx])
            im = self.highlight_edges(ax, error_d2im, target_image, predicted_image_d2im,
                                 color='black', highlight=1)
            im_list.append(im)

            # Add title only for first column
            if col_idx == 0:
                ax.set_ylabel('D2IM Model', fontsize=16, rotation=90, labelpad=10)

            ax.axis('off')

        # ============================================================
        # ROW 2: CNN Model Predictions
        # ============================================================
        for col_idx, scan_idx in enumerate(plot_num):
            print(f"Processing Model - scan {scan_idx}")

            # Prepare images
            predicted_image_model = np.flipud(predictions_model[scan_idx].reshape(output_shape))
            target_image = np.flipud(target_data_ezz[scan_idx].reshape(output_shape))

            # Calculate error
            error_model = np.abs((target_image - predicted_image_model) / target_image) * 100
            error_model = np.nan_to_num(error_model, posinf=0, neginf=0)

            # Create subplot
            ax = plt.subplot(gs[1, col_idx])
            im = self.highlight_edges(ax, error_model, target_image, predicted_image_model,
                                 color='green', highlight=3)
            im_list.append(im)

            # Add title only for first column
            if col_idx == 0:
                ax.set_ylabel('CNN Model', fontsize=16, rotation=90, labelpad=10)

            ax.axis('off')

        # ============================================================
        # Add Colorbar
        # ============================================================
        gs_colorbar = gridspec.GridSpec(1, 1)
        gs_colorbar.update(left=0.91, right=0.93, bottom=0.11, top=0.88)
        cbar_ax = fig.add_subplot(gs_colorbar[0])
        cbar = plt.colorbar(im_list[0], cax=cbar_ax)
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label('Error (%)', fontsize=20)

        plt.tight_layout()

        # Save figure
        output_file = r"C:\Users\kv7169h\PythonProjects\D2IM-Strain\_4_Figure\Strain_comparison_highlighted.jpg"
        plt.savefig(output_file, dpi=500, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.show()

