import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

class ScanLosoModel:
    def __init__(self, scan_all, mask_all, strain_all, save_dir="saved_models_loso_intact_4"):
        self.scan_all = scan_all
        self.mask_all = mask_all
        self.strain_all = strain_all
        self.save_dir = save_dir
        self.slice_counts_dict = {
            "S3_INT": 24,
            "S5_INT": 26,
            "S7_INT": 26,
            "S9_INT": 25
        }
        os.makedirs(save_dir, exist_ok=True)

    # -------------------------------
    # CNN Model
    # -------------------------------
    def create_cnn(self, input_shape1, input_shape2, output_shape, dropout_rate=0.5):
        input_layer1 = Input(shape=input_shape1)
        input_layer2 = Input(shape=input_shape2)

        x = input_layer1
        # Convolutional backbone
        for filters in [32, 64, 128, 256]:
            x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(dropout_rate)(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(output_shape[0] * output_shape[1], activation=None)(x)

        xm = Flatten()(input_layer2)
        output_layer = Multiply(name="out_ezz")([x, xm])

        model = Model(inputs=[input_layer1, input_layer2], outputs=[output_layer])
        return model

    # -------------------------------
    # Learning Rate Schedule
    # -------------------------------
    def lr_schedule(self, epoch, lr):
        if epoch >= 800:
            lr = 0.00001
        if epoch >= 600:
            lr = 0.0001
        if epoch >= 200:
            lr = 0.001
        if epoch >= 0:
            lr = 0.001
        return lr

    # -------------------------------
    # Leave-One-Scan-Out Training
    # -------------------------------
    def train_loso(self):
        start_idx = 0
        scan_indices = {}
        for name, count in self.slice_counts_dict.items():
            scan_indices[name] = (start_idx, start_idx + count)
            start_idx += count

        fold_r2_scores = {}

        for scan_name, (start, end) in scan_indices.items():
            print(f"\n===== Training (leaving out {scan_name}) =====")

            # Split data
            scan_test = self.scan_all[start:end]
            mask_test = self.mask_all[start:end]
            strain_test = self.strain_all[start:end]

            scan_train = np.concatenate([self.scan_all[:start], self.scan_all[end:]])
            mask_train = np.concatenate([self.mask_all[:start], self.mask_all[end:]])
            strain_train = np.concatenate([self.strain_all[:start], self.strain_all[end:]])

            # Reshape for CNN
            scan_train = scan_train[..., np.newaxis]
            mask_train = mask_train[..., np.newaxis]
            scan_test = scan_test[..., np.newaxis]
            mask_test = mask_test[..., np.newaxis]

            strain_train = strain_train.reshape(strain_train.shape[0], -1)
            strain_test = strain_test.reshape(strain_test.shape[0], -1)

            # Define shapes
            input_shape1 = scan_train.shape[1:]
            input_shape2 = mask_train.shape[1:]
            output_shape = (self.strain_all.shape[1], self.strain_all.shape[2])

            # Create and compile model
            model = self.create_cnn(input_shape1, input_shape2, output_shape)
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Callbacks
            model_path = os.path.join(self.save_dir, f"fold_{scan_name}_best.h5")
            checkpoint = ModelCheckpoint(model_path, monitor='loss', save_best_only=True, mode='min', verbose=1)
            early_stopping = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
            lr_scheduler = LearningRateScheduler(self.lr_schedule)

            # Train
            history = model.fit(
                [scan_train, mask_train],
                strain_train,
                epochs=300,
                batch_size=8,
                verbose=1,
                callbacks=[checkpoint, early_stopping, lr_scheduler]
            )

            # Predict on left-out scan
            preds = model.predict([scan_test, mask_test])
            r2 = r2_score(strain_test.flatten(), preds.flatten())
            fold_r2_scores[scan_name] = r2

            print(f"✅ Fold {scan_name} R²: {r2:.4f}")

            # Visualize training loss
            plt.figure()
            plt.plot(history.history['loss'], label='Train Loss')
            plt.title(f'Training Loss - {scan_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

        # Print summary
        print("\n===== R² Summary =====")
        for scan, r2 in fold_r2_scores.items():
            print(f"{scan}: {r2:.4f}")

        mean_r2 = np.mean(list(fold_r2_scores.values()))
        print(f"\nAverage R² across all folds: {mean_r2:.4f}")

        return fold_r2_scores