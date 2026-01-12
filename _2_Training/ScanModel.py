from tensorflow.keras.layers import Layer, Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from matplotlib import pyplot as plt
from tensorflow.keras.losses import Huber
import tensorflow as tf
from tensorflow.keras import regularizers

class ScanModel:

    def __init__(self, scan_train, scan_val, mask_train, mask_val, strain_train, strain_val) -> None:
        self.scan_train = scan_train
        self.scan_val = scan_val
        self.mask_train = mask_train
        self.mask_val = mask_val
        self.strain_train = strain_train
        self.strain_val = strain_val

    def create_cnn(self, input_shape1, input_shape2, output_shape, dropout_rate, l2_lambda):
        input_layer1 = Input(shape=input_shape1)
        input_layer2 = Input(shape=input_shape2)

        x = input_layer1
        # Convolutional layers for the first image
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(dropout_rate)(x)

        x = Flatten()(x)
        x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
        # x = Dense(512, activation='relu')(x)
        x = Dropout(dropout_rate)(x)

        x = BatchNormalization()(x)
        x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
        # x = Dense(512, activation='relu')(x)
        x = Dropout(dropout_rate)(x)

        x = Dense(output_shape[0] * output_shape[1], activation=None)(x)

        xm = Flatten()(input_layer2)
        output_layer_ezz = Multiply(name="out_ezz")([x, xm])

        model = Model(inputs=[input_layer1, input_layer2], outputs=[output_layer_ezz])
        return model

    def lr_schedule(self, epoch, lr):
        if epoch < 600:
            return 0.001
        elif epoch < 800:
            return 0.0001
        else:
            return 0.00001

    # Create data generators
    def data_generator(self, input_data1, input_data2, target_data_1, batch_size):
        num_samples = input_data1.shape[0]
        while True:
            for start_idx in range(0, num_samples, batch_size):
                end_idx = start_idx + batch_size
                yield (
                    (input_data1[start_idx:end_idx], input_data2[start_idx:end_idx]),
                           target_data_1[start_idx:end_idx]
                )

    def train(self):
        # Define input shapes and output shape
        input_shape1 = (self.scan_train.shape[1], self.scan_train.shape[2], 1)
        input_shape2 = (self.mask_train.shape[1], self.mask_train.shape[2], 1)
        output_shape = (self.strain_train.shape[1], self.strain_train.shape[2])

        # Define regularisation and create the model
        dropout_rate = 0.2
        l2_lambda = 1e-6
        model = self.create_cnn(input_shape1, input_shape2, output_shape, dropout_rate, l2_lambda)
        model.summary()

        # Compile the model
        # model.compile(optimizer='adam', loss={'out_ezz': 'mean_squared_error'})
        model.compile(optimizer='adam', loss={'out_ezz': 'mean_absolute_error'})
        # model.compile(optimizer='adam', loss={'out_ezz': Huber(delta=1.0)})
        # model.compile(optimizer='adam', loss={'out_ezz': self.dynamic_masked_mae})

        # Reshape input and target data
        num_channels = 1
        input_data1 = self.scan_train.reshape(self.scan_train.shape[0], self.scan_train.shape[1], self.scan_train.shape[2], num_channels)
        input_data_val1 = self.scan_val.reshape(self.scan_val.shape[0], self.scan_val.shape[1], self.scan_val.shape[2], num_channels)
        input_data2 = self.mask_train.reshape(self.mask_train.shape[0], self.mask_train.shape[1], self.mask_train.shape[2], num_channels)
        input_data_val2 = self.mask_val.reshape(self.mask_val.shape[0], self.mask_val.shape[1], self.mask_val.shape[2], num_channels)
        target_data_ezz = self.strain_train.reshape(self.strain_train.shape[0], -1)
        target_data_val_ezz = self.strain_val.reshape(self.strain_val.shape[0], -1)

        # Train the model using the data generator
        batch_size = 50
        train_gen = self.data_generator(input_data1, input_data2, target_data_ezz, batch_size)
        val_data = ([input_data_val1, input_data_val2], [target_data_val_ezz])

        checkpoint = ModelCheckpoint(
            'M1_best.h5',  # Path where the model will be saved
            monitor='val_loss',  # Monitor validation accuracy
            save_best_only=True,  # Only save when the monitored metric improves
            mode='min',  # Save when accuracy is at its maximum
            verbose=1  # Print a message when saving the model
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',  # Monitor validation accuracy
            patience=100,  # Number of epochs to wait for improvement
            restore_best_weights=True  # Restore model weights from the epoch with the best validation accuracy
        )

        lr_scheduler = LearningRateScheduler(self.lr_schedule)

        history = model.fit(
            train_gen,
            epochs=1000,
            # steps_per_epoch=len(input_data1) // batch_size,
            steps_per_epoch=4,
            validation_data=val_data,
            callbacks=[lr_scheduler, checkpoint]
        )

        self.visualise(history)

    def visualise(self, history):
        # Plot training and validation accuracy
        plt.plot(history.history['loss'][50:], label='Training Loss')
        plt.plot(history.history['val_loss'][50:], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    # MAE over masked area only
    # def dynamic_masked_mae(self, y_true, y_pred):
    #     # mask = tf.cast(y_true != 0, tf.float32)
    #     mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
    #     diff = tf.abs(y_true - y_pred) * mask
    #     return tf.reduce_sum(diff) / (tf.reduce_sum(mask) + 1e-8)