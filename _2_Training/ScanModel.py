from tensorflow.keras.layers import Layer, Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler

class ScanModel:
    def create_cnn(input_shape1, input_shape2, output_shape, dropout_rate, l2_lambda):
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
        # x = Dense(512, activation='relu', kernel_regularizer=l2(l2_lambda))(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)
        # x = Dense(512, activation='relu', kernel_regularizer=l2(l2_lambda))(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(dropout_rate)(x)

        x = BatchNormalization()(x)
        # x = Dense(512, activation='relu', kernel_regularizer=l2(l2_lambda))(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(dropout_rate)(x)

        x = Dense(output_shape[0] * output_shape[1], activation=None)(x)

        xm = Flatten()(input_layer2)
        output_layer_ezz = Multiply(name="out_ezz")([x, xm])

        model = Model(inputs=[input_layer1, input_layer2], outputs=[output_layer_ezz])
        return model

    def lr_schedule(epoch, lr):
        if epoch == 0:
            lr = 0.001
        if epoch == 200:
            lr = 0.001
        if epoch == 600:
            lr = 0.0001
        if epoch == 800:
            lr = 0.00001
        return lr

    # def train(self):
    #     checkpoint = ModelCheckpoint(
    #         'M1_best.h5',  # Path where the model will be saved
    #         monitor='val_loss',  # Monitor validation accuracy
    #         save_best_only=True,  # Only save when the monitored metric improves
    #         mode='min',  # Save when accuracy is at its maximum
    #         verbose=1  # Print a message when saving the model
    #     )
    #
    #     early_stopping = EarlyStopping(
    #         monitor='val_loss',  # Monitor validation accuracy
    #         patience=100,  # Number of epochs to wait for improvement
    #         restore_best_weights=True  # Restore model weights from the epoch with the best validation accuracy
    #     )
    #
    #     lr_scheduler = LearningRateScheduler(lr_schedule)
    #
    #     history = model.fit(
    #         train_gen,
    #         epochs=1000,
    #         # steps_per_epoch=len(input_data1) // batch_size,
    #         steps_per_epoch=4,
    #         validation_data=val_data,
    #         callbacks=[lr_scheduler, checkpoint]
    #     )
