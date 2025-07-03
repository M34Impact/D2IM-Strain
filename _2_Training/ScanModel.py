from tensorflow.keras.layers import Layer, Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Multiply
from tensorflow.keras.models import Model

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
