
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import LambdaCallback
import os

class CAE:
    def __init__(
            self,
            input_shape,
            data_type,
            kernel_height1=24,
            kernel_height2=12,
            stride1=5,
            stride2=4,
            filter1=32,
            filter2=64,
            hidden_dim=100,
            pool1_size=(2, 1),
            pool1_stride=(2, 1),
            pool2_size=(2, 1),
            pool2_stride=(2, 1),
            optimizer=tf.keras.optimizers.Adam,
            epochs=100,
            batch_size=32,
            validation_rate=0.2,
            learning_rate=0.0005
    ):
        self.data_type = data_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_rate = validation_rate
        self.learning_rate = learning_rate

        # Create a sequential model
        self.model = Sequential()

        # Encoder part
        self.model.add(Conv2D(filter1, (1, kernel_height1), strides=(1, stride1), padding='same', activation='relu',
                              input_shape=(1, input_shape, 1)))
        self.model.add(MaxPooling2D(pool_size=pool1_size, strides=pool1_stride, padding='same'))
        self.model.add(Conv2D(filter2, (1, kernel_height2), strides=(1, stride2), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=pool2_size, strides=pool2_stride, padding='same'))
        self.model.add(Flatten())
        self.model.add(Dense(units=hidden_dim, activation='relu', name='HiddenLayer'))

        # Decoder part
        self.model.add(Dense(units=filter2 * (input_shape // (stride1 * stride2 * pool1_stride[1] * pool2_stride[1])),
                             activation='relu'))
        self.model.add(Reshape((1, input_shape // (stride1 * stride2 * pool1_stride[1] * pool2_stride[1]), filter2)))
        self.model.add(UpSampling2D(size=pool2_size))
        self.model.add(Conv2DTranspose(filter2, (1, kernel_height2), strides=(1, stride2), padding='same', activation='relu'))
        self.model.add(UpSampling2D(size=pool1_size))
        self.model.add(Conv2DTranspose(filter1, (1, kernel_height1), strides=(1, stride1), padding='same', activation='relu'))
        self.model.add(Conv2DTranspose(1, (1, 1), strides=(1, 1), padding='same', activation='sigmoid'))

        self.optimizer = optimizer(lr=learning_rate)

    def fit(self, data):
        if not os.path.exists('CAEmodel'):
            os.makedirs('CAEmodel')
        # Define the model checkpoint callback
        checkpoint_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: self.model.save(f'CAEmodel/{self.data_type}_model_{epoch + 1:02d}.hdf5')
            if (epoch + 1) % 10 == 0 else None
        )

        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        data_train = tf.reshape(data.values, [-1, 1, data.shape[1], 1])
        self.model.fit(data_train, data_train,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       validation_split=self.validation_rate,
                       callbacks=[checkpoint_callback])

    def extract_feature(self, x):
        f = tf.keras.backend.function([self.model.input], [self.model.get_layer('HiddenLayer').output])
        x_reshaped = tf.reshape(x, [-1, 1, x.shape[1], 1])
        return f([x_reshaped])[0]



