# model/myCAE.py

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

class CAE:
    def __init__(
        self,
        input_shape,
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
        """
        A convolutional autoencoder class with two convolutional layers, max pooling, and a dense layer.

        Args:
            input_shape (int): The shape of the input data.
            kernel_height1 (int): The height of the first convolutional layer's kernel.
            kernel_height2 (int): The height of the second convolutional layer's kernel.
            stride1 (int): The stride of the first convolutional layer.
            stride2 (int): The stride of the second convolutional layer.
            filter1 (int): The number of filters in the first convolutional layer.
            filter2 (int): The number of filters in the second convolutional layer.
            hidden_dim (int): The dimension of the hidden layer.
            pool1_size (tuple): The size of the first max pooling layer's pool.
            pool1_stride (tuple): The stride of the first max pooling layer.
            pool2_size (tuple): The size of the second max pooling layer's pool.
            pool2_stride (tuple): The stride of the second max pooling layer.
            optimizer (tf.keras.optimizers.Optimizer): The optimizer class to use.
            epochs (int): Number of training epochs.
            batch_size (int): Training batch size.
            validation_rate (float): Proportion of validation data during training.
            learning_rate (float): Learning rate used by the optimizer.
        """
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_rate = validation_rate
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.model = Sequential()

        # Encoder
        self.model.add(Conv2D(filter1, (1, kernel_height1), strides=(1, stride1), padding='same', activation='relu',
                              input_shape=(1, input_shape, 1)))
        self.model.add(MaxPooling2D(pool_size=pool1_size, strides=pool1_stride, padding='same'))
        self.model.add(Conv2D(filter2, (1, kernel_height2), strides=(1, stride2), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=pool2_size, strides=pool2_stride, padding='same'))
        self.model.add(Flatten())
        self.model.add(Dense(units=hidden_dim, name='HiddenLayer'))

        # Decoder
        flatten_len = input_shape // (stride1 * stride2 * pool1_stride[1] * pool2_stride[1])
        self.model.add(Dense(units=filter2 * flatten_len, activation='relu'))
        self.model.add(Reshape((1, flatten_len, filter2)))
        self.model.add(UpSampling2D(size=pool2_size))
        self.model.add(Conv2DTranspose(filter2, (1, kernel_height2), strides=(1, stride2), padding='same', activation='relu'))
        self.model.add(UpSampling2D(size=pool1_size))
        self.model.add(Conv2DTranspose(filter1, (1, kernel_height1), strides=(1, stride1), padding='same', activation='relu'))
        self.model.add(Conv2DTranspose(1, (1, 1), strides=(1, 1), padding='same', activation='sigmoid'))

    def fit(self, data):
        """
        Train the autoencoder model on the given data.

        Args:
            data (pd.DataFrame): Input data to be encoded and reconstructed.
        """
        self.model.compile(loss='mean_squared_error',
                           optimizer=self.optimizer(learning_rate=self.learning_rate))
        x = data.to_numpy().reshape(-1, 1, data.shape[1], 1)
        self.model.fit(x, x,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       validation_split=self.validation_rate)

    def extract_feature(self, x):
        """
        Extract features from the hidden layer of the trained autoencoder.

        Args:
            x (pd.DataFrame): Input data for feature extraction.

        Returns:
            np.ndarray: Feature representations from the hidden layer.
        """
        intermediate_model = tf.keras.Model(inputs=self.model.input,
                                            outputs=self.model.get_layer('HiddenLayer').output)
        x = x.to_numpy().reshape(-1, 1, x.shape[1], 1)
        return intermediate_model.predict(x)
