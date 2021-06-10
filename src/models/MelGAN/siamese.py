import tensorflow as tf
from tensorflow import keras as K


class Siamese:
    def __init__(self, frequencies, sliceWidth, filters, latentDim):
        self.frequencies = frequencies
        self.sliceWidth = sliceWidth
        self.filters = filters
        self.latentDim = latentDim
        self.initializer = K.initializers.he_uniform()

    def build(self):
        inputs = K.layers.Input(
            shape=[self.frequencies, self.sliceWidth // 3, 1], name="input_image"
        )

        nextLayer = K.layers.ZeroPadding2D(padding=((0, 0), (1, 1)))(inputs)
        nextLayer = self.__downsampleLayer(
            nextLayer,
            kernel=(self.frequencies, 3),
            strides=(1, 1),
            padding="valid",
            numFilters=self.filters,
        )
        nextLayer = self.__downsampleLayer(
            nextLayer,
            kernel=(1, 9),
            strides=(1, 2),
            padding="same",
            numFilters=self.filters,
        )
        nextLayer = self.__downsampleLayer(
            nextLayer,
            kernel=(1, 7),
            strides=(1, 2),
            padding="same",
            numFilters=self.filters,
        )

        nextLayer = K.layers.Flatten()(nextLayer)
        dense = K.layers.Dense(self.latentDim)(nextLayer)

        return K.Model(inputs=inputs, outputs=dense)

    def __downsampleLayer(
        self, input, kernel, strides, padding, numFilters: int, dropout: bool = False
    ):
        next = K.layers.Conv2D(
            numFilters,
            kernel,
            strides=strides,
            padding=padding,
            use_bias=False,
            kernel_initializer=self.initializer,
        )(input)
        next = K.layers.BatchNormalization()(next)
        next = K.layers.LeakyReLU(alpha=0.2)(next)
        if dropout:
            next = K.layers.Dropout(0.2)(next)
        return next