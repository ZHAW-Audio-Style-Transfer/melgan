from models.MelGAN.layers import ConvSN2D, DenseSN
import tensorflow as tf
from tensorflow import keras as K
import tensorflow_addons as tfa


class Discriminator:
    def __init__(self, frequencies, sliceWidth, filters):
        self.frequencies = frequencies
        self.sliceWidth = sliceWidth
        self.filters = filters
        self.initializer = K.initializers.he_uniform()

    def build(self):
        inputs = K.layers.Input(
            shape=[self.frequencies, self.sliceWidth, 1], name="input_image"
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
        nextLayer = DenseSN(1, kernel_initializer=self.initializer)(nextLayer)

        return K.Model(inputs=inputs, outputs=nextLayer)

    def __downsampleLayer(
        self, input, kernel, strides, padding, numFilters: int, dropout: bool = False
    ):
        next = ConvSN2D(
            numFilters,
            kernel,
            strides=strides,
            padding=padding,
            use_bias=False,
            kernel_initializer=self.initializer,
        )(input)

        next = K.layers.LeakyReLU(alpha=0.2)(next)
        if dropout:
            next = K.layers.Dropout(0.2)(next)
        return next
