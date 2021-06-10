from models.MelGAN.layers import ConvSN2D, ConvSN2DTranspose
from tensorflow import keras as K
import tensorflow_addons as tfa


class Generator:
    def __init__(
        self, frequencies: int, sliceWidth: int, filters: int, unetBlocks: int
    ):
        self.frequencies = frequencies
        self.sliceWidth = sliceWidth
        self.filters = filters
        self.unetBlocks = unetBlocks
        self.initializer = K.initializers.he_uniform()

    def build(self):
        inputs = K.layers.Input((self.frequencies, self.sliceWidth // 3, 1))

        nextLayer = K.layers.ZeroPadding2D(padding=((0, 0), (1, 1)))(inputs)
        nextLayer = enc1 = self.__encoderBlock(
            nextLayer,
            kernel=(self.frequencies, 3),
            strides=(1, 1),
            padding="valid",
            numFilters=self.filters,
        )
        nextLayer = enc2 = self.__encoderBlock(
            nextLayer,
            kernel=(1, 9),
            strides=(1, 2),
            padding="same",
            numFilters=self.filters,
        )
        nextLayer = self.__encoderBlock(
            nextLayer,
            kernel=(1, 7),
            strides=(1, 2),
            padding="same",
            numFilters=self.filters,
        )

        nextLayer = self.__decoderBlock(
            nextLayer,
            kernel=(1, 7),
            strides=(1, 2),
            padding="same",
            skipInput=enc2,
            numFilters=self.filters,
        )
        nextLayer = self.__decoderBlock(
            nextLayer,
            kernel=(1, 9),
            strides=(1, 2),
            padding="same",
            skipInput=enc1,
            numFilters=self.filters,
            batchNorm=False,
        )

        outputs = ConvSN2DTranspose(
            1,
            (self.frequencies, 1),
            strides=(1, 1),
            padding="valid",
            activation="tanh",
            kernel_initializer=self.initializer,
        )(nextLayer)

        return K.Model(inputs=inputs, outputs=outputs)

    def __decoderBlock(
        self,
        input,
        kernel,
        strides,
        padding,
        skipInput,
        numFilters: int,
        dropout: bool = False,
        batchNorm=True,
    ):
        next = K.layers.UpSampling2D(strides)(input)
        next = ConvSN2D(
            numFilters,
            kernel,
            strides=(1, 1),
            padding=padding,
            use_bias=False,
            kernel_initializer=self.initializer,
        )(next)

        if batchNorm:
            next = K.layers.BatchNormalization()(next)

        next = K.layers.LeakyReLU(alpha=0.2)(next)
        next = K.layers.Concatenate()([next, skipInput])

        if dropout:
            next = K.layers.Dropout(0.2)(next)
        return next

    def __encoderBlock(
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

        next = K.layers.BatchNormalization()(next)
        next = K.layers.LeakyReLU(alpha=0.2)(next)
        if dropout:
            next = K.layers.Dropout(0.2)(next)
        return next
