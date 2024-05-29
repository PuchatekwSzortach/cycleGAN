"""
Module with machine learning related logic
"""

import tensorflow as tf

import net.data


class ReflectionPadding2D(tf.keras.layers.Layer):
    """
    Reflection padding layer
    """

    def __init__(self, margin: int):
        """
        Constructor

        Args:
            margin (int): padding margin
        """

        super().__init__()
        self.margin = margin

    def call(self, x):
        """
        Transformation logic

        Args:
            x: keras symbolic tensor

        Returns:
            transformed keras symbolic tensor
        """

        return tf.pad(
            tensor=x,
            paddings=tf.constant([
                [0, 0], [self.margin, self.margin], [self.margin, self.margin], [0, 0]
            ]),
            mode="REFLECT")


class GeneratorBuilder():
    """
    Generator model
    """

    def _get_resnet_block(self, input_tensor, filters_count):
        """
        ResNet block
        """

        x = ReflectionPadding2D(margin=1)(input_tensor)

        x = tf.keras.layers.Conv2D(
            filters=filters_count,
            kernel_size=3,
            strides=1,
            padding="same"
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = ReflectionPadding2D(margin=1)(input_tensor)

        x = tf.keras.layers.Conv2D(
            filters=filters_count,
            kernel_size=3,
            strides=1,
            padding="valid"
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Add()([x, input_tensor])

        return x

    def get_model(self):
        """
        Constructor
        """

        input_op = tf.keras.layers.Input(
            shape=(None, None, 3)
        )

        base_filters_count = 64

        x = ReflectionPadding2D(margin=3)(input_op)

        x = tf.keras.layers.Conv2D(
            filters=base_filters_count,
            kernel_size=7,
            strides=1,
            padding="same"
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        downsampling_count = 2

        for index in range(downsampling_count):

            multiplier = 2 ** index

            x = tf.keras.layers.Conv2D(
                filters=base_filters_count * multiplier * 2,
                kernel_size=3,
                strides=2,
                padding="valid"
            )(x)

            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        multiplier = 2 ** downsampling_count

        resnet_blocks_count = 9

        for index in range(resnet_blocks_count):

            x = self._get_resnet_block(
                input_tensor=x,
                filters_count=base_filters_count * multiplier)

        for index in range(downsampling_count):

            multiplier = 2 ** (downsampling_count - index)

            x = tf.keras.layers.Conv2DTranspose(
                filters=base_filters_count * multiplier // 2,
                kernel_size=3,
                strides=2,
                padding="same"
            )(x)

            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

            x = tf.keras.layers.Conv2D(
                filters=3,
                kernel_size=7,
                strides=1,
                padding="same",
                activation="tanh"
            )(x)

        return tf.keras.Model(inputs=input_op, outputs=x)


class DiscriminatorBuilder():
    """
    Class for building CycleGAN discriminators
    """

    def get_model(self):
        """
        Get discriminator model
        """

        input_op = tf.keras.layers.Input(
            shape=(None, None, 3)
        )

        base_filters_count = 64

        x = tf.keras.layers.Conv2D(
            filters=base_filters_count,
            kernel_size=4,
            strides=2,
            padding="same"
        )(input_op)

        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        filters_multipliers = 1

        layers_count = 3

        for index in range(1, layers_count):

            filters_multipliers = min(2 ** index, 8)

            x = tf.keras.layers.Conv2D(
                filters=base_filters_count * filters_multipliers,
                kernel_size=4,
                strides=2,
                padding="same"
            )(x)

            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        filters_multipliers = min(2 ** layers_count, 8)

        x = tf.keras.layers.Conv2D(
            filters=base_filters_count * filters_multipliers,
            kernel_size=4,
            strides=1,
            padding="same"
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        # Use 1x1 kernel to get 1 channel output
        x = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=4,
            strides=1,
            padding="same"
        )(x)

        return tf.keras.Model(inputs=input_op, outputs=x)


class CycleGANModel(tf.keras.Model):
    """
    CycleGAN model
    """

    def __init__(self):
        """
        Constructor
        """

        super().__init__()

        self.collection_a_generator = GeneratorBuilder().get_model()
        self.collection_b_generator = GeneratorBuilder().get_model()

        self.collection_a_discriminator = DiscriminatorBuilder().get_model()
        self.collection_b_discriminator = DiscriminatorBuilder().get_model()

        self.collection_a_image_pool = net.data.ImagePool(max_size=50)
        self.collection_b_image_pool = net.data.ImagePool(max_size=50)

    def _get_discriminator_loss_op(self):

        base_loss_op = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        def loss_op(source_images_op, target_images_op):

            return 1

        return loss_op

    def train_step(self, data):
        """
        Manual train step
        """

        collection_a_real_images, collection_b_real_images = data

        # Train discriminators
        self.collection_a_discriminator.trainable = True
        self.collection_b_discriminator.trainable = True
        self.collection_a_generator.trainable = False
        self.collection_b_generator.trainable = False
