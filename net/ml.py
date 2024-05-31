"""
Module with machine learning related logic
"""

import collections

import tensorflow as tf

import net.data
import net.processing


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

        model = tf.keras.Model(inputs=input_op, outputs=x)
        model.compile()
        return model


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

        model = tf.keras.Model(inputs=input_op, outputs=x)
        model.compile()
        return model


class CycleGANModel(tf.keras.Model):
    """
    CycleGAN model
    """

    def __init__(self):
        """
        Constructor
        """

        super().__init__()

        self.models_map = {
            "collection_a_generator": GeneratorBuilder().get_model(),
            "collection_b_generator": GeneratorBuilder().get_model(),
            "collection_a_discriminator": DiscriminatorBuilder().get_model(),
            "collection_b_discriminator": DiscriminatorBuilder().get_model()
        }

        self.image_pools_map = {
            "collection_a_generated_images_pool": net.processing.ImagePool(max_size=50),
            "collection_b_generated_images_pool": net.processing.ImagePool(max_size=50)
        }

        self.losses_ops_map = {
            "mean_squared_error": tf.keras.losses.MeanSquaredError(),
            "mean_absolute_error": tf.keras.losses.MeanAbsoluteError(),
        }

        self.optimizers_map = {
            "collection_a_discriminator": tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            "collection_b_discriminator": tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            "collection_a_generator": tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            "collection_b_generator": tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        }

    def _train_discriminators(self, data: tuple) -> dict:
        """
        Train discriminators

        Args:
            data (tuple): training data

        Returns:
            dict: map with losses for trained discriminators
        """

        collection_a_real_images, collection_b_real_images = data

        # Train discriminators
        self.models_map["collection_a_discriminator"].trainable = True
        self.models_map["collection_b_discriminator"].trainable = True
        self.models_map["collection_a_generator"].trainable = False
        self.models_map["collection_b_generator"].trainable = False

        losses_map = {}

        # Define a named tuple for the generator and discriminator data
        TrainingInput = collections.namedtuple(
            "TrainingInput",
            [
                "generator", "discriminator", "source_images", "target_images",
                "image_pool", "optimizer", "loss_name"])

        # Training inputs for training both discriminators
        training_inputs = [
            TrainingInput(
                generator=self.models_map["collection_a_generator"],
                discriminator=self.models_map["collection_a_discriminator"],
                source_images=collection_b_real_images,
                target_images=collection_a_real_images,
                image_pool=self.image_pools_map["collection_a_generated_images_pool"],
                optimizer=self.optimizers_map["collection_a_discriminator"],
                loss_name="discriminator_a_loss"
            ),
            TrainingInput(
                generator=self.models_map["collection_b_generator"],
                discriminator=self.models_map["collection_b_discriminator"],
                source_images=collection_a_real_images,
                target_images=collection_b_real_images,
                image_pool=self.image_pools_map["collection_b_generated_images_pool"],
                optimizer=self.optimizers_map["collection_b_discriminator"],
                loss_name="discriminator_b_loss"
            )
        ]

        # Iterate over training inputs to train discriminators
        for training_input in training_inputs:

            with tf.GradientTape() as discriminator_tape:

                generated_images = training_input.generator(training_input.source_images, training=False)

                generated_images_from_pool = training_input.image_pool.query(
                    input_images=generated_images,
                    replace_probability=0.5
                )

                discriminator_prediction_on_real_images = training_input.discriminator(
                    training_input.target_images, training=True)

                discriminator_predictions_on_generated_images = training_input.discriminator(
                    generated_images_from_pool, training=True)

                discriminator_loss = \
                    self.losses_ops_map["mean_squared_error"](
                        y_true=tf.ones_like(discriminator_prediction_on_real_images),
                        y_pred=discriminator_prediction_on_real_images) + \
                    self.losses_ops_map["mean_squared_error"](
                        y_true=tf.zeros_like(discriminator_predictions_on_generated_images),
                        y_pred=discriminator_predictions_on_generated_images)

                losses_map[training_input.loss_name] = discriminator_loss

            discriminator_gradients = discriminator_tape.gradient(
                discriminator_loss,
                training_input.discriminator.trainable_variables)

            training_input.optimizer.apply_gradients(
                zip(discriminator_gradients, training_input.discriminator.trainable_variables))

        return losses_map

    def _train_generators(self, data: tuple) -> dict:

        collection_a_real_images, collection_b_real_images = data

        # Train generators
        self.models_map["collection_a_discriminator"].trainable = False
        self.models_map["collection_b_discriminator"].trainable = False
        self.models_map["collection_a_generator"].trainable = True
        self.models_map["collection_b_generator"].trainable = True

        losses_map = {}
        losses_components = {}

        with tf.GradientTape() as generator_a_tape, tf.GradientTape() as generator_b_tape:

            generated_images_map = {
                "collection_a_generated_images": self.models_map["collection_a_generator"](
                    collection_b_real_images, training=True),
                "collection_b_generated_images": self.models_map["collection_b_generator"](
                    collection_a_real_images, training=True)
            }

            discriminator_a_predictions_on_generated_images = self.models_map["collection_a_discriminator"](
                generated_images_map["collection_a_generated_images"], training=False)

            losses_components["discriminator_a_loss"] = self.losses_ops_map["mean_squared_error"](
                y_true=tf.ones_like(discriminator_a_predictions_on_generated_images),
                y_pred=discriminator_a_predictions_on_generated_images)

            discriminator_b_predictions_on_generated_images = self.models_map["collection_b_discriminator"](
                generated_images_map["collection_b_generated_images"], training=False)

            losses_components["discriminator_b_loss"] = self.losses_ops_map["mean_squared_error"](
                y_true=tf.ones_like(discriminator_b_predictions_on_generated_images),
                y_pred=discriminator_b_predictions_on_generated_images)

            losses_components["generator_a_identity_loss"] = self.losses_ops_map["mean_absolute_error"](
                y_true=collection_a_real_images,
                y_pred=self.models_map["collection_a_generator"](collection_a_real_images, training=True)
            )

            losses_components["generator_b_identity_loss"] = self.losses_ops_map["mean_absolute_error"](
                y_true=collection_b_real_images,
                y_pred=self.models_map["collection_b_generator"](collection_b_real_images, training=True)
            )

            losses_components["collection_a_cyclic_consistency_loss"] = self.losses_ops_map["mean_absolute_error"](
                y_true=collection_a_real_images,
                y_pred=self.models_map["collection_a_generator"](
                    generated_images_map["collection_b_generated_images"], training=True)
            )

            losses_components["collection_b_cyclic_consistency_loss"] = self.losses_ops_map["mean_absolute_error"](
                y_true=collection_b_real_images,
                y_pred=self.models_map["collection_b_generator"](
                    generated_images_map["collection_a_generated_images"], training=True)
            )

            losses_map["generator_a_loss"] = \
                losses_components["discriminator_a_loss"] + \
                (10.0 * losses_components["collection_a_cyclic_consistency_loss"]) + \
                (10.0 * losses_components["collection_b_cyclic_consistency_loss"]) + \
                (5.0 * losses_components["generator_a_identity_loss"])

            losses_map["generator_b_loss"] = \
                losses_components["discriminator_b_loss"] + \
                (10.0 * losses_components["collection_a_cyclic_consistency_loss"]) + \
                (10.0 * losses_components["collection_b_cyclic_consistency_loss"]) + \
                (5.0 * losses_components["generator_b_identity_loss"])

        generator_a_gradients = generator_a_tape.gradient(
            losses_map["generator_a_loss"],
            self.models_map["collection_a_generator"].trainable_variables)

        self.optimizers_map["collection_a_generator"].apply_gradients(
            zip(generator_a_gradients, self.models_map["collection_a_generator"].trainable_variables))

        generator_b_gradients = generator_b_tape.gradient(
            losses_map["generator_b_loss"],
            self.models_map["collection_b_generator"].trainable_variables)

        self.optimizers_map["collection_b_generator"].apply_gradients(
            zip(generator_b_gradients, self.models_map["collection_b_generator"].trainable_variables))

        return losses_map

    def train_step(self, data):
        """
        Manual train step
        """

        discriminator_losses_map = self._train_discriminators(data)
        generator_losses_map = self._train_generators(data)

        losses_map = {
            **discriminator_losses_map,
            **generator_losses_map
        }

        # Disable training for discriminators and generators, so
        # that validation can be performed without affecting gradients
        self.models_map["collection_a_discriminator"].trainable = False
        self.models_map["collection_b_discriminator"].trainable = False
        self.models_map["collection_a_generator"].trainable = False
        self.models_map["collection_b_generator"].trainable = False

        return losses_map
