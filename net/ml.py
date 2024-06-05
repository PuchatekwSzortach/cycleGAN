"""
Module with machine learning related logic
"""

import collections
import glob
import os
import shutil
import tarfile
import tempfile

import numpy as np
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


class ModelCheckpoint(tf.keras.callbacks.Callback):
    """
    Callback for periodically saving model provided in constructor - so a bit different from keras' ModelCheckpoint
    """

    def __init__(self, target_model: tf.keras.Model, checkpoint_path: str, saving_interval_in_steps: int):
        """
        Constructor

        Args:
            model (tf.keras.Model): model to save
            checkpoint_path (str): path to save model at
            saving_interval_in_steps (int): specifies how often model should be saved
        """

        super().__init__()

        self.target_model = target_model
        self.checkpoint_path = checkpoint_path
        self.saving_interval_in_steps = saving_interval_in_steps

        self.steps_counter = 0

    def on_train_batch_end(self, batch, logs=None):
        """
        On train batch end callback, saves model if specified number of steps has passed since last save
        """

        if self.steps_counter == self.saving_interval_in_steps:

            shutil.rmtree(self.checkpoint_path, ignore_errors=True)
            self.target_model.save(self.checkpoint_path, save_format="h5")

            self.steps_counter = 0

        else:

            self.steps_counter += 1


class GANLearningRateSchedulerCallback(tf.keras.callbacks.Callback):
    """
    Learning rate scheduler callback for a GAN model
    """

    def __init__(
            self, generator_optimizer, discriminator_opitimizer, base_learning_rate: float,
            epochs_count: int):

        self.generator_optimizer = generator_optimizer
        self.discriminator_opitimizer = discriminator_opitimizer
        self.base_learning_rate = base_learning_rate
        self.epochs_count = epochs_count

    def on_epoch_end(self, epoch: int, logs=None):
        """
        Function to be called at the end of each epoch
        """

        half_epochs_count = float(self.epochs_count / 2.0)

        learning_rate = self.base_learning_rate * \
            (1.0 - max(0, epoch + 1.0 - half_epochs_count) / (half_epochs_count + 1.0))

        self.generator_optimizer.learning_rate = learning_rate
        self.discriminator_opitimizer.learning_rate = learning_rate


class VisualizationArchivesBuilderCallback(tf.keras.callbacks.Callback):
    """
    Keras callback that every few batches logs CycleGAN inputs and outputs into a tar archive.
    Once archive exceeds specified size, it's closed and rotated.
    Only max_archives_count archives are kept, and archives older than that are deleted.
    """

    def __init__(
            self, cycle_gan: CycleGANModel, data_iterator,
            output_directory: str, file_name: str, logging_interval: int,
            max_archive_size_in_bytes: int, max_archives_count: int):
        """
        Constructor.
        Note - constructor will delete any old archives matching pattern "output_directory/filename*.tar"

        Args:
            cycleGAN (CycleGANModel): cycleGAN instance
            data_iterator: iterator that yields sources and targets images batches
            output_directory (str): directory where archives will be saved
            file_name (str): name of archive file, without extension. Extension will be .tar, and rotated archives
            will have numbers appended to them (e.g. archive.1.tar, archive.2.tar, etc.)
            logging_interval (int): number of batches between logging
            max_archive_size_in_bytes (int): maximum size of archive in bytes,
            once archive reaches this size, it's closed and rotated
            max_archives_count (int): maximum number of archives to keep,
            once this number is reached, oldest archive is deleted
        """

        super().__init__()

        self.cycle_gan = cycle_gan
        self.output_directory = output_directory
        self.file_name = file_name
        self.data_iterator = data_iterator

        self.numeric_constraints_map = {
            "logging_interval": logging_interval,
            "max_archive_size_in_bytes": max_archive_size_in_bytes,
            "max_archives_count": max_archives_count
        }

        self.counters_map = {
            "epoch": 0,
            "batch": 0
        }

        self.tar_data = {
            "base_archive_path": os.path.join(self.output_directory, f"{self.file_name}.tar"),
            "tar_files_maps": []
        }

        # Delete any old archives
        for file_path in glob.glob(f"{self.output_directory}/{self.file_name}*.tar"):
            os.remove(file_path)

    def _rotate_archives(self):
        """
        Rotate archives
        """

        # Get list of archive files
        sorted_archives_files_paths = sorted(glob.glob(os.path.join(self.output_directory, f"{self.file_name}.*.tar")))

        # We only want to keep max_archives_count, so that means that won't be
        # backing up oldest one - it will instead be overwritten by next oldest archive
        archive_files_paths_to_keep = sorted_archives_files_paths[
            :self.numeric_constraints_map["max_archives_count"] - 1]

        # Go over archives to keep after rotation from oldest to youngest
        for rotated_archive_path in reversed(archive_files_paths_to_keep):

            # Get index of archive
            index = int(os.path.basename(rotated_archive_path).split(".")[-2])

            new_archive_path = os.path.join(self.output_directory, f"{self.file_name}.{index + 1}.tar")

            # Move archive to next index
            os.rename(rotated_archive_path, new_archive_path)

        # Move latest archive to index 1
        os.rename(self.tar_data["base_archive_path"], os.path.join(self.output_directory, f"{self.file_name}.1.tar"))

    def on_epoch_end(self, epoch: int, logs=None):
        """
        On epoch end callback
        """

        self.counters_map["epoch"] += 1

    def on_train_batch_end(self, batch, logs=None):
        """
        Visualize generator output once every x batches
        """

        if self.counters_map["batch"] == self.numeric_constraints_map["logging_interval"]:

            should_rotate_archives = \
                os.path.exists(self.tar_data["base_archive_path"]) and \
                os.path.getsize(
                    self.tar_data["base_archive_path"]) > self.numeric_constraints_map["max_archive_size_in_bytes"]

            # If archive is too big, rotate archive files and clear tar files maps
            if should_rotate_archives:

                self._rotate_archives()
                self.tar_data["tar_files_maps"].clear()

            collection_a_images, collection_b_images = next(self.data_iterator)

            self._log_cycle_gan_results(
                source_images=collection_a_images,
                generator=self.cycle_gan.models_map["collection_a_generator"],
                reverse_generator=self.cycle_gan.models_map["collection_b_generator"],
                batch_index=batch,
                collection_name="A_to_B"
            )

            self._log_cycle_gan_results(
                source_images=collection_b_images,
                generator=self.cycle_gan.models_map["collection_b_generator"],
                reverse_generator=self.cycle_gan.models_map["collection_a_generator"],
                batch_index=batch,
                collection_name="B_to_A"
            )

            # Create temporary directory
            with tempfile.TemporaryDirectory() as tmp_dir:

                temporary_archive_path = os.path.join(tmp_dir, "archive.tar")

                # Create tarfile object in temporary directory
                with tarfile.open(name=temporary_archive_path, mode="x") as tar:

                    # Add all tar files maps to tar file
                    for tar_file_map in self.tar_data["tar_files_maps"]:

                        tar_file_map["bytes"].seek(0)
                        tar.addfile(tarinfo=tar_file_map["tar_info"], fileobj=tar_file_map["bytes"])

                # Move temporary file to target path
                shutil.move(temporary_archive_path, self.tar_data["base_archive_path"])

            # Reset batch counter
            self.counters_map["batch"] = 0

        else:

            self.counters_map["batch"] += 1

    def _log_cycle_gan_results(
            self, source_images, generator: tf.keras.Model, reverse_generator: tf.keras.Model, batch_index: int,
            collection_name: str):
        """
        Log cycle gan results

        Args:
            source_images: source images
            generator (tf.keras.Model): generator used to transfer source images to target domain
            reverse_generator (_type_): generator used to transfer generated images back to source domain
            batch_index (int): batch index
            collection_name (str): name of collection
        """

        generated_images = generator.predict(source_images, verbose=False)

        reconstructed_images = reverse_generator.predict(generated_images, verbose=False)

        # Compute tar files maps for all triplets
        for triplet_index, triplet in enumerate(zip(source_images, generated_images, reconstructed_images)):

            normalized_triplet = net.processing.ImageProcessor.denormalize_batch(np.array(triplet))

            self.tar_data["tar_files_maps"].extend([
                net.processing.get_image_tar_map(
                    image=image,
                    name=f"epoch_{self.counters_map['epoch']}_batch_{batch_index}_index_{triplet_index}_{name}.jpg"
                ) for image, name in zip(
                    normalized_triplet,
                    [
                        f"{collection_name}_a_source",
                        f"{collection_name}_b_generated_image",
                        f"{collection_name}_c_recovered_image"
                    ]
                )
            ])
