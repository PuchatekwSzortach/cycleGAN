"""
Module with training commands
"""

import invoke


@invoke.task
def train_horse2zebra_model(_context, config_path):
    """
    Trains cycleGAN model on horse2zebra dataset

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import os

    import tensorflow as tf

    import net.data
    import net.ml
    import net.utilities

    tf.config.run_functions_eagerly(True)

    config = net.utilities.read_yaml(config_path)

    training_data_loader = net.data.ImagePairsDataLoader(
        first_collection_directory=config.horse2zebra_dataset.training_data.first_collection_directory,
        second_collection_directory=config.horse2zebra_dataset.training_data.second_collection_directory,
        batch_size=config.horse2zebra_model.batch_size,
        shuffle=True,
        target_size=config.horse2zebra_model.image_shape,
        augmentation_parameters=config.horse2zebra_model.data_augmentation_parameters
    )

    training_dataset = tf.data.Dataset.from_generator(
        generator=lambda: iter(training_data_loader),
        output_types=(
            tf.float32,
            tf.float32
        ),
        output_shapes=(
            tf.TensorShape([None, None, None, 3]),
            tf.TensorShape([None, None, None, 3]),
        )
    ).prefetch(32)

    validation_data_loader = net.data.ImagePairsDataLoader(
        first_collection_directory=config.horse2zebra_dataset.validation_data.first_collection_directory,
        second_collection_directory=config.horse2zebra_dataset.validation_data.second_collection_directory,
        batch_size=config.horse2zebra_model.batch_size,
        shuffle=True,
        target_size=config.horse2zebra_model.image_shape,
        augmentation_parameters=config.horse2zebra_model.data_augmentation_parameters
    )

    validation_dataset = tf.data.Dataset.from_generator(
        generator=lambda: iter(validation_data_loader),
        output_types=(
            tf.float32,
            tf.float32
        ),
        output_shapes=(
            tf.TensorShape([None, None, None, 3]),
            tf.TensorShape([None, None, None, 3]),
        )
    ).prefetch(32)

    cycle_gan = net.ml.CycleGANModel()
    cycle_gan.compile()

    cycle_gan.fit(
        training_dataset,
        steps_per_epoch=len(training_data_loader),
        epochs=config.horse2zebra_model.epochs,
        callbacks=[
            net.ml.ModelCheckpoint(
                target_model=cycle_gan.models_map["collection_a_generator"],
                checkpoint_path=config.horse2zebra_model.collection_a_generator_model_path,
                saving_interval_in_steps=500
            ),
            net.ml.ModelCheckpoint(
                target_model=cycle_gan.models_map["collection_b_generator"],
                checkpoint_path=config.horse2zebra_model.collection_b_generator_model_path,
                saving_interval_in_steps=500
            ),
            net.ml.GANLearningRateSchedulerCallback(
                generator_optimizer=cycle_gan.optimizers_map["collection_a_generator"],
                discriminator_opitimizer=cycle_gan.optimizers_map["collection_a_discriminator"],
                base_learning_rate=config.horse2zebra_model.learning_rate,
                epochs_count=config.horse2zebra_model.epochs
            ),
            net.ml.GANLearningRateSchedulerCallback(
                generator_optimizer=cycle_gan.optimizers_map["collection_b_generator"],
                discriminator_opitimizer=cycle_gan.optimizers_map["collection_b_discriminator"],
                base_learning_rate=config.horse2zebra_model.learning_rate,
                epochs_count=config.horse2zebra_model.epochs
            ),
            net.ml.VisualizationArchivesBuilderCallback(
                cycle_gan=cycle_gan,
                data_iterator=iter(validation_dataset),
                output_directory=os.path.dirname(config.logging_path),
                file_name="archive",
                logging_interval=200,
                max_archive_size_in_bytes=100 * 1024 * 1024,
                max_archives_count=10
            ),
        ]
    )
