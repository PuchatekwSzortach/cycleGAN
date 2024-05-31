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

    import tensorflow as tf

    import net.ml
    import net.utilities

    tf.config.run_functions_eagerly(True)

    config = net.utilities.read_yaml(config_path)

    data_loader = net.data.ImagePairsDataLoader(
        first_collection_directory=config.horse2zebra_dataset.training_data.first_collection_directory,
        second_collection_directory=config.horse2zebra_dataset.training_data.second_collection_directory,
        batch_size=config.horse2zebra_model.batch_size,
        shuffle=True,
        target_size=config.horse2zebra_model.image_shape,
        augmentation_parameters=config.horse2zebra_model.data_augmentation_parameters
    )

    training_dataset = tf.data.Dataset.from_generator(
        generator=lambda: iter(data_loader),
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
        steps_per_epoch=len(data_loader),
        epochs=2)
