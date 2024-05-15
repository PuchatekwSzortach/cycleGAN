"""
Module with visualization commands
"""

import invoke


@invoke.task
def horse2zebra_data(_context, config_path):
    """
    Visualize horse2zebra dataset

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import tqdm
    import vlogging

    import net.data
    import net.processing
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    data_loader = net.data.ImagePairsDataLoader(
        first_collection_directory=config.horse2zebra_dataset.training_data.first_collection_directory,
        second_collection_directory=config.horse2zebra_dataset.training_data.second_collection_directory,
        batch_size=config.horse2zebra_model.batch_size,
        shuffle=True,
        target_size=config.horse2zebra_model.image_shape,
        augmentation_parameters=config.horse2zebra_model.data_augmentation_parameters
    )

    logger = net.utilities.get_logger(config.logging_path)

    iterator = iter(data_loader)

    for _ in tqdm.tqdm(range(4)):

        first_images, second_images = next(iterator)

        for first_image, second_image in zip(first_images, second_images):

            logger.info(
                vlogging.VisualRecord(
                    "horse2zebra",
                    list(net.processing.ImageProcessor.denormalize_batch([first_image, second_image]))
                )

            )
