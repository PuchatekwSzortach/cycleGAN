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

    import net.data
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    data_loader = net.data.ImagePairsDataLoader(
        first_collection_directory=config.horse2zebra_dataset.training_data.first_collection_directory,
        second_collection_directory=config.horse2zebra_dataset.training_data.second_collection_directory,
        batch_size=config.horse2zebra_model.batch_size,
        shuffle=True,
        target_size=config.horse2zebra_model.image_shape,
        use_augmentations=True,
        augmentation_parameters=config.horse2zebra_model.data_augmentation_parameters
    )

    iterator = iter(data_loader)

    for _ in tqdm.tqdm(range(4)):

        first_images, second_images = next(iterator)

        print(first_images.shape, second_images.shape)
