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

    import icecream
    import tqdm

    import net.ml
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

    data_iterator = iter(data_loader)

    cycle_gan = net.ml.CycleGANModel()

    for _ in tqdm.tqdm(range(4)):

        sample = next(data_iterator)
        losses_map = cycle_gan.train_step(sample)

        icecream.ic(losses_map)
