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
    import numpy as np

    import net.ml
    import net.utilities

    # config = net.utilities.read_yaml(config_path)

    cycle_gan = net.ml.CycleGANModel()

    data = np.ones((4, 256, 256, 3))

    icecream.ic(data.shape)

    output = cycle_gan.collection_a_discriminator.predict(data, verbose=0)

    icecream.ic(output.shape)
