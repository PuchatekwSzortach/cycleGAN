"""
Dummy test file
"""

import numpy as np

import net.ml


def test_cyclegan_generator():
    """
    Test cyclegan generator create data of the same shape as input data
    """

    input_data = np.ones((4, 256, 256, 3))

    actual = net.ml.GeneratorBuilder().get_model().predict(input_data, verbose=0)

    assert actual.shape == input_data.shape


def test_cyclegan_discriminator():
    """
    Test cyclegan discriminator create data of the same shape as input data
    """

    input_data = np.ones((4, 256, 256, 3))

    actual = net.ml.DiscriminatorBuilder().get_model().predict(input_data, verbose=0)

    # Assert batch dimension remains the same
    assert actual.shape[0] == input_data.shape[0]

    # Assert output has 1 channel
    assert actual.shape[-1] == 1


def test_training_cyclegan_model():
    """
    Test training cyclegan model
    """

    import tensorflow as tf

    image_shape = (256, 256, 3)

    # Create a dummy dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            np.ones((4, *image_shape), dtype=np.float32),
            np.ones((4, *image_shape), dtype=np.float32)
        )
    ).batch(2)

    model = net.ml.CycleGANModel()
    model.compile()

    # We're not asserting anything here, just checking if the model can be trained without any errors
    model.fit(dataset, epochs=1, verbose=0)
