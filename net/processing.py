"""
Module with data processing utilities
"""

import functools
import io
import random
import tarfile

import cv2
import numpy as np
import tensorflow as tf


class ImageProcessor:
    """
    Image processor with common preprocessing and postprocessing logic
    """

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize image.
        Input image is assumed to be in <0, 255> range.
        Output image will be normalized to <-1, 1> range and use float32 dtype

        Args:
            image (np.ndarray): image to be normalized

        Returns:
            np.ndarray: normalized image
        """

        image = image.astype(np.float32)
        image = image - 127.5
        image = image / 127.5
        return image

    @staticmethod
    def denormalize_image(image: np.ndarray) -> np.ndarray:
        """
        Denormalize image.
        Input image is assumed to be in <-1, 1> range.
        Output image will be normalized and clipped to <0, 255> range and use uint8 dtype

        Args:
            image (np.ndarray): image to be normalized

        Returns:
            np.ndarray: denormalized image
        """

        image = image + 1
        image = image * 127.5
        return np.clip(image.astype(np.uint8), 0, 255)

    @staticmethod
    def normalize_batch(batch: np.ndarray) -> np.ndarray:
        """
        Normalize batch of images.
        Input images are assumed to be in <0, 255> range.
        Output images will be normalized to <-1, 1> range and use float32 dtype

        Args:
            batch (np.ndarray): batch of images to be normalized

        Returns:
            np.ndarray: batch of normalized images
        """

        return np.array([ImageProcessor.normalize_image(image) for image in batch])

    @staticmethod
    def denormalize_batch(batch: np.ndarray) -> np.ndarray:
        """
        Denormalize batch of images.
        Input images are assumed to be in <-1, 1> range.
        Output images will be normalized to <0, 255> range and use uint8 dtype

        Args:
            batch (np.ndarray): batch of images to be normalized

        Returns:
            np.ndarray: denormalized images
        """

        return np.array([ImageProcessor.denormalize_image(image) for image in batch])


def get_image_tar_map(image: np.ndarray, name: str) -> dict:
    """
    Get image tar map for given image and name

    Args:
        image (np.ndarray): image to compute tar file for
        name (str): name to be used in tar file

    Returns:
        dict: map with keys "tar_info" and "bytes"
    """

    _, jpg_bytes = cv2.imencode(".jpg", image)

    # Create tar info for image
    tar_info = tarfile.TarInfo(name=name)
    tar_info.size = len(jpg_bytes)

    return {
        "tar_info": tar_info,
        "bytes": io.BytesIO(jpg_bytes)
    }


class ImagePool:
    """
    Image pool class that stores up to max_size images and on query(images)
    method returns len(images) images that have 50% chance of coming from images and 50% chance
    of coming from pool. For any image returned from the pool, it's replaced with new image.
    """

    def __init__(self, max_size: int):
        """
        Constructor

        Args:
            max_size (int): maximum size of the pool
        """

        self.max_size = max_size
        self.images = []

    def query_single_image(self, input_image, replace_probability: float):
        """
        Get an image with replace_probability chance of image coming from image pool and
        otherwise returning input image.
        If image from the pool is returned, then input image is inserted into the pool.

        Args:
            input_image (tf.Tensor): input image
            replace_probability (float): probability of replacing image from pool

        Returns:
            tf.Tensor: output image
        """

        if len(self.images) < self.max_size:

            self.images.append(input_image)
            return input_image

        else:

            # If the image pool is full, randomly choose whether to replace an image from the pool or not
            if random.random() < replace_probability:

                # If chosen to replace,
                # randomly select an image from the pool and append it to the output images
                random_index = random.randint(0, len(self.images) - 1)
                output_image = self.images[random_index]

                # Replace the selected image in the pool with the new image
                # self.images[random_index] = input_image
                return output_image
            else:
                # If chosen not to replace, return input_image
                return input_image

    def query(self, input_images, replace_probability: float):
        """
        Get len(input_images) images that have replace_probability chance of coming from input_images and
        50% chance of coming from pool.
        If image comes from pool, it's replaced with new image from input_images.

        Args:
            input_images: tensorflow tensor with input images
            replace_probability (float): probability of replacing image from pool

        Returns:
            tf.Tensor: tensor with output images
        """

        # Make partial function to be able to pass replace_probability
        query_single_image_partial = functools.partial(
            self.query_single_image, replace_probability=replace_probability)

        # Jumping through hoops to be able to loop over both eager and symbolic tensors
        return tf.map_fn(query_single_image_partial, input_images)
