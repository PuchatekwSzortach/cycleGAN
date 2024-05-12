"""
Module with data processing utilities
"""

import io
import tarfile

import cv2
import numpy as np


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
