"""
Module with data IO logic
"""

import glob
import os
import random
import typing

import box
import cv2
import imgaug
import more_itertools
import numpy as np

import net.processing


class ImagePairsDataLoader:
    """
    Data loader for datasets that contain two collections of images.
    Yields tuples (images, images) where first element comes from first collection and second element
    comes from second collection.
    Loader length will be determined by shortest collection length.
    """

    def __init__(
            self,
            first_collection_directory: str, second_collection_directory: str,
            batch_size: int,
            shuffle: bool,
            target_size: typing.Tuple[int, int],
            augmentation_parameters: typing.Union[dict, None]):
        """
        Constructor

        Args:
            first_collection_directory (str): path to directory with first collection of images
            second_collection_directory (str): path to directory with second collection of images
            batch_size (int): batch size
            shuffle (bool): if True, images are shuffled randomly. Both collections are shuffled independently,
            so there is no guarantee that images from the same index in both collections will be paired together.
            target_size: typing.Tuple[int, int]: target height and width for images
            augmentation_parameters (typing.Union[dict, None]): augmentation parameters or None if no augmentation
            should be applied
        """

        self.data_map = box.Box({
            "batch_size": batch_size,
            "target_size": target_size,
            "first_collection_paths": sorted(
                glob.glob(pathname=os.path.join(first_collection_directory, "*.jpg"))),
            "second_collection_paths": sorted(
                glob.glob(pathname=os.path.join(second_collection_directory, "*.jpg")))
        })

        self.data_map.shortest_collection_length = min(
            len(self.data_map.first_collection_paths), len(self.data_map.second_collection_paths))

        self.shuffle = shuffle

        self.use_augmentations = augmentation_parameters is not None

        self.augmentation_pipeline = self._get_augmentation_pipeline(augmentation_parameters) \
            if self.use_augmentations is True else None

    def _get_augmentation_pipeline(self, augmentation_parameters: dict) -> imgaug.augmenters.Sequential:
        """
        Get augmentation pipeline

        Args:
            augmentation_parameters (dict): augmentation parameters

        Returns:
            imgaug.augmenters.Sequential: augmentation pipeline
        """

        augmenters = [
            imgaug.augmenters.Fliplr(p=0.5)
        ]

        if augmentation_parameters.get("use_up_down_flip", False) is True:
            augmenters.append(imgaug.augmenters.Flipud(p=0.5))

        augmenters.extend([
            imgaug.augmenters.Resize(augmentation_parameters["resized_image_shape"]),
            imgaug.augmenters.CropToFixedSize(
                width=augmentation_parameters["image_shape"][0],
                height=augmentation_parameters["image_shape"][1]
            )
        ])

        return imgaug.augmenters.Sequential(augmenters)

    def __len__(self) -> int:
        """
        Get number of samples dataloader can yield

        Returns:
            int: number of samples dataloader can yield
        """

        return self.data_map.shortest_collection_length // self.data_map.batch_size

    def __iter__(self):

        while True:

            first_sample_indices = list(range(len(self.data_map.first_collection_paths)))
            second_sample_indices = list(range(len(self.data_map.second_collection_paths)))

            # Shuffle indices for both collections independently,
            # since we don't intend to impose 1:1 mapping between specific samples
            if self.shuffle:
                random.shuffle(first_sample_indices)
                random.shuffle(second_sample_indices)

            # Truncate indices to the shortest collection length
            first_sample_indices = first_sample_indices[:self.data_map.shortest_collection_length]
            second_sample_indices = second_sample_indices[:self.data_map.shortest_collection_length]

            for first_indices_batch, second_indices_batch in zip(
                    more_itertools.chunked(first_sample_indices, self.data_map.batch_size, strict=True),
                    more_itertools.chunked(second_sample_indices, self.data_map.batch_size, strict=True)):

                first_collection_batch = []
                second_collection_batch = []

                for first_sample_index, second_sample_index in zip(first_indices_batch, second_indices_batch):

                    first_collection_batch.append(
                        cv2.resize(
                            cv2.imread(self.data_map.first_collection_paths[first_sample_index]),
                            self.data_map.target_size[:2],
                            interpolation=cv2.INTER_CUBIC
                        )
                    )

                    second_collection_batch.append(
                        cv2.resize(
                            cv2.imread(self.data_map.second_collection_paths[second_sample_index]),
                            self.data_map.target_size[:2],
                            interpolation=cv2.INTER_CUBIC
                        )
                    )

                if self.use_augmentations is True:

                    first_collection_batch, second_collection_batch = self.augmentation_pipeline(
                        images=first_collection_batch,
                        segmentation_maps=second_collection_batch
                    )

                yield \
                    net.processing.ImageProcessor.normalize_batch(first_collection_batch), \
                    net.processing.ImageProcessor.normalize_batch(second_collection_batch)


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

    def query(self, input_images: np.ndarray) -> np.ndarray:
        """
        Get len(input_images) images that have 50% chance of coming from input_images and
        50% chance of coming from pool.
        If image comes from pool, it's replaced with new image from input_images.

        Args:
            input_images (np.ndarray): An array of input images.

        Returns:
            np.ndarray: An array of output images.
        """

        # Create an empty list to store the output images
        output_images = []

        # Iterate over each input image
        for image in input_images:

            # If the image pool is not full,
            # add the image to the pool and append it to the output images
            if len(self.images) < self.max_size:
                self.images.append(image)
                output_images.append(image)
            else:

                # If the image pool is full, randomly choose whether to replace an image from the pool or not
                if random.random() > 0.5:

                    # If chosen to replace,
                    # randomly select an image from the pool and append it to the output images
                    random_index = random.randint(0, len(self.images) - 1)
                    output_images.append(self.images[random_index])

                    # Replace the selected image in the pool with the new image
                    self.images[random_index] = image
                else:
                    # If chosen not to replace, append the new image to the output images
                    output_images.append(image)

        # Convert the output images list to a numpy array and return it
        return np.array(output_images)
