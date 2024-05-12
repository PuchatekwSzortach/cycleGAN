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
            use_augmentations: bool, augmentation_parameters: typing.Union[dict, None]):
        """
        Constructor

        Args:
            first_collection_directory (str): path to directory with first collection of images
            second_collection_directory (str): path to directory with second collection of images
            batch_size (int): batch size
            shuffle (bool): if True, images are shuffled randomly
            target_size: typing.Tuple[int, int]: target height and width for images
            use_augmentations (bool): if True, images augmentation is used when drawing samples
            augmentation_parameters (typing.Union[dict, None]): augmentation parameters or None if use_augmentations
            is False
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

        self.use_augmentations = use_augmentations

        self.augmentation_pipeline = self._get_augmentation_pipeline(augmentation_parameters) \
            if use_augmentations is True else None

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
        Get number of images in dataset

        Returns:
            int: number of images in dataset
        """

        return self.data_map.shortest_collection_length // self.data_map.batch_size

    def __iter__(self):

        while True:

            sample_indices = list(range(self.data_map.shortest_collection_length))

            if self.shuffle:
                random.shuffle(sample_indices)

            for indices_batch in more_itertools.chunked(sample_indices, self.data_map.batch_size, strict=True):

                first_collection_batch = []
                second_collection_batch = []

                for sample_index in indices_batch:

                    first_collection_batch.append(
                        cv2.resize(
                            cv2.imread(self.data_map.first_collection_paths[sample_index]),
                            self.data_map.target_size[:2],
                            interpolation=cv2.INTER_CUBIC
                        )
                    )

                    second_collection_batch.append(
                        cv2.resize(
                            cv2.imread(self.data_map.second_collection_paths[sample_index]),
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
