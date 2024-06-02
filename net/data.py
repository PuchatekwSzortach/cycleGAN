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

        # Compute samples count so that we don't go over the shorter collection length and
        # have full batches
        self.data_map.samples_count = (
            min(
                len(self.data_map.first_collection_paths),
                len(self.data_map.second_collection_paths)
            ) // self.data_map.batch_size
        ) * self.data_map.batch_size

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

        return self.data_map.samples_count // self.data_map.batch_size

    def __iter__(self):

        while True:

            first_sample_indices = list(range(len(self.data_map.first_collection_paths)))
            second_sample_indices = list(range(len(self.data_map.second_collection_paths)))

            # Shuffle indices for both collections independently,
            # since we don't intend to impose 1:1 mapping between specific samples
            if self.shuffle:
                random.shuffle(first_sample_indices)
                random.shuffle(second_sample_indices)

            for first_indices_batch, second_indices_batch in zip(
                    more_itertools.chunked(
                        first_sample_indices[:self.data_map.samples_count], self.data_map.batch_size, strict=True),
                    more_itertools.chunked(
                        second_sample_indices[:self.data_map.samples_count], self.data_map.batch_size, strict=True)):

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
