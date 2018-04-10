"""
augment.py

Augmentation utility functions and classes.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Note:
    Make sure to install imgaug with `pip install git+https://github.com/aleju/imgaug`
    Don't use piecewise affine transformations, they're too slow (see `profile_dataset_aug_3chan()`)

Refs:
    https://github.com/aleju/imgaug/issues/41
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from imgaug import augmenters as iaa

_DBG_AUG_SET = -1

_DEFAULT_AUG_OPTIONS = {
    'aug_labels': True,
    'input_channels': 3,  # [3 | 4]
    'fliplr' : 0.5, # Horizontally flip 50% of images
    'flipud' : 0.5, # Vertically flip 50% of images
    'rotate' : (-90, 90),
    'translate' : 0.2, # Translate by -20 to +20% on x- and y-axis independently
    'piecewise_affine' : (0.01, 0.05), # See imgaug doc for details
    'scale' : (0.75, 1.25), # Scale to factor between 75 and 125% of original size
    'random_seed' : 1969}

class Augmenter(object):
    """Augmenter class.
    """

    def __init__(self, options=_DEFAULT_AUG_OPTIONS):
        """Initialize the Dataset object
        Args:
            options: see below
        Options:
            aug_labels: Augment images and labels or just images?
            data_aug: Are we augmenting 3-channel or 4-channel inputs
            fliplr: Horizontal flip probability (set to 0. to disable)
            flipud: Vertical flip probability (set to 0. to disable)
            rotate: Rotation range (set to (0,0) to disable)
            random_seed: Random seed used to init augmenters
        """
        # Save copy of options
        self._options = options

        # Create and init augmenter using provided options
        # Restrict order of operations to the one defined here (random_state=False)
        self._aug = iaa.Sequential(random_state=False)

        order = 3
        backend = 'cv2'
        mode = 'constant'

        if options['fliplr'] > 0.:
            self._aug.append(iaa.Fliplr(options['fliplr']))

        if options['flipud'] > 0.:
            self._aug.append(iaa.Fliplr(options['flipud']))

        # NOTE: The combination below does not yield any speed improvement
        # if options['rotate'] != (0, 0) and options['translate'] > 0. and options['scale'] != (0., 0.):
        #     sc = options['translate']
        #     self._aug.append(iaa.Affine(rotate=options['rotate'],
        #                                 translate_percent={"x": (-sc, sc), "y": (-sc, sc)},
        #                                 scale={"x": options['scale'], "y": options['scale']},
        #                                 mode=mode, order=order, backend=backend))

        if options['rotate'] != (0, 0):
            self._aug.append(iaa.Affine(rotate=options['rotate'], mode=mode, order=order, backend=backend))

        if options['translate'] > 0.:
            sc = options['translate']
            self._aug.append(iaa.Affine(translate_percent={"x": (-sc, sc), "y": (-sc, sc)}, mode=mode, order=order, backend=backend))

        if options['piecewise_affine'] != (0., 0.):
            self._aug.append(iaa.PiecewiseAffine(scale=options['piecewise_affine'], mode=mode, order=order))

        if options['scale'] != (0., 0.):
            self._aug.append(iaa.Affine(scale={"x": options['scale'], "y": options['scale']}, mode=mode, order=order, backend=backend))
            # BUG BUG this fails is scale factor > 1.0: self._aug.append(iaa.Scale(options['scale']))

        self._seed = options['random_seed']

        # Augement labels as well, if requested
        if options['input_channels'] == 4:
            self._aug_extra_channel=self._aug.deepcopy()
        if options['aug_labels']:
            self._aug_labels=self._aug.deepcopy()

    ###
    ### Augmentation
    ###
    def augment(self, images, labels=None):
        """Do all the preprocessing needed before training/val/test samples can be used.
        Args:
            images: list or array of image tensors in HWC format.
            labels: list or array of label tensors in HWC format.
        Returns:
            aug_images: list or array of augmented image tensors in HWC format.
            aug_labels: list or array of augmented label tensors in HWC format, if requested
        """
        # Reseed randomizers
        self._aug.reseed(self._seed)
        if self._options['input_channels'] == 4:
            self._aug_extra_channel.reseed(self._seed)
        if self._options['aug_labels'] and labels is not None:
            self._aug_labels.reseed(self._seed)

        # Augment images
        assert(type(images) is list or type(images) is np.ndarray)
        aug_images = []
        for image in images:
            assert(len(image.shape) == 3 and (image.shape[2]==3 or image.shape[2]==4))
            # image_f = image.astype(np.uint8) # image.astype(float) / 255.
            if image.shape[2]==3:
                # Augment RGB images
                aug_image = self._aug.augment_image(image)
            else:
                # Augment RGB+extra channel images
                RGB_image = self._aug.augment_image(image[:,:,:3])
                # Augmented extra channel may contain interpolated pixel values after augmentation
                extra_channel_max = np.max(image[:,:,3])
                extra_channel = self._aug_extra_channel.augment_image(image[:,:,3])
                # extra_channel = np.where(extra_channel > extra_channel_max * 0.5, extra_channel_max, 0)
                extra_channel = np.where(extra_channel > 0, extra_channel_max, 0)
                extra_channel = np.expand_dims(extra_channel, axis=-1)
                assert (len(extra_channel.shape) == 3 and extra_channel.shape[2] == 1)
                assert (extra_channel.shape[0] == RGB_image.shape[0] and extra_channel.shape[1] == RGB_image.shape[1])
                aug_image = np.concatenate([RGB_image, extra_channel], axis=-1)
            # print("Image [{},{}], Aug Image [{},{}]".format(image.shape, image.dtype, aug_image.shape, aug_image.dtype))
            aug_images.append(aug_image)
        if type(images) is np.ndarray:
            aug_images = np.asarray(aug_images)
                
        # Augment labels, if requested
        if self._options['aug_labels'] and labels is not None:
            aug_labels = []
            for label in labels:
                assert(len(label.shape) == 3 or len(label.shape) == 3 and label.shape[2]==1)
                # Augmented label may contain interpolated pixel values after augmentation
                label_max = np.max(label)
                aug_label = self._aug_labels.augment_image(label)
                aug_label = np.where(aug_label > label_max * 0.5, label_max, 0)
                # print("Label [{},{}], Aug Label [{},{}]".format(label.shape, label.dtype, aug_label.shape, aug_label.dtype))
                aug_labels.append(aug_label) # (aug_label * 255).astype(int))
            if type(labels) is np.ndarray:
                aug_labels = np.asarray(aug_labels)

        self._seed += 1

        if self._options['aug_labels'] and labels is not None:
            return aug_images, aug_labels
        else:
            return aug_images

    ###
    ### Debug utils
    ###
    def print_config(self):
        """Display configuration values."""
        print("\nConfiguration:")
        for k, v in self.options.items():
            print("  {:20} {}".format(k, v))

# def test_basic_behavior():
#
#     import visualize
#     from dataset import _DEFAULT_DSB18_OPTIONS, DSB18Dataset
#
#     # Parameters
#     num_samples = 16
#     num_augs = 16
#     save_folder = "c:\\temp\\visualizations"
#     save_folder2 = "c:\\temp\\visualizations2"
#     save_folder3 = "c:\\temp\\visualizations3"
#
#     # # Create augmenter
#     # options = _DEFAULT_AUG_OPTIONS
#     # options['input_channels'] = 3
#     # aug = Augmenter(options=options)
#
#     # # Load dataset (using semantic labels)
#     # options = _DEFAULT_DSB18_OPTIONS
#     # options['mode'] = 'semantic_labels'
#     # ds = DSB18Dataset(phase='train_val', options=options)
#     #
#     # # Display dataset configuration
#     # ds.print_config()
#     # assert(ds.train_size == 536)
#     # assert(ds.val_size == 134)
#     #
#     #
#     # # Inspect original dataset (with semantic labels)
#     # images, labels = ds.get_rand_samples_with_sem_labels(num_samples, 'train')
#     # assert(type(images) is list and type(labels) is list)
#     # assert(len(images)==num_samples and len(labels)==num_samples)
#     # aug_images = []
#     # aug_labels = []
#     # for image, label in zip(images, labels):
#     #     for n in range(num_augs):
#     #         aug_image, aug_label = aug.augment([image], [label])
#     #         aug_images.append(aug_image[0])
#     #         aug_labels.append(aug_label[0])
#     # visualize.archive_images_with_labels(aug_images, aug_labels, save_folder)
#
#     # Create augmenter
#     options = _DEFAULT_AUG_OPTIONS
#     options['input_channels'] = 4
#     aug = Augmenter(options=options)
#
#     # Load dataset (using contours)
#     options = _DEFAULT_DSB18_OPTIONS
#     options['mode'] = 'instance_contours'
#     options['input_channels'] = 4
#     ds = DSB18Dataset(phase='train_val', options=options)
#
#     # Display dataset configuration
#     ds.print_config()
#     assert(ds.train_size == 536)
#     assert(ds.val_size == 134)
#
#     # Inspect original dataset (with semantic labels)
#     images, labels = ds.get_rand_samples_with_sem_labels(num_samples, 'train')
#     assert(type(images) is list and type(labels) is list)
#     assert(len(images)==num_samples and len(labels)==num_samples)
#     aug_images = []
#     aug_labels = []
#     for image, label in zip(images, labels):
#         for n in range(num_augs):
#             aug_image, aug_label = aug.augment([image], [label])
#             aug_images.append(aug_image[0])
#             aug_labels.append(aug_label[0])
#     visualize.archive_images_with_labels(aug_images, aug_labels, save_folder2)
#     aug_labels = [np.squeeze(aug_label) for aug_label in aug_labels]
#     visualize.archive_images(aug_labels, save_folder3)
#
# def profile_dataset_aug_3chan():
#     """Compare loading+augmenting samples+labele from disk against in memory
#         Profiling results on MSI (with piecewise affine augmentations):
#
#         options['mode'] = 'semantic_labels',  options['input_channels'] = 3,
#         options['in_memory'] = True,  options['data_aug'] = 'heavy'
#         Fetching+Augmenting 100 samples+labels took  0h 1m26s
#
#         options['mode'] = 'semantic_labels',  options['input_channels'] = 3,
#         options['in_memory'] = False,  options['data_aug'] = 'heavy'
#         Fetching+Augmenting 100 samples+labels took  0h 1m27s
#
#         Profiling results on MSI (without piecewise affine augmentations):
#
#         options['mode'] = 'semantic_labels',  options['input_channels'] = 3,
#         options['in_memory'] = True,  options['data_aug'] = 'basic'
#         Fetching+Augmenting 100 samples+labels took  0h 0m 3s
#
#         options['mode'] = 'semantic_labels',  options['input_channels'] = 3,
#         options['in_memory'] = False,  options['data_aug'] = 'basic'
#         Fetching+Augmenting 100 samples+labels took  0h 0m 3s
#     """
#
#     from dataset import _DEFAULT_DSB18_OPTIONS, DSB18Dataset
#     from model import preprocess_inputs, preprocess_labels
#     import time
#
#     # Parameters
#     _NUM_SAMPLES=100
#     _BATCH_SIZE=1
#
#     # Load the dataset object for training (from file)
#     options = _DEFAULT_DSB18_OPTIONS
#     options['mode'] = 'semantic_labels'
#     options['in_memory'] = True
#     options['data_aug'] = 'heavy'
#     options['input_channels'] = 3
#     ds = DSB18Dataset(options=options)
#
#     # Get samples
#     start = time.time()
#     for n in range(_NUM_SAMPLES):
#         samples, gt_labels = ds.next_batch_sem_labels(_BATCH_SIZE, 'train')
#         preprocess_inputs(samples)
#         preprocess_labels(gt_labels)
#     end = time.time()
#     m, s = divmod(end - start, 60)
#     h, m = divmod(m, 60)
#
#     # Display dataset configuration
#     ds.print_config()
#     print('Fetching+Augmenting {} samples+labels took {:2.0f}h{:2.0f}m{:2.0f}s'.format(_NUM_SAMPLES, h, m, s))
#
#     # Load the dataset object for training (from file)
#     options = _DEFAULT_DSB18_OPTIONS
#     options['mode'] = 'semantic_labels'
#     options['in_memory'] = False
#     options['data_aug'] = 'heavy'
#     options['input_channels'] = 3
#     ds = DSB18Dataset(options=options)
#
#     # Get samples
#     start = time.time()
#     for n in range(_NUM_SAMPLES):
#         samples, gt_labels = ds.next_batch_sem_labels(_BATCH_SIZE, 'train')
#         preprocess_inputs(samples)
#         preprocess_labels(gt_labels)
#     end = time.time()
#     m, s = divmod(end - start, 60)
#     h, m = divmod(m, 60)
#
#     # Display dataset configuration
#     ds.print_config()
#     print('Fetching+Augmenting {} samples+labels took {:2.0f}h{:2.0f}m{:2.0f}s'.format(_NUM_SAMPLES, h, m, s))
#
#
#     # Load the dataset object for training (from file)
#     options = _DEFAULT_DSB18_OPTIONS
#     options['mode'] = 'semantic_labels'
#     options['in_memory'] = True
#     options['data_aug'] = 'basic'
#     options['input_channels'] = 3
#     ds = DSB18Dataset(options=options)
#
#     # Get samples
#     start = time.time()
#     for n in range(_NUM_SAMPLES):
#         samples, gt_labels = ds.next_batch_sem_labels(_BATCH_SIZE, 'train')
#         preprocess_inputs(samples)
#         preprocess_labels(gt_labels)
#     end = time.time()
#     m, s = divmod(end - start, 60)
#     h, m = divmod(m, 60)
#
#     # Display dataset configuration
#     ds.print_config()
#     print('Fetching+Augmenting {} samples+labels took {:2.0f}h{:2.0f}m{:2.0f}s'.format(_NUM_SAMPLES, h, m, s))
#
#     # Load the dataset object for training (from file)
#     options = _DEFAULT_DSB18_OPTIONS
#     options['mode'] = 'semantic_labels'
#     options['in_memory'] = False
#     options['data_aug'] = 'basic'
#     options['input_channels'] = 3
#     ds = DSB18Dataset(options=options)
#
#     # Get samples
#     start = time.time()
#     for n in range(_NUM_SAMPLES):
#         samples, gt_labels = ds.next_batch_sem_labels(_BATCH_SIZE, 'train')
#         preprocess_inputs(samples)
#         preprocess_labels(gt_labels)
#     end = time.time()
#     m, s = divmod(end - start, 60)
#     h, m = divmod(m, 60)
#
#     # Display dataset configuration
#     ds.print_config()
#     print('Fetching+Augmenting {} samples+labels took {:2.0f}h{:2.0f}m{:2.0f}s'.format(_NUM_SAMPLES, h, m, s))
#
# # if __name__ == '__main__':
# #     # test_basic_behavior()
# #     # profile_dataset_aug_3chan()
