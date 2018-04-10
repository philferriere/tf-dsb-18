"""
adapt.py

Helpers to adapt input data to convnet input size requirements.

Modifications by Phil Ferriere licensed under the MIT License (see LICENSE for details)

Based on:
    - https://github.com/matterport/Mask_RCNN/blob/master/utils.py
        Copyright (c) 2017 Matterport, Inc. / Written by Waleed Abdulla
        Licensed under the MIT License

Refs:
    SciPy Multi-dimensional image processing (ndimage) @ https://docs.scipy.org/doc/scipy-0.19.1/reference/ndimage.html
    - ndi.zoom
        https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.ndimage.zoom.html#scipy.ndimage.zoom
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import numpy as np
from scipy.misc import imresize
from scipy.ndimage import zoom


def adapt_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    assert(len(image.shape) == 3)
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        if image.shape[2] == 3:
            image = imresize(image, (round(h * scale), round(w * scale)))
        else:
            RGB = imresize(image[:,:,:3], (round(h * scale), round(w * scale)))
            extra_channel = imresize(image[:,:,3], (round(h * scale), round(w * scale)))
            extra_channel = np.expand_dims(extra_channel, axis=-1)
            image = np.concatenate([RGB, extra_channel], axis=-1)
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def adapt_label(label, scale=None, padding=None):
    """Shrink a label using the given scale and padding.
    Scale and padding are from resize_image() to ensure both, the image and the label, are resized consistently.
        Args:
            scale: mask scaling factor
            padding: Padding to add to the mask in the form [(top, bottom), (left, right), (0, 0)]
    """
    h, w = label.shape[:2]
    if scale:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            label = zoom(label, zoom=[scale, scale, 1], order=0)
    if padding:
        label = np.pad(label, padding, mode='constant', constant_values=0)
    return label


def restore_label(label, original_shape, window, scale, padding):
    """Reverses the changes of `adapt_label()`.
        Args:
            label: the resized label to restore
            original_shape: original shape to restore to
            window: (y1, x1, y2, x2). This window is the coordinates of the image part of the full image (excluding
                the padding). The x2, y2 pixels are not included.
            scale: scaling factor to undo
            padding: Padding to undo (in the form [(top, bottom), (left, right), (0, 0)])
    """
    original_h, original_w = original_shape[:2]
    if padding:
        top_pad, _ = padding[0]
        left_pad, _ = padding[1]
        window_h = window[2] - window[0]
        window_w = window[3] - window[1]
        label = label[top_pad:top_pad+window_h, left_pad:left_pad+window_w]
    if scale != 1:
        restored_label = np.zeros((original_h, original_w, 1), dtype=label.dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zoom(label, zoom=[1/scale, 1/scale, 1], output=restored_label, order=0)
            # restored_label = zoom(label, zoom=[1/scale, 1/scale, 1], order=0)
    else:
        restored_label = label
    assert(original_shape[:2] == restored_label.shape[:2])
    return restored_label


# def pack_adapt_info(original_shape, window, scale, padding):
#     """Takes adapt info for ONE image and packs it in ONE 1D array.
#     Args:
#         original_shape: [height, width, channels]
#         window: (y1, x1, y2, x2) in pixels. The area of the image where the real
#                 image is (excluding the padding)
#         scale: The scale factor used to resize the image
#         padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
#     Ret:
#         adapt_info: packed adapt info
#     """
#     # adapt_info = np.array(
#     #     list(image_shape) +     # idx 0,1,2
#     #     list(window) +          # idx 3,4,5,6
#     #     list(scale) +           # idx 7
#     #     list(padding)           # idx 8...
#     # )
#     # return adapt_info
#     return (original_shape, window, scale, padding)
#
#
# def unpack_adapt_info(adapt_info):
#     """Unpacks adapt info for ONE image.
#     See pack_adapt_info() for more details.
#     Args:
#         adapt_info: packed resize info
#     Ret:
#         List of individual resize info structs
#     """
#     # image_shape = meta[:, 0:3] # idx 0,1,2
#     # window = meta[:, 3:7]      # idx 3,4,5,6
#     # scale = meta[:, 7]         # idx 7
#     # padding = meta[:, 8:]      # idx 8...
#     # return [image_shape, window, scale, padding]
#     original_shape, window, scale, padding = adapt_info
#     return original_shape, window, scale, padding


# def test_basic_behavior_3channels():
#
#     import visualize
#     from dataset import _DEFAULT_DSB18_OPTIONS, DSB18Dataset
#     from model import _DEFAULT_MODEL_OPTIONS, Model, _DEBUG_PROCESSING_STEPS
#     from tqdm import trange
#     from visualize import archive_images
#
#     _DEBUG_ADAPT_STEPS = True
#
#     if _DEBUG_ADAPT_STEPS:
#         save_folder1 = "c:/temp/visualizations1"
#         save_folder2 = "c:/temp/visualizations2"
#         save_folder3 = "c:/temp/visualizations3"
#         save_folder4 = "c:/temp/visualizations4"
#         save_folder5 = "c:/temp/visualizations5"
#         save_folder6 = "c:/temp/visualizations6"
#
#     # Load dataset (using semantic labels)
#     options = _DEFAULT_DSB18_OPTIONS
#     options['mode'] = 'semantic_labels'
#     ds = DSB18Dataset(phase='train_val', options=options)
#
#     # Display dataset configuration
#     ds.print_config()
#
#     # Load model
#     options = _DEFAULT_MODEL_OPTIONS
#     options['unique_img_size'] = None # [None | (IMAGE_MIN_DIM, IMAGE_MAX_DIM)]
#     model = Model(ds, phase='train', options=options)
#
#     # Display model configuration
#     model.print_config()
#
#     # Parameters
#     num_rounds = 16
#     batch_size = 1
#
#     # Test with semantic labels
#     if _DEBUG_ADAPT_STEPS:
#         debug_samples, debug_labels, debug_restored_labels = [], [], []
#     desc = 'Processing sem labels {} at a time'.format(batch_size)
#     for _ in trange(num_rounds, ascii=True, ncols=100, desc=desc):
#         samples, gt_labels = model._ds.next_batch_sem_labels(batch_size, 'train') # (1, W, H, 3),  (1, W, H, 1)
#         if _DEBUG_ADAPT_STEPS:
#             debug_samples.append(np.squeeze(samples, axis=0))
#         inputs, adapt_info = model._preprocess_inputs(samples)
#         labels = model._preprocess_labels(gt_labels, adapt_info)
#         if _DEBUG_ADAPT_STEPS:
#             debug_labels.append(np.squeeze(labels, axis=0))
#         restored_labels = model._postprocess_labels(labels, adapt_info)
#         if _DEBUG_ADAPT_STEPS:
#             debug_restored_labels.append(np.squeeze(restored_labels, axis=0))
#         # assert(gt_labels.shape == restored_labels.shape)
#         # if _DEBUG_PROCESSING_STEPS:
#         #     print(np.mean(gt_labels), np.mean(restored_labels))
#         #     print(np.std(gt_labels), np.std(restored_labels))
#         # assert(np.isclose(np.mean(gt_labels), np.mean(restored_labels), rtol=1e-01, atol=1e-02))
#         # assert(np.isclose(np.std(gt_labels), np.std(restored_labels), rtol=1e-01, atol=1e-02))
#     if _DEBUG_ADAPT_STEPS:
#         archive_images(debug_samples, save_folder4)
#         archive_images(debug_labels, save_folder5)
#         archive_images(debug_restored_labels, save_folder6)
#
#     # Load model
#     options = _DEFAULT_MODEL_OPTIONS
#     options['unique_img_size'] = (None, 704) # [None | (IMAGE_MIN_DIM, IMAGE_MAX_DIM)]
#     model = Model(ds, phase='train', options=options)
#
#     # Display model configuration
#     model.print_config()
#
#     # Parameters
#     num_rounds = 16
#     batch_size = 1
#
#     # Test with semantic labels
#     if _DEBUG_ADAPT_STEPS:
#         debug_samples, debug_labels, debug_restored_labels = [], [], []
#     desc = 'Processing sem labels {} at a time'.format(batch_size)
#     for _ in trange(num_rounds, ascii=True, ncols=100, desc=desc):
#         samples, gt_labels = model._ds.next_batch_sem_labels(batch_size, 'train') # (1, W, H, 3),  (1, W, H, 1)
#         if _DEBUG_ADAPT_STEPS:
#             debug_samples.append(np.squeeze(samples, axis=0))
#         inputs, adapt_info = model._preprocess_inputs(samples)
#         assert (inputs.shape == (1,704,704,3))
#         labels = model._preprocess_labels(gt_labels, adapt_info)
#         if _DEBUG_ADAPT_STEPS:
#             debug_labels.append(np.squeeze(labels, axis=0))
#         restored_labels = model._postprocess_labels(labels, adapt_info)
#         if _DEBUG_ADAPT_STEPS:
#             debug_restored_labels.append(np.squeeze(restored_labels, axis=0))
#         # assert(gt_labels.shape == restored_labels.shape)
#         # if _DEBUG_PROCESSING_STEPS:
#         #     print(np.mean(gt_labels), np.mean(restored_labels))
#         #     print(np.std(gt_labels), np.std(restored_labels))
#         # assert(np.isclose(np.mean(gt_labels), np.mean(restored_labels), rtol=1e-01, atol=1e-02))
#         # assert(np.isclose(np.std(gt_labels), np.std(restored_labels), rtol=1e-01, atol=1e-02))
#     if _DEBUG_ADAPT_STEPS:
#         archive_images(debug_samples, save_folder4)
#         archive_images(debug_labels, save_folder5)
#         archive_images(debug_restored_labels, save_folder6)
#
#     # Parameters
#     num_rounds = 32
#     batch_size = 16
#
#     # Test with semantic labels
#     desc = 'Processing sem labels {} at a time'.format(batch_size)
#     for _ in trange(num_rounds, ascii=True, ncols=100, desc=desc):
#         samples, gt_labels = model._ds.next_batch_sem_labels(batch_size, 'train') # [W, H, 3], [W, H, 1] of size batch_size
#         if _DEBUG_ADAPT_STEPS:
#             archive_images(samples, save_folder1)
#         inputs, adapt_info = model._preprocess_inputs(samples)
#         assert (inputs.shape == (batch_size,704,704,3))
#         labels = model._preprocess_labels(gt_labels, adapt_info)
#         if _DEBUG_ADAPT_STEPS:
#             archive_images(labels, save_folder2)
#         restored_labels = model._postprocess_labels(labels, adapt_info)
#         if _DEBUG_ADAPT_STEPS:
#             archive_images(restored_labels, save_folder3)
#
#
#     # Load model
#     options = _DEFAULT_MODEL_OPTIONS
#     options['unique_img_size'] = None
#     model = Model(ds, phase='train', options=options)
#
#     # Display model configuration
#     model.print_config()
#
#     # Test with semantic labels
#     for _ in range(num_samples):
#         samples, gt_labels = model._ds.next_batch_sem_labels(batch_size, 'train')
#         inputs, adapt_info = model._preprocess_inputs(samples)
#         labels = model._preprocess_labels(gt_labels, adapt_info)
#         restored_labels = model._postprocess_labels(labels, adapt_info)
#         assert(np.array_equal(gt_labels, restored_labels))
#
# def test_basic_behavior_4channels():
#
#     import visualize
#     from dataset import _DEFAULT_DSB18_OPTIONS, DSB18Dataset
#     from model import _DEFAULT_MODEL_OPTIONS, Model, _DEBUG_PROCESSING_STEPS
#     from tqdm import trange
#     from visualize import archive_images
#
#     _DEBUG_ADAPT_STEPS = True
#
#     if _DEBUG_ADAPT_STEPS:
#         save_folder1 = "c:/temp/visualizations1"
#         save_folder2 = "c:/temp/visualizations2"
#         save_folder3 = "c:/temp/visualizations3"
#         save_folder4 = "c:/temp/visualizations4"
#         save_folder5 = "c:/temp/visualizations5"
#         save_folder6 = "c:/temp/visualizations6"
#         save_folder7 = "c:/temp/visualizations7"
#         save_folder8 = "c:/temp/visualizations8"
#         save_folder9 = "c:/temp/visualizations9"
#
#     # Load dataset (using semantic labels)
#     options = _DEFAULT_DSB18_OPTIONS
#     options['mode'] = 'semantic_contours'
#     options['input_channels'] = 4
#     ds = DSB18Dataset(phase='train_val', options=options)
#
#     # Display dataset configuration
#     ds.print_config()
#
#     # Load model
#     options = _DEFAULT_MODEL_OPTIONS
#     options['unique_img_size'] = None # [None | (IMAGE_MIN_DIM, IMAGE_MAX_DIM)]
#     model = Model(ds, phase='train', options=options)
#
#     # Display model configuration
#     model.print_config()
#
#     ##
#     ## in_memory set to False, unique_img_size set to None, batch_size set to 1
#     ##
#
#     # Parameters
#     num_rounds = 16
#     batch_size = 1
#
#     # Test with semantic labels
#     if _DEBUG_ADAPT_STEPS:
#         debug_samples, debug_labels, debug_restored_labels = [], [], []
#     desc = 'Processing sem contours {} at a time'.format(batch_size)
#     for _ in trange(num_rounds, ascii=True, ncols=100, desc=desc):
#         samples, gt_labels = model._ds.next_batch_sem_labels(batch_size, 'train') # (1, W, H, 4),  (1, W, H, 1)
#         if _DEBUG_ADAPT_STEPS:
#             debug_samples.append(np.squeeze(samples, axis=0))
#         inputs, adapt_info = model._preprocess_inputs(samples)
#         labels = model._preprocess_labels(gt_labels, adapt_info)
#         if _DEBUG_ADAPT_STEPS:
#             debug_labels.append(np.squeeze(labels, axis=0))
#         restored_labels = model._postprocess_labels(labels, adapt_info)
#         if _DEBUG_ADAPT_STEPS:
#             debug_restored_labels.append(np.squeeze(restored_labels, axis=0))
#         # assert(gt_labels.shape == restored_labels.shape)
#         # if _DEBUG_PROCESSING_STEPS:
#         #     print(np.mean(gt_labels), np.mean(restored_labels))
#         #     print(np.std(gt_labels), np.std(restored_labels))
#         # assert(np.isclose(np.mean(gt_labels), np.mean(restored_labels), rtol=1e-01, atol=1e-02))
#         # assert(np.isclose(np.std(gt_labels), np.std(restored_labels), rtol=1e-01, atol=1e-02))
#     if _DEBUG_ADAPT_STEPS:
#         archive_images(debug_samples, save_folder1)
#         archive_images(debug_labels, save_folder2)
#         archive_images(debug_restored_labels, save_folder3)
#
#     ##
#     ## in_memory set to False, unique_img_size set to (None, 704), batch_size set to 1
#     ##
#
#     # Load model
#     options = _DEFAULT_MODEL_OPTIONS
#     options['unique_img_size'] = (None, 704) # [None | (IMAGE_MIN_DIM, IMAGE_MAX_DIM)]
#     model = Model(ds, phase='train', options=options)
#
#     # Display model configuration
#     model.print_config()
#
#     # Parameters
#     num_rounds = 16
#     batch_size = 1
#
#     # Test with semantic labels
#     if _DEBUG_ADAPT_STEPS:
#         debug_samples, debug_labels, debug_restored_labels = [], [], []
#     desc = 'Processing sem contours {} at a time'.format(batch_size)
#     for _ in trange(num_rounds, ascii=True, ncols=100, desc=desc):
#         samples, gt_labels = model._ds.next_batch_sem_labels(batch_size, 'train') # (1, W, H, 3),  (1, W, H, 1)
#         if _DEBUG_ADAPT_STEPS:
#             debug_samples.append(np.squeeze(samples, axis=0))
#         inputs, adapt_info = model._preprocess_inputs(samples)
#         assert (inputs.shape == (1,704,704,4))
#         labels = model._preprocess_labels(gt_labels, adapt_info)
#         if _DEBUG_ADAPT_STEPS:
#             debug_labels.append(np.squeeze(labels, axis=0))
#         restored_labels = model._postprocess_labels(labels, adapt_info)
#         if _DEBUG_ADAPT_STEPS:
#             debug_restored_labels.append(np.squeeze(restored_labels, axis=0))
#         # assert(gt_labels.shape == restored_labels.shape)
#         # if _DEBUG_PROCESSING_STEPS:
#         #     print(np.mean(gt_labels), np.mean(restored_labels))
#         #     print(np.std(gt_labels), np.std(restored_labels))
#         # assert(np.isclose(np.mean(gt_labels), np.mean(restored_labels), rtol=1e-01, atol=1e-02))
#         # assert(np.isclose(np.std(gt_labels), np.std(restored_labels), rtol=1e-01, atol=1e-02))
#     if _DEBUG_ADAPT_STEPS:
#         archive_images(debug_samples, save_folder4)
#         archive_images(debug_labels, save_folder5)
#         archive_images(debug_restored_labels, save_folder6)
#
#     ##
#     ## in_memory set to False, unique_img_size set to (None, 704), batch_size set to 16
#     ##
#
#     # Parameters
#     num_rounds = 1
#     batch_size = 16
#
#     # Test with semantic labels
#     desc = 'Processing sem contours {} at a time'.format(batch_size)
#     for _ in trange(num_rounds, ascii=True, ncols=100, desc=desc):
#         samples, gt_labels = model._ds.next_batch_sem_labels(batch_size, 'train') # [W, H, 3], [W, H, 1] of size batch_size
#         if _DEBUG_ADAPT_STEPS:
#             archive_images(samples, save_folder1)
#         inputs, adapt_info = model._preprocess_inputs(samples)
#         assert (inputs.shape == (batch_size,704,704,4))
#         labels = model._preprocess_labels(gt_labels, adapt_info)
#         if _DEBUG_ADAPT_STEPS:
#             archive_images(labels, save_folder2)
#         restored_labels = model._postprocess_labels(labels, adapt_info)
#         if _DEBUG_ADAPT_STEPS:
#             archive_images(restored_labels, save_folder3)
#
#     if _DEBUG_ADAPT_STEPS:
#         archive_images(debug_samples, save_folder7)
#         archive_images(debug_labels, save_folder8)
#         archive_images(debug_restored_labels, save_folder9)
#
#     ##
#     ## in_memory set to True, unique_img_size set to None, batch_size set to 1
#     ##
#
#     # Load dataset (using semantic labels)
#     options = _DEFAULT_DSB18_OPTIONS
#     options['mode'] = 'semantic_contours'
#     options['in_memory'] = True
#     options['input_channels'] = 4
#     ds = DSB18Dataset(phase='train_val', options=options)
#
#     # Display dataset configuration
#     ds.print_config()
#
#     # Load model
#     options = _DEFAULT_MODEL_OPTIONS
#     options['unique_img_size'] = None # [None | (IMAGE_MIN_DIM, IMAGE_MAX_DIM)]
#     model = Model(ds, phase='train', options=options)
#
#     # Display model configuration
#     model.print_config()
#
#     # Parameters
#     num_rounds = 16
#     batch_size = 1
#
#     # Test with semantic labels
#     if _DEBUG_ADAPT_STEPS:
#         debug_samples, debug_labels, debug_restored_labels = [], [], []
#     desc = 'Processing sem contours {} at a time'.format(batch_size)
#     for _ in trange(num_rounds, ascii=True, ncols=100, desc=desc):
#         samples, gt_labels = model._ds.next_batch_sem_labels(batch_size, 'train') # (1, W, H, 4),  (1, W, H, 1)
#         if _DEBUG_ADAPT_STEPS:
#             debug_samples.append(np.squeeze(samples, axis=0))
#         inputs, adapt_info = model._preprocess_inputs(samples)
#         labels = model._preprocess_labels(gt_labels, adapt_info)
#         if _DEBUG_ADAPT_STEPS:
#             debug_labels.append(np.squeeze(labels, axis=0))
#         restored_labels = model._postprocess_labels(labels, adapt_info)
#         if _DEBUG_ADAPT_STEPS:
#             debug_restored_labels.append(np.squeeze(restored_labels, axis=0))
#         # assert(gt_labels.shape == restored_labels.shape)
#         # if _DEBUG_PROCESSING_STEPS:
#         #     print(np.mean(gt_labels), np.mean(restored_labels))
#         #     print(np.std(gt_labels), np.std(restored_labels))
#         # assert(np.isclose(np.mean(gt_labels), np.mean(restored_labels), rtol=1e-01, atol=1e-02))
#         # assert(np.isclose(np.std(gt_labels), np.std(restored_labels), rtol=1e-01, atol=1e-02))
#     if _DEBUG_ADAPT_STEPS:
#         archive_images(debug_samples, save_folder1)
#         archive_images(debug_labels, save_folder2)
#         archive_images(debug_restored_labels, save_folder3)
#
#     ##
#     ## in_memory set to True, unique_img_size set to (None, 704), batch_size set to 1
#     ##
#
#     # Load model
#     options = _DEFAULT_MODEL_OPTIONS
#     options['unique_img_size'] = (None, 704) # [None | (IMAGE_MIN_DIM, IMAGE_MAX_DIM)]
#     model = Model(ds, phase='train', options=options)
#
#     # Display model configuration
#     model.print_config()
#
#     # Parameters
#     num_rounds = 16
#     batch_size = 1
#
#     # Test with semantic labels
#     if _DEBUG_ADAPT_STEPS:
#         debug_samples, debug_labels, debug_restored_labels = [], [], []
#     desc = 'Processing sem contours {} at a time'.format(batch_size)
#     for _ in trange(num_rounds, ascii=True, ncols=100, desc=desc):
#         samples, gt_labels = model._ds.next_batch_sem_labels(batch_size, 'train') # (1, W, H, 3),  (1, W, H, 1)
#         if _DEBUG_ADAPT_STEPS:
#             debug_samples.append(np.squeeze(samples, axis=0))
#         inputs, adapt_info = model._preprocess_inputs(samples)
#         assert (inputs.shape == (batch_size,704,704,4))
#         labels = model._preprocess_labels(gt_labels, adapt_info)
#         if _DEBUG_ADAPT_STEPS:
#             debug_labels.append(np.squeeze(labels, axis=0))
#         restored_labels = model._postprocess_labels(labels, adapt_info)
#         if _DEBUG_ADAPT_STEPS:
#             debug_restored_labels.append(np.squeeze(restored_labels, axis=0))
#         # assert(gt_labels.shape == restored_labels.shape)
#         # if _DEBUG_PROCESSING_STEPS:
#         #     print(np.mean(gt_labels), np.mean(restored_labels))
#         #     print(np.std(gt_labels), np.std(restored_labels))
#         # assert(np.isclose(np.mean(gt_labels), np.mean(restored_labels), rtol=1e-01, atol=1e-02))
#         # assert(np.isclose(np.std(gt_labels), np.std(restored_labels), rtol=1e-01, atol=1e-02))
#     if _DEBUG_ADAPT_STEPS:
#         archive_images(debug_samples, save_folder4)
#         archive_images(debug_labels, save_folder5)
#         archive_images(debug_restored_labels, save_folder6)
#
#     ##
#     ## in_memory set to True, unique_img_size set to (None, 704), batch_size set to 16
#     ##
#
#     # Parameters
#     num_rounds = 1
#     batch_size = 16
#
#     # Test with semantic labels
#     desc = 'Processing sem contours {} at a time'.format(batch_size)
#     for _ in trange(num_rounds, ascii=True, ncols=100, desc=desc):
#         samples, gt_labels = model._ds.next_batch_sem_labels(batch_size, 'train') # [W, H, 3], [W, H, 1] of size batch_size
#         if _DEBUG_ADAPT_STEPS:
#             archive_images(samples, save_folder1)
#         inputs, adapt_info = model._preprocess_inputs(samples)
#         assert (inputs.shape == (batch_size,704,704,4))
#         labels = model._preprocess_labels(gt_labels, adapt_info)
#         if _DEBUG_ADAPT_STEPS:
#             archive_images(labels, save_folder2)
#         restored_labels = model._postprocess_labels(labels, adapt_info)
#         if _DEBUG_ADAPT_STEPS:
#             archive_images(restored_labels, save_folder3)
#
#     if _DEBUG_ADAPT_STEPS:
#         archive_images(debug_samples, save_folder7)
#         archive_images(debug_labels, save_folder8)
#         archive_images(debug_restored_labels, save_folder9)
#
# if __name__ == '__main__':
#     # test_basic_behavior_3channels()
#     test_basic_behavior_4channels()
