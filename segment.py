"""
segment.py

Segmentation utility functions.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""

import os, warnings
import numpy as np
from skimage.morphology import label as sk_label
from skimage.io import imsave, imread_collection

def save_sem_label(sem_label, output_file):
    """Save semantic segmentation into a file
    Args:
        sem_label: semantic segmentation in nparray(H,W,1 unot8) format and 0-255 range
        output_file: Path where to save semantic segmentation
    """
    # Convert to a format that imsave likes
    if len(sem_label.shape) == 3 and sem_label.shape[2] == 1:
        sem_label = np.squeeze(sem_label, -1)

    # Create the output folder, if necessary
    output_file_dir = os.path.dirname(output_file)
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)

    # Empty the output folder of [previous predictions
    if os.path.exists(output_file):
        os.remove(output_file)

    # Save semantic segmentation to disk
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(output_file, sem_label)

def sem_label_to_inst_masks(sem_label, as_list=True, order='instances_first'):
    """Convert semantic label to instance masks using connected components.
    Args:
        label: Possible options: 'train', 'val', 'test', or 'val_predpaths'
        as_list: Return as list or np array?
        order: Return np array as [instance count, H, W] or [H, W, instance count]?
    Returns:
        masks: [instance count, H, W]/[H, W, instance count] np array or list([H,W])
    """
    # Split semantic label into instance masks using connected components
    if len(sem_label.shape) == 3 and sem_label.shape[2] == 1:
        lab_img = sk_label(np.squeeze(sem_label, -1) > 128)
    else:
        lab_img = sk_label(sem_label > 128)

    # Build list of instance masks
    masks = [(lab_img == i).astype('uint8') * 255 for i in range(1, lab_img.max() + 1)]

    if as_list:
        return masks
    else:
        if order == 'instances_first':
            return np.asarray(masks)
        else:
            return np.rollaxis(np.asarray(masks), 0, 3)

def inst_masks_to_sem_label(inst_masks, order='instances_first'):
    """Combine instance masks into a semantic label
    Args:
        inst_masks: List or np array of instance masks
        order: Is inst_masks in [instance count, H, W] or [H, W, instance count]?
    Returns:
        sem_label: semantic label
    """
    if type(inst_masks) is list or order == 'instances_first':
        return np.amax(inst_masks, axis=0)
    else:
        return np.amax(inst_masks, axis=-1)

def save_inst_masks(inst_masks, output_folder, order='instances_first'):
    """Save instance masks into a folder
    Args:
        inst_masks: List or np array of instance masks
        order: Is inst_masks np array in [instance count, H, W] or [H, W, instance count]?
        output_folder: Path where to save instance masks
    """
    # Make sure masks are in format np array [num_instances, H, W] or list([H,W])
    if type(inst_masks) is list or order == 'instances_first':
        masks = inst_masks
    else:
        masks = np.rollaxis(inst_masks, 0, 3)

    # Create the output folder, if necessary
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # Empty the output folder of [previous predictions
        # If you don't do this and generate a submission on a folder with the remnants of
        # a previous test run in there, you may add bogus masks to your submission!
        files = [file for file in os.listdir(output_folder) if file.endswith(".png")]
        for file in files:
            os.remove(os.path.join(output_folder, file))

    # Loop through instance masks and save them to disk
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, mask in enumerate(masks, 1):
            output_file = output_folder + "/{0:04d}.png".format(i)
            imsave(output_file, mask)

def load_inst_masks(input_folder, order='instances_first'):
    """Load instance masks from a folder
    Args:
        input_folder: Path from where to load instance masks
        order: Is inst_masks np array in [instance count, H, W] or [H, W, instance count]?
    Returns:
        inst_masks: np array in [instance count, H, W] or [H, W, instance count] format
    """
    inst_masks = imread_collection(input_folder + '/*.png').concatenate()  # np array (#, H, W)
    if order != 'instances_first':
        inst_masks = np.rollaxis(np.asarray(inst_masks), 0, 3)  # np array [H, W, #]

    return inst_masks

def save_labeled_mask(labeled_mask, output_file):
    """Save labeled instance mask into a folder
    Args:
        labeled_mask: Labeled instance mask
        output_file: Path where to save labeled instance mask
    """
    # Create the output folder, if necessary
    output_file_dir = os.path.dirname(output_file)
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)

    # Empty the output folder of [previous predictions
    if os.path.exists(output_file):
        os.remove(output_file)

    # Save semantic segmentation to disk
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(output_file, labeled_mask)
