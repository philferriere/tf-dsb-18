"""
visualize.py

Visualization helpers.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Based on:
    - https://github.com/matterport/Mask_RCNN/blob/master/visualize.py
        Copyright (c) 2017 Matterport, Inc. / Written by Waleed Abdulla
        Licensed under the MIT License

References for future work:
    E:/repos/models-master/research/object_detection/utils/visualization_utils.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, random, warnings
import colorsys
import numpy as np
import matplotlib.pyplot as plt

from segment import inst_masks_to_sem_label
from skimage.io import imsave

def random_colors(N, bright=True, RGB_max=255):
    """
    Generate random colors. To get visually distinct colors, generate them in HSV space then convert to RGB.
        Args:
            N: number of colors to generate.
            bright: set to True for bright colors.
            RGB_max: set to 1.0 or 255, based on image type you're working with
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = [(color[0] * RGB_max, color[1] * RGB_max, color[2] * RGB_max) for color in colors]
    random.shuffle(colors)
    return colors

def display_images(images, titles=None, cols=4, cmaps=None, norm=None, interpolation=None):
    """Display the given set of images, optionally with titles.
        Args:
            images: list or array of image tensors in HWC format.
            titles: optional. A list of titles to display with each image.
            cols: number of images per row
            cmaps: Optional. Color maps to use. For example, "Blues". Can be None, one cmap, or a list of them.
            norm: Optional. A Normalize instance to map values to colors.
            interpolation: Optional. Image interporlation to use for display.
    """
    rows = len(images) // cols + 1
    width = 20
    plt.figure(figsize=(width, width * rows // cols))
    if type(cmaps) is not list:
        cmaps = [cmaps] * len(images)
    i = 1
    if titles:
        for image, title, cmap in zip(images, titles, cmaps):
            plt.subplot(rows, cols, i)
            plt.title(title, fontsize=width*2)
            plt.axis('off')
            plt.imshow(image.astype(np.uint8), cmap=cmap, norm=norm, interpolation=interpolation)
            i += 1
    else:
        for image, cmap in zip(images, cmaps):
            plt.subplot(rows, cols, i)
            plt.axis('off')
            plt.imshow(image.astype(np.uint8), cmap=cmap, norm=norm, interpolation=interpolation)
            i += 1
    plt.tight_layout()
    plt.show()

def display_images_with_labels(images, labels, color=(0, 255, 0), titles=None, cols=4, cmaps=None, norm=None, interpolation=None):
    """Display the given set of images, optionally with titles.
        Args:
            images: list or array of image tensors in HWC format.
            labels: list or array of label tensors in HWC format.
            color: optional. Color to use for the label overlay.
            titles: optional. A list of titles to display with each image.
            cols: number of images per row
            cmaps: Optional. Color maps to use. For example, "Blues". Can be None, one cmap, or a list of them.
            norm: Optional. A Normalize instance to map values to colors.
            interpolation: Optional. Image interporlation to use for display.
    """
    rows = len(images) // cols + 1
    width = 20
    plt.figure(figsize=(width, width * rows // cols))
    if type(cmaps) is not list:
        cmaps = [cmaps] * len(images)
    i = 1
    if titles:
        for image, label, title, cmap in zip(images, labels, titles, cmaps):
            plt.subplot(rows, cols, i)
            plt.title(title, fontsize=width*2)
            plt.axis('off')
            image = draw_mask(image, label, color, alpha=0.2)
            plt.imshow(image.astype(np.uint8), cmap=cmap, norm=norm, interpolation=interpolation)
            i += 1
    else:
        for image, label, cmap in zip(images, labels, cmaps):
            plt.subplot(rows, cols, i)
            plt.axis('off')
            image = draw_mask(image, label, color, alpha=0.2)
            plt.imshow(image.astype(np.uint8), cmap=cmap, norm=norm, interpolation=interpolation)
            i += 1
    plt.tight_layout()
    plt.show()

def archive_images_with_labels(images, labels, folder, color=(0, 255, 0), alpha=0.2):
    """Save to file the given set of images with labels overlaid.
        Args:
            images: list or array of image tensors in HWC format.
            labels: list or array of label tensors in HWC format.
            folder: where to archive the composite images.
            color: optional. Color to use for the label overlay.
            alpha: Optional. Alpha-blending factor.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        i = 0
        for image, label in zip(images, labels):
            image = draw_mask(image, label, color, alpha)
            output_file = folder + "/{0:04d}.png".format(i)
            if os.path.exists(output_file):
                os.remove(output_file)
            imsave(output_file, image)
            i += 1

def archive_images_with_contours_and_centers(images, contours, centers, folder, center_color=(0, 255, 0),
                                             contour_color=(255, 0, 0), alpha=0.2):
    """Save to file the given set of images with labels overlaid.
        Args:
            images: list or array of image tensors in HWC format.
            contours: list or array of contour tensors in HW format.
            centers: list or array of center tensors in HW format.
            folder: where to archive the composite images.
            color: optional. Color to use for the label overlay.
            alpha: Optional. Alpha-blending factor.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        i = 0
        for image, contour, center in zip(images, contours, centers):
            # Adapt to draw_mask's expected input shapes
            if len(image.shape) == 3 and image.shape[2] == 1:
                image_RGB = np.concatenate((image, image, image), axis=-1) # (H,W,1) -> (H,W,3)
            else:
                image_RGB = image
            if len(contour.shape) == 2:
                contour_ex = np.expand_dims(contour, -1) # (H,W) -> (H,W,1)
            else:
                contour_ex = contour
            if len(center.shape) == 2:
                center_ex = np.expand_dims(center, -1) # (H,W) -> (H,W,1)
            else:
                center_ex = center
            image_RGB = draw_mask(image_RGB, contour_ex, contour_color, alpha)
            draw_mask(image_RGB, center_ex, center_color, alpha, in_place=True)
            output_file = folder + "/{0:04d}.png".format(i)
            if os.path.exists(output_file):
                os.remove(output_file)
            imsave(output_file, image_RGB)
            i += 1

def archive_images(images, folder, titles=None):
    """Save to file the given set of images.
        Args:
            images: list or array of image tensors in HWC format.
            folder: where to archive the composite images.
            color: optional. Color to use for the label overlay.
            alpha: Optional. Alpha-blending factor.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        i = 0
        for image in images:
            if len(image.shape) == 3 and image.shape[2] == 1:
                image_RGB = np.concatenate((image, image, image), axis=-1) # (H,W,1) -> (H,W,3)
            else:
                image_RGB = image
            if titles:
                filename = "/{}.png".format(titles[i])
            else:
                filename = "/{0:04d}.png".format(i)
            output_file = folder + filename
            if os.path.exists(output_file):
                os.remove(output_file)
            imsave(output_file, image_RGB)
            i += 1


def display_sem_label_gt_vs_pred(gt_sem_label, pred_sem_label, ID):
    """Show the overlap between a ground truth semantic label and a predicted one.
        Args:
            gt_sem_label: np array in (H, W, 1) format.
            pred_sem_label: np array in (H, W, 1) format.
            ID: Image ID
        Notes:
            If we define a positive event to be the detection of an object when an object is in fact present:
            False Negative: An object is not detected when there is an object.
            False Positive: An object is detected when there isn't an object.
            True Positive: An object is detected where there is an object.
            True Negative: An object is not detected where there is not an object.
    """
    # print(gt_sem_label.shape, pred_sem_label.shape, gt_sem_label.dtype, pred_sem_label.dtype)
    # Adjust input fomat, if necesary
    if len(gt_sem_label.shape) == 3 and gt_sem_label.shape[2] == 1:
        gt_sem_label = np.squeeze(gt_sem_label, -1)
    if len(pred_sem_label.shape) == 3 and pred_sem_label.shape[2] == 1:
        pred_sem_label = np.squeeze(pred_sem_label, -1)

    # Display TP(G),FN(R),FP(B)
    true_pos = gt_sem_label & pred_sem_label
    false_neg = gt_sem_label - true_pos
    false_pos = pred_sem_label - true_pos
    overlap = np.stack((false_neg, true_pos, false_pos), axis=-1).astype(np.uint8)
    images = [overlap, true_pos, false_neg, false_pos]
    titles = [ID[:5]+" TP(G),FN(R),FP(B)", ID[:5]+" TPs (correct objects)",
              ID[:5]+" FNs (missed objects)", ID[:5]+" FPs (bogus additions)"]

    # cmaps = [None, "gray", "gray", "gray"]
    display_images(images, titles, cols=2, cmaps="gray")

def display_sem_label_gt_and_pred(image, gt_sem_label, pred_sem_label, ID):
    """Display two images, one with ground truth semantic label, one with predicted semantic label.
        Args:
            image: input image (H, W, 3)
            gt_sem_label: np array in (H, W, 1) format.
            pred_sem_label: np array in (H, W, 1) format.
            ID: Image ID
        Notes:
            If we define a positive event to be the detection of an object when an object is in fact present:
            False Negative: An object is not detected when there is an object.
            False Positive: An object is detected when there isn't an object.
            True Positive: An object is detected where there is an object.
            True Negative: An object is not detected where there is not an object.
    """
    if len(image.shape)==3 and image.shape[2]==4:
        image = image[:,:,:3]
    _sem_label = np.expand_dims(gt_sem_label, 0) # (H, W, 1) -> (1, H, W, 1)
    gt =  draw_masks(image, None, _sem_label, in_place=False)

    _sem_label = np.expand_dims(pred_sem_label, 0) # (H, W, 1) -> (1, H, W, 1)
    pred =  draw_masks(image, None, _sem_label, in_place=False)

    titles = [ID[:5]+" RGB+Gt",  ID[:5]+" RGB+Pred"]

    display_images([gt, pred], titles, cols=2)

def display_image_and_pred_sem_label(image, pred_sem_label, ID):
    """Display two images, one with ground truth semantic label, one with predicted semantic label.
        Args:
            image: input image (H, W, 3) or  (H, W, 4)
            pred_sem_label: np array in (H, W, 1) format.
            ID: Image ID
    """
    if len(image.shape)==3 and image.shape[2]==4:
        image = image[:,:,:3]
    _sem_label = np.expand_dims(pred_sem_label, 0) # (H, W, 1) -> (1, H, W, 1)
    pred =  draw_masks(image, None, _sem_label, in_place=False)

    titles = [ID[:5]+" RGB",  ID[:5]+" RGB+Pred"]

    display_images([image, pred], titles, cols=2)

def display_inst_masks_gt_vs_pred(gt_inst_masks, pred_inst_masks, order='instances_first'):
    """Show the overlap between a ground truth set of instance masks and predicted ones.
    This is useful to identify issues with the semantic segmentation.
        Args:
            gt_inst_masks: np array
            pred_inst_masks: np array
            order: are np arrays in [instance count, H, W] or [H, W, instance count] format?
    """
    gt_sem_label = inst_masks_to_sem_label(gt_inst_masks)
    pred_sem_label = inst_masks_to_sem_label(pred_inst_masks)
    display_sem_label_gt_vs_pred(gt_sem_label, pred_sem_label)

def draw_box(image, bbox, color, in_place=True):
    """Draw (in-place, or not) 3-pixel-width bounding bboxes on an image.
        Args:
            image: image (H,W,3) or (H,W,1) 
            bbox: y1, x1, y2, x2 bounding box
            color: color list of 3 int values for RGB
            in_place: in place / copy flag
        Returns:
            image with bounding box
    """
    y1, x1, y2, x2 = bbox
    result = image if in_place == True else np.copy(image)
    result[y1:y1 + 2, x1:x2] = color
    result[y2:y2 + 2, x1:x2] = color
    result[y1:y2, x1:x1 + 2] = color
    result[y1:y2, x2:x2 + 2] = color
    return result

def draw_centroid(image, centroid, color=255, in_place=True):
    """Draw (in-place, or not) 3-pixel-width bounding bboxes on an image.
        Args:
            imaged: image (H,W,3) or (H,W,1)
            bbox: y1, x1, y2, x2 bounding box
            color: grey level of centroid
            in_place: in place / copy flag
        Returns:
            image with bounding box
    """
    cx, cy = centroid[0], centroid[1]
    if cx==0:
        cx += 1
    if cy==0:
        cy += 1
    if cx==image.shape[1]-1:
        cx -= 1
    if cy==image.shape[0]-1:
        cy -= 1
    result = image if in_place == True else np.copy(image)
    result[cy-1:cy+1+1, cx] = color # When slicing, +1 is needed?
    result[cy, cx-1:cx+1+1] = color
    return result

def draw_mask(image, mask, color, alpha=0.7, in_place=False):
    """Draw (in-place, or not) a mask on an image.
        Args:
            image: input image (H,W,3)
            mask: mask (H,W,1)
            color: color list of 3 int values for RGB
            alpha: alpha blending level
            in_place: in place / copy flag
        Returns:
            image with mask
    """
    # print(image.shape, mask.shape)
    assert(len(image.shape) == len(mask.shape) == len(color) == 3)
    assert(image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1])
    threshold = (np.max(mask) - np.min(mask)) / 2
    multiplier = 1 if np.amax(color) > 1 else 255
    masked_image = image if in_place == True else np.copy(image)
    for c in range(3):
        masked_image[:, :, c] = np.where(mask[:,:,0] > threshold,
                                         masked_image[:, :, c] *
                                         (1 - alpha) + alpha * color[c] * multiplier,
                                         masked_image[:, :, c])
    return masked_image

def draw_masks(image, bboxes, masks, alpha=0.7, in_place=True):
    """Apply the given instance masks to the image and draw their bboxes.
        Args:
            image: input image (H, W, 3)
            bboxes: (num_instances, (y1, x1, y2, x2)) bounding boxes as numpy array
            masks: masks (num_instances, H, W, 1) as numpy array
            alpha: alpha blending level
            in_place: in place / copy flag
        Returns:
            image with masks overlaid
    """
    # Number of instances
    num_instances = masks.shape[0]
    if bboxes is not None:
        assert (num_instances == bboxes.shape[0])

    # Make a copy of the input image, if requested
    masked_image = image if in_place == True else np.copy(image)

    # Adapt to draw_mask's expected input shapes
    if len(masked_image.shape) == 3 and masked_image.shape[2] == 1:
        image_RGB = np.concatenate((masked_image, masked_image, masked_image), axis=-1)  # (H,W,1) -> (H,W,3)
    elif len(masked_image.shape) == 3 and masked_image.shape[2] == 4:
        image_RGB = masked_image[:, :, :3] # (H,W,4) -> (H,W,3)
    elif len(masked_image.shape) == 2:
        masked_image = np.expand_dims(masked_image, axis=-1)
        image_RGB = np.concatenate((masked_image, masked_image, masked_image), axis=-1)  # (H,W,1) -> (H,W,3)
    else:
        image_RGB = masked_image

    # Draw bboxes and masks on the image, if the bbox is not empty, using a random color
    colors = random_colors(num_instances)

    if bboxes is not None:
        for instance in range(num_instances):
            if not np.any(bboxes[instance]):
                continue
            color = colors[instance]
            draw_mask(image_RGB, masks[instance], color, alpha=alpha, in_place=True)
            draw_box(image_RGB, bboxes[instance], color, in_place=True)
    else:
        for instance in range(num_instances):
            color = colors[instance]
            draw_mask(image_RGB, masks[instance], color, alpha=alpha, in_place=True)

    return image_RGB

def display_instances(image, bboxes, masks, figsize=(16, 16), ax=None):
    """Display image with its individual instance masks and their bounding boxes.
    Args:
        image: input image (H, W, 3)
        bboxes: [num_instances, (y1, x1, y2, x2)] in image coordinates.
        masks: [num_instances, height, width]
        figsize: (optional) the size of the image.
    """
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    # _masks = np.rollaxis(np.asarray(masks), 2, 0) # IFF [num_instances, height, width] instead for masks
    _masks = np.expand_dims(np.asarray(masks), -1) # [num_instances, height, width, 1]
    masked_image =  draw_masks(image, bboxes, _masks, in_place=False)
    ax.imshow(masked_image)
    plt.show()

def archive_instances(images, maskss, folder, titles=None):
    """Save to file the given set of images.
        Args:
            images: list or array of image tensors in HWC format.
            maskss: list or array of mask tensors in HW or HWC format.
            folder: where to archive the composite images.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        i = 0
        for image, masks in zip(images, maskss):
            if type(masks) is list and len(masks)>0 and len(masks[0].shape) == 2:
                _masks = np.expand_dims(np.asarray(masks), -1) # [num_instances, height, width, 1]
            elif type(masks) is np.ndarray and len(masks) > 0 and len(masks[0].shape) == 2:
                _masks = np.expand_dims(masks, -1)  # [num_instances, height, width, 1]
            else:
                _masks = masks
            masked_image =  draw_masks(image, None, _masks, in_place=False)
            if titles:
                filename = "/{}.png".format(titles[i])
            else:
                filename = "/{0:04d}.png".format(i)
            output_file = folder + filename
            if os.path.exists(output_file):
                os.remove(output_file)
            imsave(folder + filename, masked_image)
            i += 1

def display_gt_and_pred_masks(image, gt_masks, pred_masks):
    """Display two images, one with gt masks, one with predicted instance masks.
    Each mask is displayed in a unique color. This is useful to debug issues with instance separation.
    Args:
        image: input image (H, W, 3)
        masks: [height, width, num_instances1]
        pred_masks: [height, width, num_instances2]
    """
    _masks = np.expand_dims(gt_masks, -1) # [num_instances, height, width, 1]
    gt =  draw_masks(image, None, _masks, in_place=False)

    _masks = np.expand_dims(pred_masks, -1) # [num_instances, height, width, 1]
    pred =  draw_masks(image, None, _masks, in_place=False)

    display_images([gt, pred], cols=2)

def display_gt_and_pred(image, bboxes, masks, pred_bboxes, pred_masks, figsize=(16, 16), ax=None):
    """Display image with its individual instance masks and their bounding boxes.
    TODO: Look at display_top_masks to see how to display images side by side
    TODO: Show image_with_gt, image_with_predictions
    Args:
        image: input image (H, W, 3)
        bboxes: [num_instances1, (y1, x1, y2, x2)] in image coordinates.
        masks: [height, width, num_instances1]
        pred_bboxes: [num_instances2, (y1, x1, y2, x2)] in image coordinates.
        pred_masks: [height, width, num_instances2]
        figsize: (optional) the size of the image.
    """
    if not bboxes.shape[0]:
        print("\n*** No instances to display *** \n")
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    _masks = np.rollaxis(np.asarray(masks), 2, 0)
    _masks = np.expand_dims(_masks, -1) # [num_instances, height, width, 1]
    gt =  draw_masks(image, bboxes, _masks, in_place=False)

    _masks = np.rollaxis(np.asarray(pred_masks), 2, 0)
    _masks = np.expand_dims(_masks, -1) # [num_instances, height, width, 1]
    pred =  draw_masks(image, pred_bboxes, _masks, in_place=False)

    display_images([gt, pred], cols=2)

def display_image_and_pred_masks(image, pred_masks, ID=None):
    """Display two images: original RGB image and RGB image with predicted instance masks overlayed.
    Each mask is displayed in a unique color. This is useful to debug issues with instance separation.
        Args:
            image: input image (H, W, 3)
            pred_masks: [height, width, num_instances2]
            ID: Image ID
    """
    _masks = np.expand_dims(pred_masks, -1) # (#, H, W) -> (#, H, W, 1)
    pred =  draw_masks(image, None, _masks, in_place=False)

    if ID:
        titles = [ID[:5]+" RGB",  ID[:5]+" RGB+Pred"]
    else:
        titles = None

    display_images([image, pred], titles, cols=2)

# def display_instances(image, boxes, masks, title="", figsize=(16, 16), ax=None):
#     """Display image with its individual instance masks and their bounding boxes.
#     Args:
#         image: input image (H, W, 3)
#         boxes: [num_instance, (y1, x1, y2, x2)] in image coordinates.
#         masks: [height, width, num_instances]
#         title: (optional) title
#         figsize: (optional) the size of the image.
#     """
#     # Number of instances
#     N = boxes.shape[0]
#     if not N:
#         print("\n*** No instances to display *** \n")
#     else:
#         assert boxes.shape[0] == masks.shape[-1]
#
#     if not ax:
#         _, ax = plt.subplots(1, figsize=figsize)
#
#     # Generate random colors
#     colors = random_colors(N)
#
#     # Show area outside image boundaries.
#     height, width = image.shape[:2]
#     ax.set_ylim(height + 10, -10)
#     ax.set_xlim(-10, width + 10)
#     ax.axis('off')
#     ax.set_title(title)
#
#     masked_image = image.astype(np.uint32).copy()
#     for i in range(N):
#         color = colors[i]
#
#         # Bounding box
#         if not np.any(boxes[i]):
#             # Skip this instance. Has no bbox. Likely lost in image cropping.
#             continue
#         y1, x1, y2, x2 = boxes[i]
#         p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
#                               alpha=0.7, linestyle="dashed", edgecolor=color, facecolor='none')
#         ax.add_patch(p)
#
#         # Mask
#         mask = masks[:, :, i]
#         masked_image = apply_mask(masked_image, mask, color, True)
#
#         # Mask Polygon
#         # Pad to ensure proper polygons for masks that touch image edges.
#         padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
#         padded_mask[1:-1, 1:-1] = mask
#         contours = find_contours(padded_mask, 0.5)
#         for verts in contours:
#             # Subtract the padding and flip (y, x) to (x, y)
#             verts = np.fliplr(verts) - 1
#             p = Polygon(verts, facecolor="none", edgecolor=color)
#             ax.add_patch(p)
#     ax.imshow(masked_image.astype(np.uint8))
#     plt.show()
    

def save_instances(image, bboxes, masks, file_path, figsize=(16, 16), ax=None):
    """Save image with its individual instance masks and their bounding boxes.
        image: input image (H, W, 3)
        bboxes: [num_instance, (y1, x1, y2, x2)] in image coordinates.
        masks: [height, width, num_instances]
        figsize: (optional) the size of the image.
        file_path: output PNG
    """
    if not bboxes.shape[0]:
        print("\n*** No instances to display *** \n")
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    _masks = np.rollaxis(np.asarray(masks), 2, 0)
    _masks = np.expand_dims(_masks, -1) # [num_instances, height, width, 1]
    masked_image =  draw_masks(image, bboxes, _masks, in_place=False)
    ax.imshow(masked_image)
    plt.savefig(file_path)

