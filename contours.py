"""
contours.py

Contours utility functions and classes.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Notes:
    Object Contour Detection with a Fully Convolutional Encoder-Decoder Network from Adobe
    or
    Can you just learn contours from input image + segementation as a 4th channel

    https://arxiv.org/abs/1705.03159v1
    https://arxiv.org/abs/1608.00116v1
    https://arxiv.org/abs/1510.08174v3
    https://arxiv.org/abs/1708.07281v1

    This looks simple:
    - run osvos
    - run object detector
    - use https://arxiv.org/abs/1702.05506v1 to extract individual cells
"""

import numpy as np
import cv2

# from skimage.io import imsave
from visualize import draw_centroid

def draw_inst_mask_contour_opencv(image, inst_mask, color=255, thickness=4):
    """
    Find and draw in-place 1-pixel-width contour of an instance mask onto an image.
    Args:
        image: Image on which to paint mask (HxW)
        inst_mask: Instance mask (HxW)
        color: Color of the contour (R,G,B) if RGB image, grey level if Grey image
        thickness: Thickness of the contour
    Returns:
        image: input image with mask contour painted in
    Note:
        https://mmeysenburg.github.io/image-processing/08-contours/
    """
    assert(len(image.shape) == 2 and len(inst_mask.shape) == 2)
    assert(image.shape[0] == inst_mask.shape[0] and image.shape[1] == inst_mask.shape[1])

    # cv2.findContours modifies the input, so use a work copy
    inst_mask_contour = np.copy(inst_mask)

    # Find contours of the mask and only draw the outermost
    _, contours, _ = cv2.findContours(inst_mask_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours=contours, contourIdx=0, color=color, thickness=thickness)

    return image

def inst_masks_contours_to_label(inst_masks, thickness=4):
    """Combine instance mask contours into a semantic label
    Args:
        inst_masks: List or np array of instance masks
    Returns:
        contour_label: semantic label with output mask contours
    """
    # Allocate image to paint mask contours on
    contour_label = np.zeros_like(inst_masks[0])

    for inst_mask in inst_masks:
        draw_inst_mask_contour_opencv(contour_label, inst_mask, thickness=thickness)

    return contour_label

def inst_masks_centroids(inst_masks):
    """Get the centroids of a list of masks
    Args:
        inst_masks: List or np array of instance masks in (#,H,W) format
    Returns:
        centroids: List of instance mask centroids
    """
    # Allocate image to paint mask contours on
    # contour_label = np.zeros_like(inst_masks[0])
    centroids = []
    for inst_mask in inst_masks: # inst_mask (H,W)
        # cv2.findContours modifies the input, so use a work copy
        inst_mask_contour = np.copy(inst_mask)

        # Get the centroid of the outermost contour of the mask
        _, contours, _ = cv2.findContours(inst_mask_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        M = cv2.moments(contours[0])
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        else:
            # Contour is 1-pel wide, probably on the edge of the image
            line = np.reshape(np.asarray(contours[0]), (-1,2))
            cx_min, cx_max = np.amin(line[:,0]), np.amax(line[:,0])
            cy_min, cy_max = np.amin(line[:,1]), np.amax(line[:,1])
            if cx_min == cx_max:
                cx = cx_min
                cy = int((cy_min + cy_max) / 2.)
            else:
                cx = int((cx_min + cx_max) / 2.)
                cy = cy_min
        centroids.append((cx, cy))


    return centroids

def inst_masks_centroids_to_label(inst_masks):
    """Combine instance mask centroids into a semantic label
    Args:
        inst_masks: List or np array of instance masks in (#,H,W) format
    Returns:
        contour_label: semantic label with output mask contours
    """
    # Find the centroids
    centroids = inst_masks_centroids(inst_masks)

    # Allocate image to paint mask contours on
    centroids_label = np.zeros_like(inst_masks[0]) # (H,W)

    draw_inst_mask_centroids(centroids_label, centroids)

    return centroids_label

def draw_inst_mask_centroids(image, centroids, color=255):
    """
    Draw instance mask centroids in-place
    Args:
        image: Image on which to paint centroids (HxW)
        centroids: Lsit of centroids in (cx,cy) format
        color: Grey level of the centroid
    Returns:
        image: input image with mask contour painted in
    Note:
        https://mmeysenburg.github.io/image-processing/08-contours/
    """
    assert(len(image.shape) == 2)

    for centroid in centroids:
        draw_centroid(image, centroid, color, in_place=True)

    return image

# from bboxes import extract_bbox
# from skimage.measure import find_contours
# from dataset import DSB18Dataset, _DEFAULT_DSB18_OPTIONS
# from skimage.io import imsave
# from skimage import img_as_float
# from skimage.segmentation import active_contour
#
# import dataset
#
# def UNUSED_draw_inst_mask_contour_skimage(image, inst_mask, color=255):
#     """
#     Find and draw in-place 1-pixel-width contour of an instance mask onto an image.
#     Args:
#         image: Image on which to paint mask (HxW)
#         inst_mask: Instance mask (HxW)
#         color: Color of the contour (R,G,B) if RGB image, grey level if Grey image
#     Returns:
#         inst_mask_contour: mask contour
#     See:
#         https://stackoverflow.com/questions/45222413/how-to-draw-bounding-boxes-on-the-detected-contours-in-scikit-image
#         https://stackoverflow.com/questions/39642680/create-mask-from-skimage-contour
#         \http://scikit-image.org/docs/dev/auto_examples/edges/plot_active_contours.html
#         http://scikit-image.org/docs/dev/auto_examples/edges/plot_contours.html
#     """
#     assert(len(image.shape) == 2 and len(inst_mask.shape) == 2)
#     assert(image.shape[0] == inst_mask.shape[0] and image.shape[1] == inst_mask.shape[1])
#
#     # Allocate image to paint mask contours on on
#     # inst_mask_contour = np.zeros_like(inst_mask)
#     corners = extract_bbox(inst_mask, order='corners')
#     snake = active_contour(inst_mask, corners, alpha=0.015, beta=10, gamma=0.001)
#     coordinates = snake.astype(int)
#     x_s = coordinates[:,0]
#     y_s = coordinates[:,1]
#     image[coordinates[:, 0], coordinates[:, 1]] = color
#     imsave('c:/temp/temp.png', image)
#     # Find contour of the mask and make sure the coordinates are expressed as integers
#     # toto = img_as_float(inst_mask)
#     # skimage version:  contours = find_contours(img_as_float(inst_mask), 1.0)
#     _, contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     assert(len(contours) == 1)
#     cv2.drawContours(image, contours=contours, contourIdx=0, color=color, thickness=1)
#     imsave('c:/temp/temp.png', image)
#
#
#     coords = np.asarray(contours)
#     coords2 = np.asarray(contours[0])
#     x_s = coords[0, :]
#     y_s = coords[1, :]
#     x2_s = coords2[0, :]
#     y2_s = coords2[1, :]
#
#     # Draw contour
#     image[contours[:, 0], contours[:, 1]] = color
#
#     return image
#
# if __name__ == '__main__':
#
#     # Load dataset (using instance masks)
#     options = dataset._DEFAULT_DSB18_OPTIONS
#     options['mode'] = 'instance_masks'
#     options['in_memory'] = False
#     ds_inst_masks = dataset.DSB18Dataset(phase='train_val', options=options)
#
#     # Display dataset configuration
#     ds_inst_masks.print_config()
#     assert(ds_inst_masks.train_size == 536)
#     assert(ds_inst_masks.val_size == 134)
#
#     # Inspect original dataset (with instance masks)
#     num_samples = 4
#     images, inst_masks, _ = ds_inst_masks.get_rand_samples_with_inst_masks(num_samples, 'train')
#     assert(type(images) is list and type(inst_masks) is list)
#     assert(len(images)==num_samples and len(inst_masks)==num_samples)
#     for image, masks in zip(images, inst_masks):
#         assert (type(masks) is np.ndarray)
#         assert (len(image.shape) == 3 and (image.shape[2]==3 or image.shape[2]==4))
#         contour_label = inst_masks_to_contour_label(masks)
#         # imsave('c:/temp/temp.png', contour_label)
#         assert (type(contour_label) is np.ndarray)
#         assert (len(contour_label.shape) == 2)
#         assert (image.shape[0] == contour_label.shape[0] and image.shape[1] == contour_label.shape[1])
#         assert (contour_label.min() == 0 and contour_label.max() == 255)
#         print("Image [{}], Contours [{}]".format(image.shape, contour_label.shape))
#
