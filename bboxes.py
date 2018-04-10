"""
bboxes.py

Bounding box utility functions.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Based on:
    - https://github.com/matterport/Mask_RCNN/blob/master/utils.py
        Copyright (c) 2017 Matterport, Inc. / Written by Waleed Abdulla
        Licensed under the MIT License

References for future work:
    - https://github.com/tensorflow/models/blob/master/research/object_detection/core/box_list_ops.py
    - https://github.com/tensorflow/models/blob/master/research/object_detection/utils/np_box_ops.py
      https://github.com/tensorflow/models/blob/master/research/object_detection/utils/ops.py
        Copyright 2017 The TensorFlow Authors. All Rights Reserved.
        Licensed under the Apache License, Version 2.0
    - https://github.com/tangyuhao/DAVIS-2016-Chanllege-Solution/blob/master/Step1-SSD/tf_extended/bboxes.py
        https://github.com/tangyuhao/DAVIS-2016-Chanllege-Solution/blob/master/Step1-SSD/bounding_box.py
        Copyright (c) 2017 Paul Balanca / Written by Paul Balanca
        Licensed under the Apache License, Version 2.0, January 2004
"""

import numpy as np

def extract_bbox(mask, order='y1x1y2x2'):
    """Compute bounding box from a mask.
    Param:
        mask: [height, width]. Mask pixels are either >0 or 0.
        order: ['y1x1y2x2' | 'x1y1x2y2' | 'corners']
    Returns:
        bbox numpy array [y1, x1, y2, x2] or tuple x1, y1, x2, y2.
    Based on:
        https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    """
    horizontal_indicies = np.where(np.any(mask, axis=0))[0]
    vertical_indicies = np.where(np.any(mask, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    if order == 'x1y1x2y2':
        return x1, y1, x2, y2
    elif order == 'corners':
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x2, y1]])
    else:
        return np.array([y1, x1, y2, x2]).astype(np.int32)

def extract_bboxes(masks, order='instances_first'):
    """Compute bounding boxes from an array of masks. Mask pixels are either >0 or 0.
    Params
        masks: List or np array of instance masks
        order: Is inst_masks np array in [instance count, H, W] or [H, W, instance count]?
    Returns:
        bbox numpy arrays [instance count, (y1, x1, y2, x2)].
    """
    # Make sure masks are in format np array [num_instances, H, W] or list([H,W])
    if type(masks) is list or order == 'instances_first':
        _masks = masks
    else:
        _masks = np.rollaxis(masks, 0, 3)

    bboxes = [extract_bbox(mask) for mask in _masks]

    return np.asarray(bboxes)

