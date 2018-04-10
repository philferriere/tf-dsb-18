"""
visualize.py

Performance measure helpers.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def dice_coef_theoretical_metric(y_pred, y_true):
    """Define the dice coefficient
    Args:
        y_pred: Raw output of the network (most likely not between 0 and 1))
        y_true: Ground truth label (can be between 0 and 1, or 0 and 255)
    Returns:
        Dice coefficient
        """

    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)

    y_pred_f = tf.nn.sigmoid(y_pred)
    y_pred_f = tf.cast(tf.greater(y_pred_f, 0.5), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred_f, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    dice = (2. * intersection) / (union + 0.00001)

    if (tf.reduce_sum(y_pred) == 0) and (tf.reduce_sum(y_true) == 0):
        dice = 1

    return dice


def iou_scores(gt_inst_masks, pred_inst_masks, order='instances_first'):
    """Compute iou score for each predicted instance mask.
        Args:
            gt_inst_masks: np array in [H, W, count1] or [count1, H, W] format.
            pred_inst_masks: np array in [H, W, count2] or [count2, H, W] format.
            order: Are instance masks in [instance count, H, W] or [H, W, instance count] format?
        Returns:
            iou: a list of iou scores, one per predicted mask
        Based on:
            https://www.kaggle.com/glenslade/alternative-metrics-kernel
    """
    # Make sure masks are in format np array [num_instances, H, W] or list([H,W])
    if type(gt_inst_masks) is list:
        gt_masks = np.asarray(gt_inst_masks)
    else:
        if order != 'instances_first':
            gt_masks = np.rollaxis(gt_inst_masks, 0, 3)
        else:
            gt_masks = gt_inst_masks

    if type(pred_inst_masks) is list:
        pred_masks = np.asarray(pred_inst_masks)
    else:
        if order != 'instances_first':
            pred_masks = np.rollaxis(pred_inst_masks, 0, 3)
        else:
            pred_masks = pred_inst_masks

    gt_masks = np.where(gt_masks > 0, 1., 0.).astype(np.float)
    pred_masks = np.where(pred_masks > 0, 1., 0.).astype(np.float)

    iou = []
    for pred_mask in pred_masks:
        bol = 0  # best overlap
        bun = 1e-9  # corresponding best union
        for gt_mask in gt_masks:
            olap = pred_mask * gt_mask  # Intersection points
            osz = np.sum(olap)  # Add the intersection points to see size of overlap
            if osz > bol:  # Choose the match with the biggest overlap
                bol = osz
                bun = np.sum(np.maximum(pred_mask, gt_mask))  # Union formed with sum of maxima
        iou.append(bol / bun)

    return iou

def average_precision(gt_inst_masks, pred_inst_masks, order='instances_first', verbose=False):
    """Compute the average precision at a pre-defined set of thresholds between groundtruth and predicted masks.
        Args:
            gt_inst_masks: np array in [H, W, count1] or [count1, H, W] format.
            pred_inst_masks: np array in [H, W, count2] or [count2, H, W] format.
            order: Are instance masks in [instance count, H, W] or [H, W, instance count] format?
            verbose: Print results?
        Returns:
            ap: average precision over pre-defined thresholds
        Based on:
            https://www.kaggle.com/glenslade/alternative-metrics-kernel
    """
    # Make sure masks are in format np array [num_instances, H, W] or list([H,W])
    if type(gt_inst_masks) is list and type(pred_inst_masks) is list or order == 'instances_first':
        gt_masks, pred_masks = gt_inst_masks, pred_inst_masks
    else:
        gt_masks, pred_masks = np.rollaxis(gt_inst_masks, 0, 3), np.rollaxis(pred_inst_masks, 0, 3)
    if len(gt_masks.shape) == 4 and gt_masks.shape[3] == 1:
        gt_masks = np.squeeze(gt_masks, -1)
    if len(pred_masks.shape) == 4 and pred_masks.shape[3] == 1:
        pred_masks = np.squeeze(pred_masks, -1)

    # Compute iou score for each predicted instance mask
    iou = iou_scores(gt_masks, pred_masks, order='instances_first')

    # Loop over IoU thresholds
    gt_count, pred_count = len(gt_inst_masks), len(pred_inst_masks)
    p = 0
    if verbose:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        matches = iou > t
        tp = np.count_nonzero(matches)  # True positives
        fp = pred_count - tp  # False positives
        fn = gt_count - tp  # False negatives
        p += tp / (tp + fp + fn)
        if verbose:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, tp / (tp + fp + fn)))
    if verbose:
        print("AP\t-\t-\t-\t{:1.3f}".format(p / 10))

    return p / 10

