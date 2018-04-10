"""
losses.py

Loss functions.

Modifications by Phil Ferriere licensed under the MIT License (see LICENSE for details)

Based on:
    - https://github.com/scaelles/OSVOS-TensorFlow/blob/master/osvos_parent_demo.py
        Written by Sergi Caelles (scaelles@vision.ee.ethz.ch)
        This file is part of the OSVOS paper presented in:
        Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
        One-Shot Video Object Segmentation
        CVPR 2017
        MIT code license

    - http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/cost.html
    Apache License 2.0

    - https://github.com/ternaus/TernausNet
    - https://github.com/ternaus/robot-surgery-segmentation
        Copyright (c) 2018 Vladimir Iglovikov, Alexey Shvets, Alexandr A. Kalinin, Alexander Rakhlin
        MIT code license

Refs:
    pretrosgk/Kaggle-Carvana-Image-Masking-Challenge/model/losses.py
    # https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge/blob/master/model/losses.py

    Great set of loss functions implemented in TF:
    @ https://github.com/blei-lab/edward/blob/master/edward/criticisms/evaluate.py

    Another great set of loss functions implemented in TF:
    @ http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/cost.html

    Source code for tensorflow/tensorflow/python/ops/losses/losses_impl.py
    @ https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/losses/losses_impl.py

    Source code for torch.nn.modules.loss
    @ http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html

    Cross Entropy Introduction
    @ http://geek.csdn.net/news/detail/126833

Debugging Notes:
# Training parameters
gpu_id = 0
iter_mean_grad = 1
max_training_iters_3 = 36000 # 300
max_training_iters_4 = 60000 # 500
save_step = 1000 # 50
display_step = 100 # 10
ini_lr = 1e-8
boundaries = [42000, 48000, 54000, 60000] # [100, 150, 250, 300, 400]
values = [ini_lr, ini_lr * 0.5, ini_lr * 0.1, ini_lr * 0.05]
batch_size = 32
2018-04-01 09:55:27 Iter 36100: Train Loss = 21443.70 Train Loss Jaccard = 0.41 Train Dice = 0.87 Val Loss = 27929.89 Val Dice = 0.84
21443.70 * 1e-8 = 0.00021443

# Training parameters
gpu_id = 0
iter_mean_grad = 1
max_training_iters_3 = 36000 # 300
max_training_iters_4 = 60000 # 500
save_step = 1000 # 50
display_step = 100 # 10
ini_lr = 1e-8
boundaries = [42000, 48000, 54000, 60000] # [100, 150, 250, 300, 400]
values = [ini_lr, ini_lr * 0.5, ini_lr * 0.1, ini_lr * 0.05]
batch_size = 32
2018-04-01 09:55:27 Iter 36100: Train Loss = 21443.70 Train Loss Jaccard = 0.41 Train Dice = 0.87 Val Loss = 27929.89 Val Dice = 0.84
0.41 * 5e-4 = 0.00021443

2018-04-02 20:05:29 Iter 36100: Train Loss = 21442.45 Train Loss Jaccard = 0.10 Train Dice = 0.87 Val Loss = 27922.58 Val Dice = 0.84

loss += weight *  tf.log((intersection + smooth) / (union - intersection + smooth))
leads to:
2018-04-02 20:18:57 Iter 36100: Train Loss = 173560.31 Train Loss Jaccard = -0.08 Train Dice = 0.78 Val Loss = 134006.12 Val Dice = 0.83

loss -= weight *  tf.log((intersection + smooth) / (union - intersection + smooth))
leads to:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# TODO Review your implementation against: pretrosgk/Kaggle-Carvana-Image-Masking-Challenge/model/losses.py
# https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge/blob/master/model/losses.py

def class_balanced_cross_entropy_loss(output, label, priors=None):
    """Define the class balanced cross entropy loss to train the network.
    As in DRIU, to train the network, we adopt the class-balancing cross entropy loss function
    originally proposed in Holistically-nested edge detection, by Xie et al. (ICCV 2015) for the
    task of contour detection in natural images. The multiplier beta is used to handle the
    imbalance of the substantially greater number of background compared to foreground pixels
    (which could be vessels or the optic disc in retinal images, or lesions in liver CT or MRI
    images). Below, num_labels_pos and num_labels_neg denote the foreground and background sets
    of the ground truth label, respectively. In this case, the foreground loss is scaled by
    beta=num_labels_neg/(num_labels_pos+num_labels_neg) and the background loss is scaled by
    1-beta=num_labels_neg/(num_labels_pos+num_labels_neg).
    Args:
        output: Raw output of the network (most likely not between 0 and 1))
        label: Ground truth label (between 0 and 1)
        priors: Use pre-defined dataset-wide hardcoded (beta, 1-beta), otherwise compute them
    Returns:
        Tensor that evaluates the loss
    """

    labels = tf.cast(tf.greater(label, 0.5), tf.float32)

    if priors is None:
        num_labels_pos = tf.reduce_sum(labels)
        num_labels_neg = tf.reduce_sum(1.0 - labels)
        num_total = tf.add(num_labels_pos, num_labels_neg)
        beta = tf.div(num_labels_neg, num_total)
        one_minus_beta = tf.div(num_labels_pos, num_total)
    else:
        beta, one_minus_beta = priors

    output_gt_zero = tf.cast(tf.greater_equal(output, 0), tf.float32)
    loss_val = tf.multiply(output, (labels - output_gt_zero)) - tf.log(
        1 + tf.exp(output - 2 * tf.multiply(output, output_gt_zero)))

    loss_pos = tf.reduce_sum(-tf.multiply(labels, loss_val))
    loss_neg = tf.reduce_sum(-tf.multiply(1.0 - labels, loss_val))

    final_loss = tf.add(tf.multiply(beta, loss_pos), tf.multiply(one_minus_beta, loss_neg))

    return final_loss


def class_balanced_cross_entropy_loss_theoretical(output, label):
    """Theoretical version of the class balanced cross entropy loss to train the network .
    This is used as a performance measure, here, not as the loss of the network during training.
    During training, we use `class_balanced_cross_entropy_loss()`.
    Args:
	    output: Raw output of the network (most likely not between 0 and 1))
	    label: Ground truth label (can be between 0 and 1, or 0 and 255)
    Returns:
	    Tensor that evaluates the loss
    Note:
        Per the original DRIU authors, this loss produces unstable results. Hence the use of
        `class_balanced_cross_entropy_loss()` instead.
    """
    output = tf.nn.sigmoid(output)

    labels_pos = tf.cast(tf.greater(label, 0), tf.float32)
    labels_neg = tf.cast(tf.less(label, 1), tf.float32)

    num_labels_pos = tf.reduce_sum(labels_pos)
    num_labels_neg = tf.reduce_sum(labels_neg)
    num_total = num_labels_pos + num_labels_neg

    loss_pos = tf.reduce_sum(tf.multiply(labels_pos, tf.log(output + 0.00001)))
    loss_neg = tf.reduce_sum(tf.multiply(labels_neg, tf.log(1 - output + 0.00001)))

    final_loss = -num_labels_neg / num_total * loss_pos - num_labels_pos / num_total * loss_neg

    return final_loss

def soft_jaccard_binary_loss(output, label, weight=1):
    """
        Original:
        class LossBinary:
        Loss defined as BCE - log(soft_jaccard)
        Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
        Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
        arXiv:1706.06169

        def __init__(self, jaccard_weight=0):
            self.nll_loss = nn.BCEWithLogitsLoss()
            self.jaccard_weight = jaccard_weight

        def __call__(self, outputs, targets):
            loss = self.nll_loss(outputs, targets)

            if self.jaccard_weight:
                eps = 1e-15
                jaccard_target = (targets == 1).float()
                jaccard_output = F.sigmoid(outputs)

                intersection = (jaccard_output * jaccard_target).sum()
                union = jaccard_output.sum() + jaccard_target.sum()

                loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
            return loss
        or,
        class Loss:
            def __init__(self, dice_weight=1):
                self.nll_loss = nn.BCELoss()
                self.dice_weight = dice_weight

            def __call__(self, outputs, targets):
                loss = self.nll_loss(outputs, targets)
                if self.dice_weight:
                    eps = 1e-15
                    dice_target = (targets == 1).float()
                    dice_output = outputs
                    intersection = (dice_output * dice_target).sum()
                    union = dice_output.sum() + dice_target.sum() + eps

                    loss -= torch.log(2 * intersection / union)

                return loss
    """
    label = tf.cast(tf.greater(label, 0), tf.float32)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=label))

    if weight > 0:
        output = tf.nn.sigmoid(output)
        smooth = 1e-15
        intersection = tf.reduce_sum(output * label)
        union = tf.reduce_sum(output) + tf.reduce_sum(label)
        loss -= weight *  tf.log((intersection + smooth) / (union - intersection + smooth))

    return loss


def soft_jaccard_multiclass_loss(output, label):
    """
        Original:
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
        if class_weights is not None:
            nll_weight = utils.cuda(
                torch.from_numpy(class_weights.astype(np.float32)))
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss2d(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes=num_classes

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        if self.jaccard_weight:
            cls_weight = self.jaccard_weight / self.num_classes
            eps = 1e-15
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum() + eps
                loss += (1 - intersection / (union - intersection)) * cls_weight

            loss /= (1 + self.jaccard_weight)
        return loss
    """
    return NotImplemented
