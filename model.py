"""
model.py

Segmentation backbone networks.

Modifications by Phil Ferriere licensed under the MIT License (see LICENSE for details)

Based on:
  - https://github.com/scaelles/OSVOS-TensorFlow/blob/master/osvos_parent_demo.py
    Written by Sergi Caelles (scaelles@vision.ee.ethz.ch)
    This file is part of the OSVOS paper presented in:
      Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
      One-Shot Video Object Segmentation
      CVPR 2017
    Unknown code license

References for future work:
    https://github.com/scaelles/OSVOS-TensorFlow
    http://localhost:8889/notebooks/models-master/research/slim/slim_walkthrough.ipynb
    https://github.com/bryanyzhu/two-stream-pytorch
    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
    https://github.com/kwotsin/TensorFlow-ENet/blob/master/predict_segmentation.py
    https://github.com/fperazzi/davis-2017/tree/master/python/lib/davis/measures
    https://github.com/suyogduttjain/fusionseg
    https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, warnings
import numpy as np
import time
from skimage.io import imsave

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils
slim = tf.contrib.slim

from tqdm import trange

from adapt import adapt_image, adapt_label, restore_label
from losses import class_balanced_cross_entropy_loss, class_balanced_cross_entropy_loss_theoretical
from losses import soft_jaccard_binary_loss
from metrics import dice_coef_theoretical_metric, average_precision
from segment import sem_label_to_inst_masks, save_inst_masks, save_sem_label
from logger import TBLogger

_DEBUG_PROCESSING_STEPS = False

# Tuned hyperparams
_TUNED_TRAINING_ITER = 37500
# Semantic labels
# 224 thresh -> Dice score: 0.9021668403006312 (average of 134 labels)
# 233 thresh -> Dice score: 0.9032469725430902 (average of 134 labels)
# 234 thresh -> Dice score: 0.9032926999810916 (average of 134 labels) *
# 235 thresh -> Dice score: 0.9032519633200631 (average of 134 labels)
# 236 thresh -> Dice score: 0.9030610736626298 (average of 134 labels)
# 240 thresh -> Dice score: 0.9013736430388778 (average of 134 labels)
# 244 threash -> Dice score: 0.8975354664361299 (average of 134 labels)
# Semantic contours:
#  44 thresh -> Dice score: 0.7298585837011906 (average of 134 labels)
#  50 thresh -> Dice score: 0.7301103330370206 (average of 134 labels)
#  52 thresh -> Dice score: 0.7301515638828278 (average of 134 labels) *
#  53 thresh -> Dice score: 0.730105691214106  (average of 134 labels)
#  54 thresh -> Dice score: 0.7301123611517807 (average of 134 labels)
#  60 thresh -> Dice score: 0.7299897510613969 (average of 134 labels)
#  64 thresh -> Dice score: 0.7299571780126486 (average of 134 labels)
#  84 thresh -> Dice score: 0.7292558499681416 (average of 134 labels)
#  94 thresh -> Dice score: 0.7285335673325097 (average of 134 labels)
# 104 thresh -> Dice score: 0.7278623607621264 (average of 134 labels)
# 114 thresh -> Dice score: 0.7270165692959258 (average of 134 labels)
# 124 thresh -> Dice score: 0.7259050785605587 (average of 134 labels)
# 134 thresh -> Dice score: 0.7247679598295866 (average of 134 labels)
# 144 thresh -> Dice score: 0.7235300162834908 (average of 134 labels)
# 154 thresh -> Dice score: 0.7221686728854677 (average of 134 labels)
# 164 thresh -> Dice score: 0.7207021099417957 (average of 134 labels)
# 174 thresh -> Dice score: 0.7189529620444597 (average of 134 labels)
# 184 thresh -> Dice score: 0.7168942334047005 (average of 134 labels)
# 194 thresh -> Dice score: 0.714408086751824 (average of 134 labels)
# 204 thresh -> Dice score: 0.711150137123777 (average of 134 labels)
# 214 thresh -> Dice score: 0.7066310757576529 (average of 134 labels)
# 224 thresh -> Dice score: 0.7001429144571076 (average of 134 labels)
# 234 thresh -> Dice score: 0.6897677438917444 (average of 134 labels)
# 244 thresh -> Dice score: 0.669176440296778 (average of 134 labels)

_DEFAULT_MODEL_TRAIN_OPTIONS = {
    'unique_img_size': None, # (None, 256),  # [None | (IMAGE_MIN_DIM, IMAGE_MAX_DIM)]
    'loss': 'class_balanced_cross_entropy_loss' # ['class_balanced_cross_entropy_loss' | 'soft_jaccard_binary_loss']
    }

_DEFAULT_MODEL_VAL_TEST_OPTIONS = {
    'unique_img_size': None,  # [None | (IMAGE_MIN_DIM, IMAGE_MAX_DIM)]
    'loss': 'class_balanced_cross_entropy_loss' # ['class_balanced_cross_entropy_loss' | 'soft_jaccard_binary_loss']
    }

# For backward compatibility only
_DEFAULT_MODEL_OPTIONS = _DEFAULT_MODEL_TRAIN_OPTIONS

class Model(object):
    """Model class.
    """

    def __init__(self, ds, phase='train', options=_DEFAULT_MODEL_TRAIN_OPTIONS):
        """Initialize the Model object
        Args:
            ds: Dataset to use with this model
            phase: Possible options: 'train', 'val', 'test'
            options: see below
        Options:
            key: Purpose
        """
        self._ds = ds
        self.options = options
        self._phase = phase

        # Setup loss function
        assert(options['loss'] in ['class_balanced_cross_entropy_loss', 'soft_jaccard_binary_loss'])
        # if options['loss'] == class_balanced_cross_entropy_loss:
        #     self._loss_fn = class_balanced_cross_entropy_loss
        # else:
        #     self._loss_fn = soft_jaccard_binary_loss

    def _backbone_arg_scope(self, weight_decay=0.0002):
        """Defines the network's arg scope.
        Args:
            weight_decay: The l2 regularization coefficient.
        Returns:
            An arg_scope.
        """
        with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.random_normal_initializer(stddev=0.001),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_initializer=tf.zeros_initializer(),
                            biases_regularizer=None,
                            padding='SAME') as arg_sc:
            return arg_sc


    def _crop_features(self, feature, out_size):
        """Crop the center of a feature map
        This is necessary when large upsampling results in a (width x height) size larger than the original input.
        Args:
            feature: Feature map to crop
            out_size: Size of the output feature map
        Returns:
            Tensor that performs the cropping
        """
        up_size = tf.shape(feature)
        ini_w = tf.div(tf.subtract(up_size[1], out_size[1]), 2)
        ini_h = tf.div(tf.subtract(up_size[2], out_size[2]), 2)
        slice_input = tf.slice(feature, (0, ini_w, ini_h, 0), (-1, out_size[1], out_size[2], -1))
        return tf.reshape(slice_input, [int(feature.get_shape()[0]), out_size[1], out_size[2], int(feature.get_shape()[3])])


    def _backbone(self, inputs, backbone_name='dsb18'):
        """Defines the backbone network (same as the OSVOS network, with variation in input size)
        Args:
            inputs: Tensorflow placeholder that contains the input image (either 3 or 4 channels)
            backbone_name: Name of the convnet backbone
        Returns:
            net: Output Tensor of the network
            end_points: Dictionary with all Tensors of the network
        Reminder:
            This is how a VGG16 network looks like:
    
            Layer (type)                     Output Shape          Param #     Connected to
            ====================================================================================================
            input_1 (InputLayer)             (None, 480, 854, 3)   0
            ____________________________________________________________________________________________________
            block1_conv1 (Convolution2D)     (None, 480, 854, 64)  1792        input_1[0][0]
            ____________________________________________________________________________________________________
            block1_conv2 (Convolution2D)     (None, 480, 854, 64)  36928       block1_conv1[0][0]
            ____________________________________________________________________________________________________
            block1_pool (MaxPooling2D)       (None, 240, 427, 64)  0           block1_conv2[0][0]
            ____________________________________________________________________________________________________
            block2_conv1 (Convolution2D)     (None, 240, 427, 128) 73856       block1_pool[0][0]
            ____________________________________________________________________________________________________
            block2_conv2 (Convolution2D)     (None, 240, 427, 128) 147584      block2_conv1[0][0]
            ____________________________________________________________________________________________________
            block2_pool (MaxPooling2D)       (None, 120, 214, 128) 0           block2_conv2[0][0]
            ____________________________________________________________________________________________________
            block3_conv1 (Convolution2D)     (None, 120, 214, 256) 295168      block2_pool[0][0]
            ____________________________________________________________________________________________________
            block3_conv2 (Convolution2D)     (None, 120, 214, 256) 590080      block3_conv1[0][0]
            ____________________________________________________________________________________________________
            block3_conv3 (Convolution2D)     (None, 120, 214, 256) 590080      block3_conv2[0][0]
            ____________________________________________________________________________________________________
            block3_conv4 (Convolution2D)     (None, 120, 214, 256) 590080      block3_conv3[0][0]
            ____________________________________________________________________________________________________
            block3_pool (MaxPooling2D)       (None, 60, 107, 256)  0           block3_conv4[0][0]
            ____________________________________________________________________________________________________
            block4_conv1 (Convolution2D)     (None, 60, 107, 512)  1180160     block3_pool[0][0]
            ____________________________________________________________________________________________________
            block4_conv2 (Convolution2D)     (None, 60, 107, 512)  2359808     block4_conv1[0][0]
            ____________________________________________________________________________________________________
            block4_conv3 (Convolution2D)     (None, 60, 107, 512)  2359808     block4_conv2[0][0]
            ____________________________________________________________________________________________________
            block4_conv4 (Convolution2D)     (None, 60, 107, 512)  2359808     block4_conv3[0][0]
            ____________________________________________________________________________________________________
            block4_pool (MaxPooling2D)       (None, 30, 54, 512)   0           block4_conv4[0][0]
            ____________________________________________________________________________________________________
            block5_conv1 (Convolution2D)     (None, 30, 54, 512)   2359808     block4_pool[0][0]
            ____________________________________________________________________________________________________
            block5_conv2 (Convolution2D)     (None, 30, 54, 512)   2359808     block5_conv1[0][0]
            ____________________________________________________________________________________________________
            block5_conv3 (Convolution2D)     (None, 30, 54, 512)   2359808     block5_conv2[0][0]
            ____________________________________________________________________________________________________
            block5_conv4 (Convolution2D)     (None, 30, 54, 512)   2359808     block5_conv3[0][0]
            ____________________________________________________________________________________________________
            block5_pool (MaxPooling2D)       (None, 15, 27, 512)   0           block5_conv4[0][0]
            ____________________________________________________________________________________________________
            flatten (Flatten)                (None, 207360)        0           block5_pool[0][0]
            ____________________________________________________________________________________________________
            fc1 (Dense)                      (None, 4096)          xxx         flatten[0][0]
            ____________________________________________________________________________________________________
            fc2 (Dense)                      (None, 4096)          yyy         fc1[0][0]
            ____________________________________________________________________________________________________
            predictions (Dense)              (None, 1000)          zzz         fc2[0][0]
            ====================================================================================================
        Original Code:
            ETH Zurich
        """
        im_size = tf.shape(inputs)
    
        with tf.variable_scope(backbone_name, backbone_name, [inputs]) as sc:
            end_points_collection = sc.name + '_end_points'
            # Collect outputs of all intermediate layers.
            # Make sure convolution and max-pooling layers use SAME padding by default
            # Also, group all end points in the same container/collection
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='SAME',
                                outputs_collections=end_points_collection):
    
                # VGG16 stage 1 has 2 convolution blocks followed by max-pooling
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
    
                # VGG16 stage 2 has 2 convolution blocks followed by max-pooling
                net_2 = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net_2, [2, 2], scope='pool2')
    
                # VGG16 stage 3 has 3 convolution blocks followed by max-pooling
                net_3 = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net_3, [2, 2], scope='pool3')
    
                # VGG16 stage 4 has 3 convolution blocks followed by max-pooling
                net_4 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net_4, [2, 2], scope='pool4')
    
                # VGG16 stage 5 has 3 convolution blocks...
                net_5 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                # ...but here, it is not followed by max-pooling, as in the original VGG16 architecture.
    
                # This is where the specialization of the VGG network takes place, as described in DRIU and
                # OSVOS-S. The idea is to extract *side feature maps* and design *specialized layers* to perform
                # *deep supervision* targeted at a different task (here, segmentation) than the one used to
                # train the base network originally (i.e., large-scale natural image classification).
    
                # As explained in DRIU, each specialized side output produces feature maps in 16 different channels,
                # which are resized to the original image size and concatenated, creating a volume of fine-to-coarse
                # feature maps. one last convolutional layer linearly combines the feature maps from the volume
                # created by the specialized side outputs into a regressed result.  The convolutional layers employ
                # 3 x 3 convolutional filters for efficiency, except the ones used for linearly combining the outputs
                # (1 x 1 filters).
    
                with slim.arg_scope([slim.conv2d], activation_fn=None):
    
                    # Convolve last layer of stage 2 (before max-pooling) -> side_2 (None, 240, 427, 16)
                    side_2 = slim.conv2d(net_2, 16, [3, 3], scope='conv2_2_16')
    
                    # Convolve last layer of stage 3 (before max-pooling) -> side_3 (None, 120, 214, 16)
                    side_3 = slim.conv2d(net_3, 16, [3, 3], scope='conv3_3_16')
    
                    # Convolve last layer of stage 4 (before max-pooling) -> side_3 (None, 60, 117, 16)
                    side_4 = slim.conv2d(net_4, 16, [3, 3], scope='conv4_3_16')
    
                    # Convolve last layer of stage 3 (before max-pooling) -> side_3 (None, 30, 54, 16)
                    side_5 = slim.conv2d(net_5, 16, [3, 3], scope='conv5_3_16')
    
                    # The _S layears are the side output that will be used for deep supervision
    
                    # Dim reduction - linearly combine side_2 feature maps -> side_2_s (None, 240, 427, 1)
                    side_2_s = slim.conv2d(side_2, 1, [1, 1], scope='score-dsn_2')
    
                    # Dim reduction - linearly combine side_3 feature maps -> side_3_s (None, 120, 214, 1)
                    side_3_s = slim.conv2d(side_3, 1, [1, 1], scope='score-dsn_3')
    
                    # Dim reduction - linearly combine side_4 feature maps -> side_4_s (None, 60, 117, 1)
                    side_4_s = slim.conv2d(side_4, 1, [1, 1], scope='score-dsn_4')
    
                    # Dim reduction - linearly combine side_5 feature maps -> side_5_s (None, 30, 54, 1)
                    side_5_s = slim.conv2d(side_5, 1, [1, 1], scope='score-dsn_5')
    
                    # As repeated in OSVOS-S, upscaling operations take place wherever necessary, and feature
                    # maps from the separate paths are concatenated to construct a volume with information from
                    # different levels of detail. We linearly fuse the feature maps to a single output which has
                    # the same dimensions as the input image.
                    with slim.arg_scope([slim.convolution2d_transpose],
                                        activation_fn=None, biases_initializer=None, padding='VALID',
                                        outputs_collections=end_points_collection, trainable=False):
    
                        # Upsample the side outputs for deep supervision and center-cop them to the same size as
                        # the input. Note that this is straight upsampling (we're not trying to learn upsampling
                        # filters), hence the trainable=False param.
    
                        # Upsample side_2_s (None, 240, 427, 1) -> (None, 480, 854, 1)
                        # Center-crop (None, 480, 854, 1) to original image size (None, 480, 854, 1)
                        side_2_s = slim.convolution2d_transpose(side_2_s, 1, 4, 2, scope='score-dsn_2-up')
                        side_2_s = self._crop_features(side_2_s, im_size)
                        utils.collect_named_outputs(end_points_collection, backbone_name + '/score-dsn_2-cr', side_2_s)
    
                        # Upsample side_3_s (None, 120, 214, 1) -> (None, 484, 860, 1)
                        # Center-crop (None, 484, 860, 1) to original image size (None, 480, 854, 1)
                        side_3_s = slim.convolution2d_transpose(side_3_s, 1, 8, 4, scope='score-dsn_3-up')
                        side_3_s = self._crop_features(side_3_s, im_size)
                        utils.collect_named_outputs(end_points_collection, backbone_name + '/score-dsn_3-cr', side_3_s)
    
                        # Upsample side_4_s (None, 60, 117, 1) -> (None, 488, 864, 1)
                        # Center-crop (None, 488, 864, 1) to original image size (None, 480, 854, 1)
                        side_4_s = slim.convolution2d_transpose(side_4_s, 1, 16, 8, scope='score-dsn_4-up')
                        side_4_s = self._crop_features(side_4_s, im_size)
                        utils.collect_named_outputs(end_points_collection, backbone_name + '/score-dsn_4-cr', side_4_s)
    
                        # Upsample side_5_s (None, 30, 54, 1) -> (None, 496, 880, 1)
                        # Center-crop (None, 496, 880, 1) to original image size (None, 480, 854, 1)
                        side_5_s = slim.convolution2d_transpose(side_5_s, 1, 32, 16, scope='score-dsn_5-up')
                        side_5_s = self._crop_features(side_5_s, im_size)
                        utils.collect_named_outputs(end_points_collection, backbone_name + '/score-dsn_5-cr', side_5_s)
    
                        # Upsample the main outputs and center-crop them to the same size as the input
                        # Note that this is straight upsampling (we're not trying to learn upsampling filters),
                        # hence the trainable=False param. Then, concatenate them in a big volume of fine-to-coarse
                        # feature maps of the same size.
    
                        # Upsample side_2 (None, 240, 427, 16) -> side_2_f (None, 480, 854, 16)
                        # Center-crop (None, 480, 854, 16) to original image size (None, 480, 854, 16)
                        side_2_f = slim.convolution2d_transpose(side_2, 16, 4, 2, scope='score-multi2-up')
                        side_2_f = self._crop_features(side_2_f, im_size)
                        utils.collect_named_outputs(end_points_collection, backbone_name + '/side-multi2-cr', side_2_f)
    
                        # Upsample side_2 (None, 120, 214, 16) -> side_2_f (None, 488, 864, 16)
                        # Center-crop (None, 488, 864, 16) to original image size (None, 480, 854, 16)
                        side_3_f = slim.convolution2d_transpose(side_3, 16, 8, 4, scope='score-multi3-up')
                        side_3_f = self._crop_features(side_3_f, im_size)
                        utils.collect_named_outputs(end_points_collection, backbone_name + '/side-multi3-cr', side_3_f)
    
                        # Upsample side_2 (None, 60, 117, 16) -> side_2_f (None, 488, 864, 16)
                        # Center-crop (None, 488, 864, 16) to original image size (None, 480, 854, 16)
                        side_4_f = slim.convolution2d_transpose(side_4, 16, 16, 8, scope='score-multi4-up')
                        side_4_f = self._crop_features(side_4_f, im_size)
                        utils.collect_named_outputs(end_points_collection, backbone_name + '/side-multi4-cr', side_4_f)
    
                        # Upsample side_2 (None, 30, 54, 16) -> side_2_f (None, 496, 880, 16)
                        # Center-crop (None, 496, 880, 16) to original image size (None, 480, 854, 16)
                        side_5_f = slim.convolution2d_transpose(side_5, 16, 32, 16, scope='score-multi5-up')
                        side_5_f = self._crop_features(side_5_f, im_size)
                        utils.collect_named_outputs(end_points_collection, backbone_name + '/side-multi5-cr', side_5_f)
    
                    # Build the main volume concat_side (None, 496, 880, 16x4)
                    concat_side = tf.concat([side_2_f, side_3_f, side_4_f, side_5_f], axis=3)
    
                    # Dim reduction - linearly combine concat_side feature maps -> (None, 496, 880, 1)
                    net = slim.conv2d(concat_side, 1, [1, 1], scope='upscore-fuse')
    
                    # Note that the FC layers of the original VGG16 network are not part of the DRIU architecture
    
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return net, end_points


    def _upsample_filt(self, size):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


    def _interp_surgery(self, variables):
        """ Set parameters so that deconvolutional layers compute bilinear interpolation
        N.B. this is for deconvolution without groups
        """
        interp_tensors = []
        for v in variables:
            if '-up' in v.name:
                h, w, k, m = v.get_shape()
                tmp = np.zeros((m, k, h, w))
                if m != k:
                    raise ValueError('input + output channels need to be the same')
                if h != w:
                    raise ValueError('filters need to be square')
                up_filter = self._upsample_filt(int(h))
                tmp[range(m), range(k), :, :] = up_filter
                interp_tensors.append(tf.assign(v, tmp.transpose((2, 3, 1, 0)), validate_shape=True, use_locking=True))
        return interp_tensors


    def _preprocess_inputs(self, inputs):
        """Preprocess the inputs to adapt them to the network requirements
        Args:
            inputs: images we want to input to the network in (batch_size,W,H,3) or (batch_size,W,H,4) np array
        Returns:
            Images ready to input to the network in (batch_size,W,H,3) or (batch_size,W,H,4) np array with means substracted
        """
        assert(type(inputs) is np.ndarray or type(inputs) is list)
        if type(inputs) is np.ndarray:
            assert(len(inputs.shape) == 4)
            assert(inputs.shape[3] == 3 or inputs.shape[3] == 4)
            num_channels = inputs.shape[3]
        else:
            assert(len(inputs[0].shape) == 3)
            assert(inputs[0].shape[2] == 3 or inputs[0].shape[2] == 4)
            num_channels = inputs[0].shape[2]

        # Channel averages for normalization
        if num_channels == 4:
            input_mean = np.array((104.00699, 116.66877, 122.67892, 128.), dtype=np.float32)
        else:
            input_mean = np.array((104.00699, 116.66877, 122.67892), dtype=np.float32)

        if self.options['unique_img_size']:
            adapted_inputs, resize_info = [], []
            for input in inputs:
                # Remove mean
                adapted_input = np.subtract(input.astype(np.float32), input_mean)
                image_shape = adapted_input.shape
                # Resize, if requested
                min_dim, max_dim = self.options['unique_img_size']
                adapted_input, window, scale, padding = adapt_image(adapted_input, min_dim, max_dim, True)
                if _DEBUG_PROCESSING_STEPS:
                    print("\nImage:\n  Original: shape={}\n  Adapted: shape={}, window={}, scale={}, padding={}".format(image_shape, adapted_input.shape, window, scale, padding))
                # mask = resize_mask(mask, scale, padding)
                adapted_inputs.append(adapted_input)
                # Save resize info
                resize_info.append((image_shape, window, scale, padding))
            adapted_inputs = np.asarray(adapted_inputs)
        else:
            # Remove mean
            if type(inputs) is np.ndarray:
                adapted_inputs = np.subtract(inputs.astype(np.float32), input_mean)
            else:
                adapted_inputs = []
                input_shape = inputs[0].shape
                unique_size = True
                for input in inputs:
                    if input.shape != input_shape:
                        unique_size = False
                        break
                if unique_size:
                    adapted_inputs = np.asarray(inputs)
                    adapted_inputs = np.subtract(adapted_inputs.astype(np.float32), input_mean)
                else:
                    for input in inputs:
                        adapted_input = np.subtract(input.astype(np.float32), input_mean)
                        adapted_inputs.append(adapted_input)
            resize_info = None

        return adapted_inputs, resize_info
    
    
    def _preprocess_labels(self, labels, packed_adapt_info=None):
        """Preprocess the labels to adapt them to the loss computation requirements.
        Rescale labels between 0 and 1, each pixel value is either 0 or 1.
        Args:
            labels: (batch_size,W,H) or (batch_size,W,H,1) in numpy array
            packed_adapt_info: (batch_size,resize_info_structs) in numpy array
        Returns:
            Label(s) ready to compute the loss (batch_size,W,H,1)
        """
        assert(type(labels) is np.ndarray or type(labels) is list)
        if type(labels) is np.ndarray:
            assert(len(labels.shape) == 4)
            assert(labels.shape[3] == 1)
        else:
            assert(len(labels[0].shape) == 3)
            assert(labels[0].shape[2] == 1)

        # Make sure the labels are binary masks [0..1] in this specific two-class classification problem
        if type(labels) is np.ndarray:
            max_mask = np.max(labels) * 0.5
            labels = np.greater(labels, max_mask).astype(np.float32)
        else:
            for n, label in enumerate(labels):
                max_mask = np.max(label) * 0.5
                labels[n] = np.greater(label, max_mask).astype(np.float32)

        if self.options['unique_img_size']:
            adapted_labels = []
            for label, adapt_info in zip(labels, packed_adapt_info):
                _, _, scale, padding = adapt_info
                adapted_label = adapt_label(label, scale, padding)
                adapted_labels.append(adapted_label)
                if _DEBUG_PROCESSING_STEPS:
                    print("Label:\n  Original: shape={}\n  Adapted: shape={}".format(label.shape, adapted_label.shape))
            labels = np.asarray(adapted_labels)

        if type(labels) is np.ndarray:
            if len(labels.shape) == 3:
                labels = np.expand_dims(labels, axis=-1)
        else:
            label_shape = labels[0].shape
            unique_size = True
            for label in labels:
                if label.shape != label_shape:
                    unique_size = False
                    break
            if unique_size:
                labels = np.asarray(labels)
                if len(labels.shape) == 3:
                    labels = np.expand_dims(labels, axis=-1)
            else:
                for n, label in enumerate(labels):
                    if len(label.shape) == 2:
                        labels[n] = np.expand_dims(label, axis=-1)

        return labels

    def _postprocess_labels(self, labels, packed_adapt_info=None, return_unthresholded=False):
        """Postprocess the predicted labels coming from the network.
        Labels are rescaled, if necessary, and then thresholded to have a value between 0...255.
        Args:
            labels: (batch_size,W,H,1) in numpy array
            packed_adapt_info: (batch_size,resize_info_structs) in numpy array
            return_unthresholded: If True, also return the unthresholded version of the restored label
        Returns:
            Thresholded label(s) with the same size as the original image,
            along with unthresholded label(s) if `return_unthresholded` is True
        """
        assert(type(labels) is np.ndarray or type(labels) is list)
        if type(labels) is np.ndarray:
            assert(len(labels.shape) == 4)
            assert(labels.shape[3] == 1)
        else:
            assert(len(labels[0].shape) == 3)
            assert(labels[0].shape[3] == 1)


        if self.options['unique_img_size']:
            restored_labels = []
            if return_unthresholded:
                unthresholded_labels = []
            for label, adapt_info in zip(labels, packed_adapt_info):
                restored_shape, window, scale, padding = adapt_info
                unthresholded_label = restore_label(label, restored_shape, window, scale, padding)
                restored_label = np.where(unthresholded_label.astype(np.float32) < self._ds.bin_threshold / 255.0, 0, 255).astype('uint8')
                restored_labels.append(restored_label)
                if return_unthresholded:
                    unthresholded_labels.append(unthresholded_label)
                if _DEBUG_PROCESSING_STEPS:
                    print("Label:\n  Adapted: shape={}, type={}\n  Restored: shape={}, type={}".format(label.shape, label.dtype, restored_label.shape, restored_label.dtype))
        else:
            restored_labels = np.where(labels.astype(np.float32) < self._ds.bin_threshold / 255.0, 0, 255).astype('uint8')
            unthresholded_labels = labels

        if return_unthresholded:
            return restored_labels, unthresholded_labels
        else:
            return restored_labels

    def _load_pre_trained(self, ckpt_path, parent_name, ckpt_name):
        """Initialize the network parameters using pre-trained params of the parent model
        Args:
            ckpt_path: Path to the checkpoint, either the 3-channel or 4-channel input version
            parent_name: Name of the parent convnet backbone
            ckpt_name: Name of the child convnet backbone
        Returns:
            Function that takes a session and initializes the network
        """
        reader = tf.train.NewCheckpointReader(ckpt_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
    
        # for k, v in var_to_shape_map.items():
        #     print(' var_to_shape_map:  name = {}, shape = {}'.format(k, v))
    
        vars_corresp = dict()
    
        # for v in slim.get_model_variables():
        #     print(' slim.get_model_variables()  name = {}, shape = {}'.format(v.name, v.get_shape()))
    
        for v in var_to_shape_map:
            if "conv" in v and "Momentum" not in v:
                vars_corresp[v] = slim.get_model_variables(v.replace(parent_name, ckpt_name))[0]
    
        # for k, v in vars_corresp.items():
        #     print(' vars_corresp:  name = {} to name = {}, shape = {}'.format(k, v.name, v.get_shape()))
    
        init_fn = slim.assign_from_checkpoint_fn(ckpt_path, vars_corresp)
        return init_fn


    def _parameter_lr(self, backbone_name):
        """Specify the relative learning rate for every parameter. The final learning rate
        in every parameter will be the one defined here multiplied by the global one
        Args:
            backbone_name: Name of the convnet backbone
        Returns:
            Dictionary with the relative learning rate for every parameter
        """

        vars_corresp = dict()
        vars_corresp[backbone_name + '/conv1/conv1_1/weights'] = 1
        vars_corresp[backbone_name + '/conv1/conv1_1/biases'] = 2
        vars_corresp[backbone_name + '/conv1/conv1_2/weights'] = 1
        vars_corresp[backbone_name + '/conv1/conv1_2/biases'] = 2

        vars_corresp[backbone_name + '/conv2/conv2_1/weights'] = 1
        vars_corresp[backbone_name + '/conv2/conv2_1/biases'] = 2
        vars_corresp[backbone_name + '/conv2/conv2_2/weights'] = 1
        vars_corresp[backbone_name + '/conv2/conv2_2/biases'] = 2

        vars_corresp[backbone_name + '/conv3/conv3_1/weights'] = 1
        vars_corresp[backbone_name + '/conv3/conv3_1/biases'] = 2
        vars_corresp[backbone_name + '/conv3/conv3_2/weights'] = 1
        vars_corresp[backbone_name + '/conv3/conv3_2/biases'] = 2
        vars_corresp[backbone_name + '/conv3/conv3_3/weights'] = 1
        vars_corresp[backbone_name + '/conv3/conv3_3/biases'] = 2

        vars_corresp[backbone_name + '/conv4/conv4_1/weights'] = 1
        vars_corresp[backbone_name + '/conv4/conv4_1/biases'] = 2
        vars_corresp[backbone_name + '/conv4/conv4_2/weights'] = 1
        vars_corresp[backbone_name + '/conv4/conv4_2/biases'] = 2
        vars_corresp[backbone_name + '/conv4/conv4_3/weights'] = 1
        vars_corresp[backbone_name + '/conv4/conv4_3/biases'] = 2

        vars_corresp[backbone_name + '/conv5/conv5_1/weights'] = 1
        vars_corresp[backbone_name + '/conv5/conv5_1/biases'] = 2
        vars_corresp[backbone_name + '/conv5/conv5_2/weights'] = 1
        vars_corresp[backbone_name + '/conv5/conv5_2/biases'] = 2
        vars_corresp[backbone_name + '/conv5/conv5_3/weights'] = 1
        vars_corresp[backbone_name + '/conv5/conv5_3/biases'] = 2

        vars_corresp[backbone_name + '/conv2_2_16/weights'] = 1
        vars_corresp[backbone_name + '/conv2_2_16/biases'] = 2
        vars_corresp[backbone_name + '/conv3_3_16/weights'] = 1
        vars_corresp[backbone_name + '/conv3_3_16/biases'] = 2
        vars_corresp[backbone_name + '/conv4_3_16/weights'] = 1
        vars_corresp[backbone_name + '/conv4_3_16/biases'] = 2
        vars_corresp[backbone_name + '/conv5_3_16/weights'] = 1
        vars_corresp[backbone_name + '/conv5_3_16/biases'] = 2

        vars_corresp[backbone_name + '/score-dsn_2/weights'] = 0.1
        vars_corresp[backbone_name + '/score-dsn_2/biases'] = 0.2
        vars_corresp[backbone_name + '/score-dsn_3/weights'] = 0.1
        vars_corresp[backbone_name + '/score-dsn_3/biases'] = 0.2
        vars_corresp[backbone_name + '/score-dsn_4/weights'] = 0.1
        vars_corresp[backbone_name + '/score-dsn_4/biases'] = 0.2
        vars_corresp[backbone_name + '/score-dsn_5/weights'] = 0.1
        vars_corresp[backbone_name + '/score-dsn_5/biases'] = 0.2

        vars_corresp[backbone_name + '/upscore-fuse/weights'] = 0.01
        vars_corresp[backbone_name + '/upscore-fuse/biases'] = 0.02
        return vars_corresp


    def _train(self, initial_ckpt, supervision, learning_rate, ckpt_logs_path, max_training_iters,
               save_step, display_step, global_step, iter_mean_grad=1, batch_size=1,
               resume_training=False, momentum=0.9, config=None, finetune=0, batch_val_size=0,
               val_images=None, val_labels=None, val_IDs=None, dice_score=False, mAP_score=False,
               test_images=None, test_IDs=None,
               parent_name='dsb18', ckpt_name='dsb18', verbose=False):
        """Train network
        Args:
            ds: Reference to a Dataset object instance
            initial_ckpt: Path to the checkpoint to initialize the network (May be pre-trained Imagenet)
            supervision: Level of the side outputs supervision: 1-Strong 2-Weak 3-No supervision
            learning_rate: Value for the learning rate. It can be a number or a learning rate object instance.
            ckpt_logs_path: Path where to store the checkpoints **and** logs
            max_training_iters: Number of training iterations
            save_step: A checkpoint will be created every save_steps
            display_step: Information of the training will be displayed every display_steps
            global_step: Reference to a Variable that keeps track of the training steps
            iter_mean_grad: Number of gradient computations that are average before updating the weights
            batch_size: Size of the training batch
            resume_training: Boolean to try to restore from a previous checkpoint (True) or not (False)
            momentum: Value of the momentum parameter for the Momentum optimizer
            config: Reference to a Configuration object used in the creation of a Session
            finetune: Use to select the type of training, 0 for base training and 1 for finetunning
            batch_val_size: Measure current performance on batch_val_size samples from the validation ds
            dice_score: Show dice score along with loss
            val_images: If val images are provided, every display_step the result of the network with these images is stored
            val_labels: If val labels are provided, every display_step the result of the network with these images is stored
            test_images: If test image are provided, every save_step the result of the network with these images is stored
            parent_name: Name of the parent convnet backbone to load from when doing transfer learning
            ckpt_name: Checkpoint name **and** name of the child convnet backbone when doing transfer learning
            verbose: if True, the convnet layers and params are listed
        Returns:
        """
        model_name = ckpt_logs_path + '/' + ckpt_name + ".ckpt"
        if config is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # config.log_device_placement = True
            # config.gpu_options.per_process_gpu_memory_fraction = 0.95
            config.allow_soft_placement = True

        tf.logging.set_verbosity(tf.logging.INFO)

        # Prepare the input and output data
        input_channels = self._ds.options['input_channels']
        input_image = tf.placeholder(tf.float32, [batch_size, None, None, input_channels])
        input_label = tf.placeholder(tf.float32, [batch_size, None, None, 1])
        predicted_label = tf.placeholder(tf.float32, [batch_size, None, None, 1])
        # NOTE: Setting the batch size prevents us from using different batch sizes
        # between training data and validation data used at the end of each epoch, BUT...
        # if we don't set the size we get the following error:
        #   File "E:/repos/tf-img-seg/tfimgseg\model.py", line 295, in _backbone
        #     side_2_s = self._crop_features(side_2_s, im_size)
        #   File "E:/repos/tf-img-seg/tfimgseg\model.py", line 138, in _crop_features
        #     return tf.reshape(slice_input, [int(feature.get_shape()[0]), out_size[1], out_size[2], int(feature.get_shape()[3])])
        #  TypeError: __int__ returned non-int (type NoneType)
        # So, just remember to set batch_size and val_batch_size at the same size
        # input_image = tf.placeholder(tf.float32, [None, None, None, input_channels])
        # input_label = tf.placeholder(tf.float32, [None, None, None, 1])
        # predicted_label = tf.placeholder(tf.float32, [None, None, None, 1])

        # Create the convnet
        with slim.arg_scope(self._backbone_arg_scope()):
            net, end_points = self._backbone(input_image, ckpt_name)
        probabilities = tf.nn.sigmoid(net)

        if verbose:
            # Print name and shape of each tensor.
            print("\nNetwork Layers:")
            for k, v in end_points.items():
                print('   name = {}, shape = {}'.format(v.name, v.get_shape()))

            # Print name and shape of parameter nodes (values not yet initialized)
            print("\nNetwork Parameters:")
            for v in slim.get_model_variables():
                print('   name = {}, shape = {}'.format(v.name, v.get_shape()))
            print("\n")

        # Initialize weights from pre-trained model
        if finetune == 0:
            init_weights = self._load_pre_trained(initial_ckpt, parent_name, ckpt_name)

        # Define loss and metrics
        with tf.name_scope('losses'):
            # Class-balanced cross entropy losses
            if supervision == 1 or supervision == 2:
                dsn_2_loss = class_balanced_cross_entropy_loss(end_points[ckpt_name + '/score-dsn_2-cr'], input_label)
                tf.summary.scalar('dsn_2_loss', dsn_2_loss)
                dsn_3_loss = class_balanced_cross_entropy_loss(end_points[ckpt_name + '/score-dsn_3-cr'], input_label)
                tf.summary.scalar('dsn_3_loss', dsn_3_loss)
                dsn_4_loss = class_balanced_cross_entropy_loss(end_points[ckpt_name + '/score-dsn_4-cr'], input_label)
                tf.summary.scalar('dsn_4_loss', dsn_4_loss)
                dsn_5_loss = class_balanced_cross_entropy_loss(end_points[ckpt_name + '/score-dsn_5-cr'], input_label)
                tf.summary.scalar('dsn_5_loss', dsn_5_loss)

            main_loss = class_balanced_cross_entropy_loss(net, input_label)
            tf.summary.scalar('main_loss', main_loss)

            if supervision == 1:
                output_loss = dsn_2_loss + dsn_3_loss + dsn_4_loss + dsn_5_loss + main_loss
            elif supervision == 2:
                output_loss = 0.5 * dsn_2_loss + 0.5 * dsn_3_loss + 0.5 * dsn_4_loss + 0.5 * dsn_5_loss + main_loss
            elif supervision == 3:
                output_loss = main_loss
            else:
                sys.exit('Incorrect supervision id, select 1 for supervision of the side outputs, 2 for weak supervision '
                         'of the side outputs and 3 for no supervision of the side outputs')
            tf.summary.scalar('output_loss', output_loss)

            total_loss = output_loss + tf.add_n(tf.losses.get_regularization_losses())
            tf.summary.scalar('total_loss', total_loss)

            # Soft Jaccard binary losses
            if supervision == 1 or supervision == 2:
                dsn_2_loss_jaccard = soft_jaccard_binary_loss(end_points[ckpt_name + '/score-dsn_2-cr'], input_label)
                tf.summary.scalar('dsn_2_loss_jaccard', dsn_2_loss_jaccard)
                dsn_3_loss_jaccard = soft_jaccard_binary_loss(end_points[ckpt_name + '/score-dsn_3-cr'], input_label)
                tf.summary.scalar('dsn_3_loss_jaccard', dsn_3_loss_jaccard)
                dsn_4_loss_jaccard = soft_jaccard_binary_loss(end_points[ckpt_name + '/score-dsn_4-cr'], input_label)
                tf.summary.scalar('dsn_4_loss_jaccard', dsn_4_loss_jaccard)
                dsn_5_loss_jaccard = soft_jaccard_binary_loss(end_points[ckpt_name + '/score-dsn_5-cr'], input_label)
                tf.summary.scalar('dsn_5_loss_jaccard', dsn_5_loss_jaccard)

            main_loss_jaccard = soft_jaccard_binary_loss(net, input_label)
            tf.summary.scalar('main_loss_jaccard', main_loss_jaccard)

            if supervision == 1:
                output_loss_jaccard = dsn_2_loss_jaccard + dsn_3_loss_jaccard + dsn_4_loss_jaccard + dsn_5_loss_jaccard + main_loss_jaccard
            elif supervision == 2:
                output_loss_jaccard = 0.5 * dsn_2_loss_jaccard + 0.5 * dsn_3_loss_jaccard + 0.5 * dsn_4_loss_jaccard + 0.5 * dsn_5_loss_jaccard + main_loss_jaccard
            elif supervision == 3:
                output_loss_jaccard = main_loss_jaccard
            else:
                sys.exit('Incorrect supervision id, select 1 for supervision of the side outputs, 2 for weak supervision '
                         'of the side outputs and 3 for no supervision of the side outputs')
            tf.summary.scalar('output_loss_jaccard', output_loss_jaccard)

            total_loss_jaccard = output_loss_jaccard + tf.add_n(tf.losses.get_regularization_losses())
            tf.summary.scalar('total_loss_jaccard', total_loss_jaccard)

        # Define optimization method
        with tf.name_scope('optimization'):
            tf.summary.scalar('learning_rate', learning_rate)
            if self.options['loss'] == 'soft_jaccard_binary_loss':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate)
                grads_and_vars = optimizer.compute_gradients(total_loss_jaccard)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
                grads_and_vars = optimizer.compute_gradients(total_loss)
            with tf.name_scope('grad_accumulator'):
                grad_accumulator = {}
                for ind in range(0, len(grads_and_vars)):
                    if grads_and_vars[ind][0] is not None:
                        grad_accumulator[ind] = tf.ConditionalAccumulator(grads_and_vars[ind][0].dtype)
            with tf.name_scope('apply_gradient'):
                layer_lr = self._parameter_lr(ckpt_name)
                grad_accumulator_ops = []
                for var_ind, grad_acc in grad_accumulator.items(): # Phil: was: for var_ind, grad_acc in grad_accumulator.iteritems():
                    var_name = str(grads_and_vars[var_ind][1].name).split(':')[0]
                    var_grad = grads_and_vars[var_ind][0]
                    grad_accumulator_ops.append(grad_acc.apply_grad(var_grad * layer_lr[var_name], local_step=global_step))
            with tf.name_scope('take_gradients'):
                mean_grads_and_vars = []
                for var_ind, grad_acc in grad_accumulator.items(): # Phil: was: for var_ind, grad_acc in grad_accumulator.iteritems():
                    mean_grads_and_vars.append(
                        (grad_acc.take_grad(iter_mean_grad), grads_and_vars[var_ind][1]))
                apply_gradient_op = optimizer.apply_gradients(mean_grads_and_vars, global_step=global_step)

        # Log training info
        if dice_score:
            with tf.name_scope('metrics'):
                dice_coef_op = dice_coef_theoretical_metric(net, input_label)
                # dice_coef_op = dice_coef_theoretical_metric(predicted_label, input_label)
                # cbce_loss_op = class_balanced_cross_entropy_loss_theoretical(predicted_label, input_label)
                tf.summary.scalar('dice_coeff', dice_coef_op)

        merged_summary_op = tf.summary.merge_all()

        # Initialize variables
        init = tf.global_variables_initializer()
        with tf.Session(config=config) as sess:
            print('Init variable')
            sess.run(init)

            # op to write logs to Tensorboard
            summary_writer = tf.summary.FileWriter(ckpt_logs_path, graph=tf.get_default_graph())
            if batch_val_size > 0 or val_images is not None:
                val_tb = TBLogger(ckpt_logs_path + '/val')

            # Create saver to manage checkpoints
            saver = tf.train.Saver(max_to_keep=None)

            # Restore from checkpoint, if requested
            last_ckpt_path = tf.train.latest_checkpoint(ckpt_logs_path)
            if last_ckpt_path is not None and resume_training:
                # Load last checkpoint
                print('Initializing from previous checkpoint...')
                saver.restore(sess, last_ckpt_path)
                step = global_step.eval() + 1
            else:
                # Load pre-trained model
                if finetune == 0:
                    print('Initializing from pre-traine model...')
                    init_weights(sess)
                else:
                    print('Initializing from previous checkpoint...')
                    # init_weights(sess)
                    var_list = []
                    for var in tf.global_variables():
                        var_type = var.name.split('/')[-1]
                        if 'weights' in var_type or 'bias' in var_type:
                            var_list.append(var)
                    saver_res = tf.train.Saver(var_list=var_list)
                    saver_res.restore(sess, initial_ckpt)
                step = 1
            sess.run(self._interp_surgery(tf.global_variables()))
            print('Weights initialized')

            # Log evolution of test images
            if test_images is not None:
                test_tb = TBLogger(ckpt_logs_path + '/test')

            print('Start training')
            while step < max_training_iters + 1:
                # Average the gradient
                for _ in range(0, iter_mean_grad):
                    # Get a batch of samples and make them conform to the network's requirements
                    samples, gt_labels = self._ds.next_batch_sem_labels(batch_size, 'train')
                    inputs, resize_info = self._preprocess_inputs(samples)
                    gt_labels = self._preprocess_labels(gt_labels, resize_info)

                    # Run the samples through the network
                    if dice_score:
                        run_res = sess.run([total_loss, merged_summary_op, dice_coef_op, total_loss_jaccard] + grad_accumulator_ops,
                                           feed_dict={input_image: inputs, input_label: gt_labels})
                        batch_loss_jaccard = run_res[3]
                    else:
                        run_res = sess.run([total_loss, merged_summary_op, total_loss_jaccard] + grad_accumulator_ops,
                                           feed_dict={input_image: inputs, input_label: gt_labels})
                        batch_loss_jaccard = run_res[2]

                    batch_loss = run_res[0]
                    summary = run_res[1]
                    if dice_score:
                      train_dice_coef = run_res[2]

                # Apply the gradients
                sess.run(apply_gradient_op)

                # Save summary reports
                summary_writer.add_summary(summary, step)

                # Test progress on validation ds, if requested
                if batch_val_size > 0 and step % display_step == 0:

                    # Because images are of different sizes, we validate a single input at a time
                    # See https://stackoverflow.com/questions/40788785/how-to-average-summaries-over-multiple-batches
                    val_batch_loss = []
                    val_batch_loss_jaccard = []
                    if dice_score:
                        val_batch_dice_coef = []
                    # if batch_val_size > batch_size:
                    rounds, _ = divmod(batch_val_size, batch_size)
                    for _round in range(rounds):
                        samples, gt_labels = self._ds.next_batch_sem_labels(batch_size, 'val')
                        inputs, resize_info = self._preprocess_inputs(samples)
                        gt_labels = self._preprocess_labels(gt_labels, resize_info)
                        if dice_score:
                            run_res = sess.run([probabilities, total_loss, dice_coef_op, total_loss_jaccard], feed_dict={input_image: inputs, input_label: gt_labels})
                            val_batch_loss.append(run_res[1])
                            val_batch_dice_coef.append(run_res[2])
                            val_batch_loss_jaccard.append(run_res[3])
                        else:
                            run_res = sess.run([probabilities, total_loss, total_loss_jaccard], feed_dict={input_image: inputs, input_label: gt_labels})
                            val_batch_loss.append(run_res[1])
                            val_batch_loss_jaccard.append(run_res[2])
                    # else:
                    #     assert(batch_size == batch_val_size)
                    #     samples, gt_labels = self._ds.next_batch_sem_labels(batch_val_size, 'val')
                    #     inputs, resize_info = self._preprocess_inputs(samples)
                    #     gt_labels = self._preprocess_labels(gt_labels, resize_info)
                    #     if dice_score:
                    #         run_res = sess.run([probabilities, total_loss, dice_coef_op],
                    #                            feed_dict={input_image: inputs, input_label: gt_labels})
                    #         val_batch_loss.append(run_res[1])
                    #         val_batch_dice_coef.append(run_res[2])
                    #     else:
                    #         run_res = sess.run([probabilities, total_loss],
                    #                            feed_dict={input_image: inputs, input_label: gt_labels})
                    #         val_batch_loss.append(run_res[1])
                    val_batch_loss = np.mean(val_batch_loss)
                    val_batch_loss_jaccard = np.mean(val_batch_loss_jaccard)
                    if dice_score:
                        val_batch_dice_coef = np.mean(val_batch_dice_coef)

                    if mAP_score:
                        val_map = []
                        rounds, _ = divmod(batch_val_size, batch_size)
                        for _round in range(rounds):
                            samples, gt_inst_maskss, _ = self._ds.next_batch_inst_masks(batch_size, 'val') # samples (1, H, W, 1) gt_inst_masks (1, #, H, W)
                            inputs, resize_info = self._preprocess_inputs(samples)
                            pred_sem_labels = sess.run(probabilities, feed_dict={input_image: inputs}) # (1, H, W, 1)
                            pred_sem_labels = self._postprocess_labels(pred_sem_labels)
                            for pred_sem_label, gt_inst_masks in zip(pred_sem_labels, gt_inst_maskss):
                                pred_inst_masks = sem_label_to_inst_masks(pred_sem_label) # (44, H, W)
                                val_map.append(average_precision(gt_inst_masks, pred_inst_masks)) # gt_masks (61, H, W) pred_inst_masks (44, H, W)
                        val_map = np.mean(val_map)

                    val_tb.log_scalar("losses/total_loss", val_batch_loss, step)
                    val_tb.log_scalar("losses/total_loss_jaccard", val_batch_loss_jaccard, step)
                    if dice_score:
                        val_tb.log_scalar("metrics/dice_coeff", val_batch_dice_coef, step)
                    if mAP_score:
                        val_tb.log_scalar("metrics/mean_AP", val_map, step)

                # Display training status
                if step % display_step == 0:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    status = "{} Iter {}: Train Loss = {:.2f} Train Loss Jaccard = {:.2f}".format(timestamp, step, batch_loss, batch_loss_jaccard)
                    if dice_score:
                        status += " Train Dice = {:.2f}".format(train_dice_coef)
                    if batch_val_size > 0:
                        status += " Val Loss = {:.2f}".format(val_batch_loss)
                        if dice_score:
                            status += " Val Dice = {:.2f}".format(val_batch_dice_coef)
                        if mAP_score:
                                status += " Val mAP = {:.2f}".format(val_map)
                    print(status)

                # Save a checkpoint
                if step % save_step == 0:

                    # Log evolution of test images
                    if test_images is not None:
                        pred_th_labels, pred_raw_labels = [], []
                        assert(type(test_images) is np.ndarray or type(test_images) is list)
                        if batch_size == 1 and len(test_images) >= 1:
                            for image in test_images:
                                inputs, adapt_info = self._preprocess_inputs(np.expand_dims(image, axis=0))  # (1, H, W, 3) or (1, H, W, 4)
                                pred_raw_label = sess.run(probabilities, feed_dict={input_image: inputs})  # (1, H, W, 1)
                                post_pred_label, post_pred_label_raw = self._postprocess_labels(pred_raw_label, adapt_info, True)
                                pred_raw_labels.append(post_pred_label_raw[0])
                                pred_th_labels.append(post_pred_label[0])
                        else:
                            assert (batch_size == len(test_images))
                            inputs, adapt_info = self._preprocess_inputs(test_images)
                            pred_raw_labels = sess.run(probabilities, feed_dict={input_image: inputs})
                            pred_th_labels, pred_raw_labels = self._postprocess_labels(pred_raw_labels, adapt_info, True)
                        test_tb.log_images('test/{}_image', test_images, step, test_IDs)
                        test_tb.log_images('test/{}_pred_th_label', pred_th_labels, step, test_IDs)
                        test_tb.log_images('test/{}_pred_raw_label', pred_raw_labels, step, test_IDs)

                    # Log evolution of val images
                    if val_images is not None:
                        # pred_th_labels, pred_raw_labels, gt_labels = [], [], []
                        pred_th_labels, pred_raw_labels = [], []
                        assert(type(val_images) is np.ndarray or type(val_images) is list)
                        rounds, _ = divmod(len(val_images), batch_size)
                        for _round in range(rounds):
                            image_batch = val_images[batch_size * _round:batch_size * (_round+1)]
                            # gt_labels_batch =  val_labels[batch_size * _round:batch_size * (_round+1)]
                            if batch_size == 1:
                                np.expand_dims(image_batch, axis=0)
                            inputs, adapt_info = self._preprocess_inputs(image_batch)  # (1, H, W, 3) or (1, H, W, 4)
                            # gt_labels.append(self._preprocess_labels(gt_labels_batch, adapt_info))
                            pred_raw_labels_batch = sess.run(probabilities, feed_dict={input_image: inputs})  # (1, H, W, 1)
                            pred_th_labels_batch, pred_raw_labels_batch = self._postprocess_labels(pred_raw_labels_batch, adapt_info, True)
                            if type(pred_th_labels_batch) is np.ndarray:
                                if pred_th_labels == []:
                                    pred_th_labels = pred_th_labels_batch.copy()
                                else:
                                    pred_th_labels = np.concatenate((pred_th_labels, pred_th_labels_batch))
                            else:
                                pred_th_labels.append(pred_th_labels_batch)
                            if type(pred_raw_labels_batch) is np.ndarray:
                                if pred_raw_labels == []:
                                    pred_raw_labels = pred_raw_labels_batch.copy()
                                else:
                                    pred_raw_labels = np.concatenate((pred_raw_labels, pred_raw_labels_batch))
                            else:
                                pred_raw_labels.append(pred_raw_labels_batch)

                        # else:
                        #     assert (batch_size == len(val_images))
                        #     inputs, adapt_info = self._preprocess_inputs(val_images)
                        #     gt_labels = self._preprocess_labels(gt_labels, adapt_info)
                        #     pred_raw_labels = sess.run(probabilities, feed_dict={input_image: inputs})
                        #     pred_th_labels, pred_raw_labels = self._postprocess_labels(pred_raw_labels, adapt_info, True)
                        val_tb.log_images('val/{}_image', val_images, step, val_IDs)
                        val_tb.log_images('val/{}_label', val_labels, step, val_IDs)
                        val_tb.log_images('val/{}_pred_th_label', pred_th_labels, step, val_IDs)
                        val_tb.log_images('val/{}_pred_raw_label', pred_raw_labels, step, val_IDs)

                    # Save model
                    save_path = saver.save(sess, model_name, global_step=global_step)
                    print("Model saved in file: %s" % save_path)

                step += 1

            # Save model
            if (step - 1) % save_step != 0:
                save_path = saver.save(sess, model_name, global_step=global_step)
                print("Model saved in file: %s" % save_path)

            print('Finished training.')


    def train(self, initial_ckpt, supervision, learning_rate, ckpt_logs_path, max_training_iters,
              save_step, display_step, global_step, iter_mean_grad=1, batch_size=1,
              resume_training=False, momentum=0.9, config=None, batch_val_size=0,
              val_images=None, val_labels=None, val_IDs=None, dice_score=False, mAP_score=False,
              test_images=None, test_IDs=None,
              parent_name='dsb18', ckpt_name='dsb18', verbose=False):
        """Train the segmentation network
        Args:
            See _train()
        Returns:
        """
        finetune = 0
        self._train(initial_ckpt, supervision, learning_rate, ckpt_logs_path, max_training_iters,
               save_step, display_step, global_step, iter_mean_grad, batch_size,
               resume_training, momentum, config, finetune, batch_val_size,
               val_images, val_labels, val_IDs, dice_score, mAP_score,
               test_images, test_IDs,
               parent_name, ckpt_name, verbose)


    def finetune(self, initial_ckpt, supervision, learning_rate, ckpt_logs_path, max_training_iters,
                 save_step, display_step, global_step, iter_mean_grad=1, batch_size=1,
                 resume_training=False, momentum=0.9, config=None, batch_val_size=0,
                 val_images=None, val_labels=None, val_IDs=None, dice_score=False, mAP_score=False,
                 test_images=None, test_IDs=None,
                 parent_name='dsb18', ckpt_name='dsb18', verbose=False):
        """Finetune the segmentation network
        Args:
            See _train()
        Returns:
        """
        finetune = 1
        self._train(initial_ckpt, supervision, learning_rate, ckpt_logs_path, max_training_iters,
               save_step, display_step, global_step, iter_mean_grad, batch_size,
               resume_training, momentum, config, finetune, batch_val_size,
               val_images, val_labels, val_IDs, dice_score, mAP_score,
               test_images, test_IDs,
               parent_name, ckpt_name, verbose)


    def _test_sem_labels(self, checkpoint_file, backbone_name='dsb18', config=None, verbose=False):
        """Run the segmentation network on the test dataset to generate semantic labels
        Args:
            ds: Reference to a Dataset object instance
            checkpoint_file: Path of the saved model to load
            backbone_name: Name of the convnet backbone
            config: Reference to a Configuration object used in the creation of a Session
            verbose: if True, the convnet layers and params are listed
        Returns:
        """
        if config is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # config.log_device_placement = True
            config.allow_soft_placement = True
        tf.logging.set_verbosity(tf.logging.INFO)

        # Input data
        batch_size = 1
        input_channels = self._ds.options['input_channels']
        input_image = tf.placeholder(tf.float32, [batch_size, None, None, input_channels])

        # Create the convnet
        with slim.arg_scope(self._backbone_arg_scope()):
            net, end_points = self._backbone(input_image, backbone_name)
        probabilities = tf.nn.sigmoid(net)

        if verbose:
            # Print name and shape of each tensor.
            print("\nNetwork Layers:")
            for k, v in end_points.items():
                print('   name = {}, shape = {}'.format(v.name, v.get_shape()))

            # Print name and shape of parameter nodes (values not yet initialized)
            print("\nNetwork Parameters:")
            for v in slim.get_model_variables():
                print('   name = {}, shape = {}'.format(v.name, v.get_shape()))
            print("\n")

        # Create a saver to load the network
        saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name and '-cr' not in v.name])

        ds_size = self._ds.test_size

        with tf.Session(config=config) as sess:
            # Load up saved model
            sess.run(tf.global_variables_initializer())
            sess.run(self._interp_surgery(tf.global_variables()))
            saver.restore(sess, checkpoint_file)

            # Chunk dataset
            rounds, rounds_left = divmod(ds_size, batch_size)
            if rounds_left:
                rounds += 1

            desc = 'Saving sem seg preds as PNGs'

            # Go through input samples and generate predictions
            for _round in trange(rounds, ascii=True, ncols=100, desc=desc):
                # Read in input batch
                samples, output_files = self._ds.next_batch_sem_labels(batch_size, 'test_with_pred_paths')

                # Preprocess images and run them through convnet
                inputs, adapt_info = self._preprocess_inputs(samples)
                pred_raw_labels = sess.run(probabilities, feed_dict={input_image: inputs})
                pred_th_labels, pred_raw_labels = self._postprocess_labels(pred_raw_labels, adapt_info, True)

                # Threshold predictions and save resulting semantic segmentation
                for pred_label, pred_label_raw, output_file in zip(pred_th_labels, pred_raw_labels, output_files):
                    save_sem_label(pred_label, output_file)
                    save_sem_label(pred_label_raw, output_file.replace(self._ds.pred_label_folder, self._ds.pred_label_folder_raw))


    def _test_inst_masks(self, checkpoint_file, backbone_name='dsb18', config=None, verbose=False):
        """Run the segmentation network on the test dataset to generate instance masks
        Args:
            ds: Reference to a Dataset object instance
            checkpoint_file: Path of the saved model to load
            backbone_name: Name of the convnet backbone
            config: Reference to a Configuration object used in the creation of a Session
            verbose: if True, the convnet layers and params are listed
        Returns:
        """
        if config is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # config.log_device_placement = True
            config.allow_soft_placement = True
        tf.logging.set_verbosity(tf.logging.INFO)

        # Input data
        batch_size = 1
        input_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])

        # Create the convnet
        with slim.arg_scope(self._backbone_arg_scope()):
            net, end_points = self._backbone(input_image, backbone_name)
        probabilities = tf.nn.sigmoid(net)

        if verbose:
            # Print name and shape of each tensor.
            print("\nNetwork Layers:")
            for k, v in end_points.items():
                print('   name = {}, shape = {}'.format(v.name, v.get_shape()))

            # Print name and shape of parameter nodes (values not yet initialized)
            print("\nNetwork Parameters:")
            for v in slim.get_model_variables():
                print('   name = {}, shape = {}'.format(v.name, v.get_shape()))
            print("\n")

        # Create a saver to load the network
        saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name and '-cr' not in v.name])

        with tf.Session(config=config) as sess:
            # Load up saved model
            sess.run(tf.global_variables_initializer())
            sess.run(self._interp_surgery(tf.global_variables()))
            saver.restore(sess, checkpoint_file)

            # Chunk dataset
            rounds, rounds_left = divmod(self._ds.test_size, batch_size)
            if rounds_left:
                rounds += 1

            # Go through input samples and generate predictions
            for _round in trange(rounds, ascii=True, ncols=100, desc='Saving inst mask preds as PNGs'):
                # Get samples from dataset along with folder paths where we can save predictions
                samples, pred_folders = self._ds.next_batch_inst_masks(batch_size, 'test_with_pred_paths')

                # Preprocess images and run them through convnet
                inputs = self._preprocess_inputs(samples)
                pred_sem_labels = sess.run(probabilities, feed_dict={input_image: inputs})

                # Threshold predictions and turn result into instance masks using connected components
                for pred_sem_label, pred_folder in zip(pred_sem_labels, pred_folders):
                    th_pred_sem_label = np.where(pred_sem_label.astype(np.float32) < self._ds.bin_threshold / 255.0, 0, 255).astype('uint8')
                    pred_inst_masks = sem_label_to_inst_masks(th_pred_sem_label)
                    save_inst_masks(pred_inst_masks, pred_folder)


    def test(self, checkpoint_file, backbone_name='dsb18', config=None, verbose=False):
        """Run the segmentation network on the test dataset
        Args:
            ds: Reference to a Dataset object instance
            checkpoint_file: Path of the saved model to load
            backbone_name: Name of the convnet backbone
            config: Reference to a Configuration object used in the creation of a Session
            verbose: if True, the convnet layers and params are listed
        Returns:
        """
        if self._ds.options['mode'] == 'semantic_labels' or self._ds.options['mode'] == 'semantic_contours':
            self._test_sem_labels(checkpoint_file, backbone_name, config, verbose)
        else:
            self._test_inst_masks(checkpoint_file, backbone_name, config, verbose)


    def _validate_sem_labels(self, checkpoint_file, backbone_name='dsb18', config=None, test_thresholds=False, verbose=False, save_preds=False):
        """Test the segmentation network on the validation split
        Args:
            dataset: Reference to a Dataset object instance
            checkpoint_file: Path of the saved model to load
            backbone_name: Name of the convnet backbone
            config: Reference to a Configuration object used in the creation of a Session
            test_thresholds: If True, test 10 different thresholds and report dice scores for each
            verbose: if True, the convnet layers and params are listed
            save_preds: if True, the predictions are saved to disk
        Returns:
            dice score for the entire dataset
        Note:
            Here's the result of a single run of the threshold test:
            Dice score: 0.8518371546446387, CBCE: 8459.504040002823 (computed over 134 labels with 130.0 threshold)
            Dice score: 0.8531420150799538, CBCE: 8425.063483057627 (computed over 134 labels with 132.0 threshold)
            Dice score: 0.8543733589684785, CBCE: 8393.46040225118 (computed over 134 labels with 134.0 threshold)
            Dice score: 0.8555805807683006, CBCE: 8362.330550525616 (computed over 134 labels with 136.0 threshold)
            Dice score: 0.8567736340102865, CBCE: 8334.17109070963 (computed over 134 labels with 138.0 threshold)
            Dice score: 0.8580066312604876, CBCE: 8301.334026078679 (computed over 134 labels with 140.0 threshold)
            Dice score: 0.8592976229404335, CBCE: 8273.32314491094 (computed over 134 labels with 142.0 threshold)
            Dice score: 0.8605125626521324, CBCE: 8247.562483515312 (computed over 134 labels with 144.0 threshold)
            Dice score: 0.8616840305613048, CBCE: 8226.25542128086 (computed over 134 labels with 146.0 threshold)
            Dice score: 0.8628463367028023, CBCE: 8209.59751162867 (computed over 134 labels with 148.0 threshold)
            Dice score: 0.8640216473323196, CBCE: 8192.24647356414 (computed over 134 labels with 150.0 threshold)
            Dice score: 0.8651804505889096, CBCE: 8176.3894657261335 (computed over 134 labels with 152.0 threshold)
            Dice score: 0.8663609481569546, CBCE: 8160.1797391932405 (computed over 134 labels with 154.0 threshold)
            Dice score: 0.867567390203476, CBCE: 8151.337793882213 (computed over 134 labels with 156.0 threshold)
            Dice score: 0.86875133461027, CBCE: 8137.50351312801 (computed over 134 labels with 158.0 threshold)
            Dice score: 0.8699696864654769, CBCE: 8128.232469065865 (computed over 134 labels with 160.0 threshold)
            >> best is below:
            Dice score: 0.8712408716109261, CBCE: 8120.74282263464 (computed over 134 labels with 162.0 threshold)
            >>
            Dice score: 0.8723834090268434, CBCE: 8122.693916080603 (computed over 134 labels with 164.0 threshold)
            Dice score: 0.873583611712527, CBCE: 8121.4720877302225 (computed over 134 labels with 166.0 threshold)
            Dice score: 0.8747817740511539, CBCE: 8122.778636717085 (computed over 134 labels with 168.0 threshold)
            Dice score: 0.8758676830512374, CBCE: 8132.6266939657835 (computed over 134 labels with 170.0 threshold)
            Dice score: 0.8769456294935141, CBCE: 8143.730113056168 (computed over 134 labels with 172.0 threshold)
            Dice score: 0.8781061612848026, CBCE: 8162.06876138253 (computed over 134 labels with 174.0 threshold)
            Dice score: 0.8792135128334387, CBCE: 8173.496174924409 (computed over 134 labels with 176.0 threshold)
            Dice score: 0.8804206643531571, CBCE: 8194.767461329699 (computed over 134 labels with 178.0 threshold)
            Dice score: 0.8815796108388189, CBCE: 8218.295748932593 (computed over 134 labels with 180.0 threshold)
        """
        if config is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # config.log_device_placement = True
            config.allow_soft_placement = True
        tf.logging.set_verbosity(tf.logging.INFO)

        # Input data
        batch_size = 1
        input_channels = self._ds.options['input_channels']
        input_image = tf.placeholder(tf.float32, [batch_size, None, None, input_channels])
        input_label = tf.placeholder(tf.float32, [batch_size, None, None, 1])
        predicted_label = tf.placeholder(tf.float32, [batch_size, None, None, 1])

        # Create the convnet
        with slim.arg_scope(self._backbone_arg_scope()):
            net, end_points = self._backbone(input_image, backbone_name)
        probabilities = tf.nn.sigmoid(net)

        if verbose:
            # Print name and shape of each tensor.
            print("\nNetwork Layers:")
            for k, v in end_points.items():
                print('   name = {}, shape = {}'.format(v.name, v.get_shape()))

            # Print name and shape of parameter nodes (values not yet initialized)
            print("\nNetwork Parameters:")
            for v in slim.get_model_variables():
                print('   name = {}, shape = {}'.format(v.name, v.get_shape()))
            print("\n")

        # Create a saver to load the network
        saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name and '-cr' not in v.name])

        # Use TF to compute perf measures
        dice_coef_op = dice_coef_theoretical_metric(predicted_label, input_label)
        cbce_loss_op = class_balanced_cross_entropy_loss_theoretical(predicted_label, input_label)
        ds_size = self._ds.val_size
        global_dice_coef = 0.

        if test_thresholds:
            thresholds = [float(threshold) for threshold in range(130, 182, 2)]
            thresholded_dice_coefs = [0] * len(thresholds)
            cbce_losses = [0] * len(thresholds)

        with tf.Session(config=config) as sess:
            # Load up saved model
            sess.run(tf.global_variables_initializer())
            sess.run(self._interp_surgery(tf.global_variables()))
            saver.restore(sess, checkpoint_file)

            # Chunk dataset
            rounds, rounds_left = divmod(ds_size, batch_size)
            if rounds_left:
                rounds += 1

            desc = 'Measuring DICE and saving sem seg preds' if save_preds else 'Measuring DICE'

            for _round in trange(rounds, ascii=True, ncols=100, desc=desc):
                # Get samples from dataset along with groundtruths so we can evaluate predictions
                samples, gt_labels, output_files = self._ds.next_batch_sem_labels(batch_size, 'val_with_pred_paths')

                # Preprocess images and run them through convnet
                inputs, adapt_info = self._preprocess_inputs(samples)
                gt_labels = self._preprocess_labels(gt_labels, adapt_info)
                pred_raw_labels = sess.run(probabilities, feed_dict={input_image: inputs})
                pred_th_labels, pred_raw_labels = self._postprocess_labels(pred_raw_labels, adapt_info, True)

                if test_thresholds:
                    # Test binary image thresholding at different thresholds and measure prediction performance
                    for idx in range(len(thresholds)):
                        thresholded_masks = np.where(pred_raw_labels.astype(np.float32) < thresholds[idx] / 255.0, 0, 255).astype('uint8')
                        thresholded_masks = self._preprocess_labels(thresholded_masks, adapt_info)
                        dice_coefs = sess.run(dice_coef_op, feed_dict={predicted_label: thresholded_masks, input_label: gt_labels})
                        cbce_loss = sess.run(cbce_loss_op, feed_dict={predicted_label: thresholded_masks, input_label: gt_labels})
                        thresholded_dice_coefs[idx] += np.sum(dice_coefs)
                        cbce_losses[idx] += np.sum(cbce_loss)
                else:
                    # Save raw and thresholded prediction
                    # pred_sem_labels_raw = pred_sem_labels.astype(np.float32)
                    # pred_sem_labels = np.where(pred_sem_labels_raw < self._ds.bin_threshold / 255.0, 0, 255).astype('uint8')
                    if save_preds:
                        for pred_label, pred_label_raw, output_file in zip(pred_th_labels, pred_raw_labels, output_files):
                            save_sem_label(pred_label, output_file)
                            save_sem_label(pred_label_raw, output_file.replace(self._ds.pred_label_folder, self._ds.pred_label_folder_raw))
                    # Measure prediction performance
                    # pred_sem_labels = self._preprocess_labels(pred_sem_labels)
                    dice_coefs = sess.run(dice_coef_op, feed_dict={predicted_label: pred_th_labels, input_label: gt_labels})
                    global_dice_coef += np.sum(dice_coefs)

        if test_thresholds:
            for idx in range(len(thresholds)):
                print("Dice score: {}, CBCE: {} (computed over {} labels with {} threshold)".format(thresholded_dice_coefs[idx]/ds_size, cbce_losses[idx]/ds_size, ds_size, thresholds[idx]))
            best_threshold_idx = np.argmax(thresholded_dice_coefs)
            print("Best threshold is {}".format(thresholds[best_threshold_idx]))
        else:
            return global_dice_coef/ds_size


    def _validate_inst_masks(self, checkpoint_file, backbone_name='dsb18', config=None, test_thresholds=False, verbose=False, save_preds=False):
        """Test the segmentation network on the validation split.
        We return a mean average precision of the IoU at a set of thresholds.
        Args:
            ds: Reference to a Dataset object instance
            checkpoint_file: Path of the saved model to load
            backbone_name: Name of the convnet backbone
            config: Reference to a Configuration object used in the creation of a Session
            test_thresholds: If True, test 10 different thresholds and report mAP of the IoU scores for each
            verbose: if True, the convnet layers and params are listed
            save_preds: if True, the predictions are saved to disk
        Returns:
            mAP for the entire dataset
        Note:
            Here's the result of a single run of the threshold test:
            mAP: 0.36943722728155953 (over 134 labels with 130.0 threshold)
            mAP: 0.3721465875715825 (over 134 labels with 132.0 threshold)
            mAP: 0.37371437618769404 (over 134 labels with 134.0 threshold)
            mAP: 0.3751947304978974 (over 134 labels with 136.0 threshold)
            mAP: 0.3769107748020378 (over 134 labels with 138.0 threshold)
            mAP: 0.3786374974250231 (over 134 labels with 140.0 threshold)
            mAP: 0.38192444549863064 (over 134 labels with 142.0 threshold)
            mAP: 0.3836121606307755 (over 134 labels with 144.0 threshold)
            mAP: 0.38537071942223994 (over 134 labels with 146.0 threshold)
            mAP: 0.38769571559945004 (over 134 labels with 148.0 threshold)
            mAP: 0.3895884827809215 (over 134 labels with 150.0 threshold)
            mAP: 0.39172943932307863 (over 134 labels with 152.0 threshold)
            mAP: 0.39320667759446254 (over 134 labels with 154.0 threshold)
            mAP: 0.39528503862548925 (over 134 labels with 156.0 threshold)
            mAP: 0.3978522006174672 (over 134 labels with 158.0 threshold)
            mAP: 0.3996349836573802 (over 134 labels with 160.0 threshold)
            mAP: 0.40019509752168675 (over 134 labels with 162.0 threshold)
            mAP: 0.4013115790628193 (over 134 labels with 164.0 threshold)
            mAP: 0.4024284805046497 (over 134 labels with 166.0 threshold)
            mAP: 0.40277877880037927 (over 134 labels with 168.0 threshold)
            mAP: 0.40384377656562 (over 134 labels with 170.0 threshold)
            mAP: 0.40607537709486635 (over 134 labels with 172.0 threshold)
            mAP: 0.4072852791463322 (over 134 labels with 174.0 threshold)
            mAP: 0.4082697427810162 (over 134 labels with 176.0 threshold)
            mAP: 0.409066883298893 (over 134 labels with 178.0 threshold)
            mAP: 0.41113315341516893 (over 134 labels with 180.0 threshold)
            mAP: 0.4130830401916921 (over 134 labels with 184.0 threshold)
            mAP: 0.41558930374508357 (over 134 labels with 186.0 threshold)
            mAP: 0.41790133583467004 (over 134 labels with 188.0 threshold)
            mAP: 0.4198910172995109 (over 134 labels with 190.0 threshold)
            mAP: 0.42044460662034266 (over 134 labels with 192.0 threshold)
            mAP: 0.42282938038447454 (over 134 labels with 194.0 threshold)
            mAP: 0.42423871069348945 (over 134 labels with 196.0 threshold)
            mAP: 0.42684488305203433 (over 134 labels with 198.0 threshold)
            mAP: 0.42966657591296276 (over 134 labels with 200.0 threshold)
            mAP: 0.43102161475534695 (over 134 labels with 202.0 threshold)
            mAP: 0.43250514053476347 (over 134 labels with 204.0 threshold)
            mAP: 0.4340077202924674 (over 134 labels with 206.0 threshold)
            mAP: 0.43439064550615536 (over 134 labels with 208.0 threshold)
            mAP: 0.43606531705107376 (over 134 labels with 210.0 threshold)
            mAP: 0.43739846969721435 (over 134 labels with 212.0 threshold)
            mAP: 0.43848257103929683 (over 134 labels with 214.0 threshold)
            mAP: 0.4390262890590712 (over 134 labels with 216.0 threshold)
            mAP: 0.44032881854804967 (over 134 labels with 218.0 threshold)
            mAP: 0.44148910627475646 (over 134 labels with 220.0 threshold)
            mAP: 0.44148511180584626 (over 134 labels with 222.0 threshold)
            mAP: 0.44296936031602446 (over 134 labels with 224.0 threshold)
            mAP: 0.44245124611540126 (over 134 labels with 226.0 threshold)
            mAP: 0.44184552673011507 (over 134 labels with 228.0 threshold)
            mAP: 0.44185667865464595 (over 134 labels with 230.0 threshold)
            Best threshold is 224.0
            iter 37500: mask_mAP = 0.44296936031602446
        """
        if config is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # config.log_device_placement = True
            config.allow_soft_placement = True
        tf.logging.set_verbosity(tf.logging.INFO)

        # Input data
        batch_size = 1
        input_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])

        # Create the convnet
        with slim.arg_scope(self._backbone_arg_scope()):
            net, end_points = self._backbone(input_image, backbone_name)

        if verbose:
            # Print name and shape of each tensor.
            print("\nNetwork Layers:")
            for k, v in end_points.items():
                print('   name = {}, shape = {}'.format(v.name, v.get_shape()))

            # Print name and shape of parameter nodes (values not yet initialized)
            print("\nNetwork Parameters:")
            for v in slim.get_model_variables():
                print('   name = {}, shape = {}'.format(v.name, v.get_shape()))
            print("\n")

        probabilities = tf.nn.sigmoid(net)

        # Create a saver to load the network
        saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name and '-cr' not in v.name])

        ds_size = self._ds.val_size
        global_mAP = 0.

        if test_thresholds:
            # thresholds = [float(threshold) for threshold in range(130, 182, 2)]
            # thresholds = [float(threshold) for threshold in range(184, 212, 2)]
            thresholds = [float(threshold) for threshold in range(212, 232, 2)]
            th_mAPs = [0] * len(thresholds)

        with tf.Session(config=config) as sess:
            # Load up saved model
            sess.run(tf.global_variables_initializer())
            sess.run(self._interp_surgery(tf.global_variables()))
            saver.restore(sess, checkpoint_file)

            # Chunk dataset
            rounds, rounds_left = divmod(ds_size, batch_size)
            if rounds_left:
                rounds += 1

            if test_thresholds:
                desc = 'Measuring mAP and saving preds at different thresholds' if save_preds else 'Measuring mAP at different thresholds'
            else:
                desc = 'Measuring mAP and saving preds' if save_preds else 'Measuring mAP'

            # Go through input samples and generate predictions
            for _round in trange(rounds, ascii=True, ncols=100, desc=desc):
                # Get samples from dataset along with groundtruths so we can evaluate predictions
                samples, gt_masks, _, pred_folders = self._ds.next_batch_inst_masks(batch_size, 'val_with_pred_paths')

                # Preprocess images and run them through convnet
                inputs = self._preprocess_inputs(samples)
                pred_sem_labels = sess.run(probabilities, feed_dict={input_image: inputs})

                if test_thresholds:
                    # Test binary image thresholding at different thresholds and measure prediction performance
                    for idx in range(len(thresholds)):
                        for pred_sem_label, gt_inst_masks in zip(pred_sem_labels, gt_masks):
                            th_pred_sem_label = np.where(pred_sem_label.astype(np.float32) < thresholds[idx] / 255.0, 0,
                                                         255).astype('uint8')
                            pred_inst_masks = sem_label_to_inst_masks(th_pred_sem_label)
                            th_mAPs[idx] += average_precision(gt_inst_masks, pred_inst_masks)
                            # print("mAP: {} (over {} labels with {} threshold)".format(th_mAPs[idx] / ds_size, ds_size, thresholds[idx]))
                else:
                    # Measure prediction performance
                    for pred_sem_label, gt_inst_masks, pred_folder in zip(pred_sem_labels, gt_masks, pred_folders):
                        th_pred_sem_label = np.where(pred_sem_label.astype(np.float32) < self._ds.bin_threshold / 255.0, 0, 255).astype('uint8')
                        pred_inst_masks = sem_label_to_inst_masks(th_pred_sem_label)
                        if save_preds:
                            save_inst_masks(pred_inst_masks, pred_folder)
                        global_mAP += average_precision(gt_inst_masks, pred_inst_masks)

        if test_thresholds:
            for idx in range(len(thresholds)):
                print("mAP: {} (over {} labels with {} threshold)".format(th_mAPs[idx]/ds_size, ds_size, thresholds[idx]))
            best_th_idx = np.argmax(th_mAPs)
            global_mAP = th_mAPs[best_th_idx]
            print("Best threshold is {}".format(thresholds[best_th_idx]))

        return global_mAP / ds_size


    def validate(self, checkpoint_file, backbone_name='dsb18', config=None, test_thresholds=False, verbose=False, save_preds=False):
        """Test the segmentation network on the validation split.
        It the dataset is in "semantic labels" mode, we return a dice score for the dataset.
        It the dataset is in "instance masks" mode, we return a mean average precision of the IoU at a set of thresholds.
        Args:
            ds: Reference to a Dataset object instance
            checkpoint_file: Path of the saved model to load
            backbone_name: Name of the convnet backbone
            config: Reference to a Configuration object used in the creation of a Session
            test_thresholds: If True, test 10 different thresholds and report dice scores for each
            verbose: if True, the convnet layers and params are listed
            save_preds: if True, the predictions are saved to disk
        Returns:
            In "semantic labels" mode:
                dice score for the entire dataset
            In "instance masks" mode:
                mean average precision of the IoU at a set of thresholds for the entire dataset
        """
        if self._ds.options['mode'] == 'semantic_labels' or self._ds.options['mode'] == 'semantic_contours':
                return self._validate_sem_labels(checkpoint_file, backbone_name, config, test_thresholds, verbose, save_preds)
        else:
            return self._validate_inst_masks(checkpoint_file, backbone_name, config, test_thresholds, verbose, save_preds)

    ###
    ### Debug utils
    ###
    def print_config(self):
        """Display configuration values."""
        print("\nModel Configuration:")
        for k, v in self.options.items():
            print("  {:20} {}".format(k, v))
        print("  {:20} {}".format('phase', self._phase))
