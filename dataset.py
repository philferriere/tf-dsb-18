"""
dataset.py

Dataset utility functions and classes.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Notes:
    Basic stats about the training and test sets:
        (Training examples, Test examples): ( 670 , 65 )

    Image sizes present in the training set:
        (256, 256, 3) : 334
        (1024, 1024, 3) : 16
        (520, 696, 3) : 92
        (360, 360, 3) : 91
        (512, 640, 3) : 13
        (256, 320, 3) : 112
        (1040, 1388, 3) : 1
        (260, 347, 3) : 5
        (603, 1272, 3) : 6

    Paths:
    ID/images/ID.png input RGB image (given)
    ID/masks/IDs.png input bin masks (given)

    ID/labels/ID.png equivalent sem seg bin label from given input bin masks (generated)
    ID/contours/ID.png sem seg bin label of contours computed from given bin input masks (generated)
    ID/combined/ID.png input + masks + contours (genereted)

    ID/pred_sem_label/ID.png pred sem seg bin label (predicted)
    ID/pred_sem_label_raw/ID.png pred sem seg unthresholded label (predicted)

    ID/pred_contours_#px/#s.png pred sem seg of bin contours (predicted)
    ID/pred_contours_#px_raw/#s.png pred sem seg of unthresholded contours (predicted)

    ID/pred_inst_masks/#s.png pred instance masks (predicted)
    ID/pred_labeled_mask/ID.png pred instance masks combined in a single labeled mask (predicted)
    ID/pred_contours/#s.png pred sem seg of contours (predicted)
    ID/pred_combined/ID.png input + pred instance masks (predicted)
"""

# TODO Add support for TFRecords

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, warnings
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
from scipy import ndimage as ndi

from bboxes import extract_bboxes
from contours import inst_masks_contours_to_label
from segment import load_inst_masks, inst_masks_to_sem_label
from augment import Augmenter, _DEFAULT_AUG_OPTIONS
from sampling import Sampler, _DEFAULT_SAMPLING_OPTIONS

if sys.platform.startswith("win"):
    _DSB18_DATASET = "E:/datasets/dsb18"
else:
    _DSB18_DATASET = '/media/EDrive/datasets/dsb18'

_DBG_TRAIN_VAL_TEST_SETS = -1

_DEFAULT_DS_NUCLEI_TRAIN_OPTIONS = {
    'mode': 'semantic_labels',  # ['instance_masks' | 'semantic_labels' | 'instance_contours' | 'semantic_contours']
    'in_memory': True,
    'data_aug': 'basic', # None or 'basic' or 'heavy'
    'compute_bboxes': False,
    'crop_first': (256, 256),
    'unique_img_size': None, # (None, 256),
    'input_channels': 3,  # [3 | 4]
    'contour_thickness': 2,
    'bin_threshold': 214,
    # Sampling options
    'sampling': 'oversampling', # ['normal' | 'oversampling']
    'train_val_split': 'stratified', # ['random' | 'stratified']
    'clustering': 'hsv_clusters' , # None or ['hsv_clusters' | 'hed_clusters' | rgb_clusters]
    'num_clusters': 3,
    'num_colors': 1,
    'random_seed' : 1969,
    'val_split' : 0.2}

_DEFAULT_DS_NUCLEI_VAL_TEST_OPTIONS = {
    'mode': 'semantic_labels',  # ['instance_masks' | 'semantic_labels' | 'instance_contours' | 'semantic_contours']
    'in_memory': True,
    'data_aug': None, # None or 'basic' or 'heavy'
    'compute_bboxes': False,
    'crop_first': None,
    'unique_img_size': None,
    'input_channels': 3,  # [3 | 4]
    'contour_thickness': 2,
    'bin_threshold': 214,
    # Sampling options
    'sampling': 'oversampling', # ['normal' | 'oversampling']
    'train_val_split': 'stratified', # ['random' | 'stratified']
    'clustering': 'hsv_clusters' , # None or ['hsv_clusters' | 'hed_clusters' | rgb_clusters]
    'num_clusters': 3,
    'num_colors': 1,
    'random_seed' : 1969,
    'val_split' : 0.2}

_DEFAULT_DS_CONTOURS_TRAIN_OPTIONS = {
    'mode': 'semantic_contours',  # ['instance_masks' | 'semantic_labels' | 'instance_contours' | 'semantic_contours']
    'in_memory': True,
    'data_aug': 'basic', # None or 'basic' or 'heavy'
    'compute_bboxes': False,
    'crop_first': (256, 256),
    'unique_img_size': (None, 256),
    'input_channels': 4,  # [3 | 4]
    'contour_thickness': 2,
    'bin_threshold': 214,
    # Sampling options
    'sampling': 'oversampling', # ['normal' | 'oversampling']
    'train_val_split': 'stratified', # ['random' | 'stratified']
    'clustering': 'hsv_clusters' , # None or ['hsv_clusters' | 'hed_clusters' | rgb_clusters]
    'num_clusters': 3,
    'num_colors': 1,
    'random_seed' : 1969,
    'val_split' : 0.2}

_DEFAULT_DS_CONTOURS_VAL_TEST_OPTIONS = {
    'mode': 'semantic_contours',  # ['instance_masks' | 'semantic_labels' | 'instance_contours' | 'semantic_contours']
    'in_memory': True,
    'data_aug': None, # None or 'basic' or 'heavy'
    'compute_bboxes': False,
    'crop_first': None,
    'unique_img_size': None,
    'input_channels': 4,  # [3 | 4]
    'contour_thickness': 2,
    'bin_threshold': 214,
    # Sampling options
    'sampling': 'oversampling', # ['normal' | 'oversampling']
    'train_val_split': 'stratified', # ['random' | 'stratified']
    'clustering': 'hsv_clusters' , # None or ['hsv_clusters' | 'hed_clusters' | rgb_clusters]
    'num_clusters': 3,
    'num_colors': 1,
    'random_seed' : 1969,
    'val_split' : 0.2}

_DEFAULT_DS_DIRECTION_TRAIN_OPTIONS = {
    'mode': 'dwt_direction',  # ['instance_masks' | 'semantic_labels' | 'instance_contours' | 'semantic_contours']
    'in_memory': True,
    'data_aug': None, # None or 'basic' or 'heavy'
    'compute_bboxes': False,
    'crop_first': (256, 256),
    'unique_img_size': (None, 256),
    'input_channels': 4,  # [3 | 4]
    # Sampling options
    'sampling': 'oversampling', # ['normal' | 'oversampling']
    'train_val_split': 'stratified', # ['random' | 'stratified']
    'clustering': 'hsv_clusters' , # None or ['hsv_clusters' | 'hed_clusters' | rgb_clusters]
    'num_clusters': 3,
    'num_colors': 1,
    'random_seed' : 1969,
    'val_split' : 0.2}

_DEFAULT_DS_DIRECTION_VAL_TEST_OPTIONS = {
    'mode': 'dwt_direction',  # ['instance_masks' | 'semantic_labels' | 'instance_contours' | 'semantic_contours']
    'in_memory': True,
    'data_aug': None, # None or 'basic' or 'heavy'
    'compute_bboxes': False,
    'crop_first': None,
    'unique_img_size': None,
    'input_channels': 4,  # [3 | 4]
    # Sampling options
    'sampling': 'oversampling', # ['normal' | 'oversampling']
    'train_val_split': 'stratified', # ['random' | 'stratified']
    'clustering': 'hsv_clusters' , # None or ['hsv_clusters' | 'hed_clusters' | rgb_clusters]
    'num_clusters': 3,
    'num_colors': 1,
    'random_seed' : 1969,
    'val_split' : 0.2}

# For backward compatibility only
_DEFAULT_DSB18_OPTIONS = _DEFAULT_DS_NUCLEI_TRAIN_OPTIONS

class DSB18Dataset(object):
    """2018 Data Science Bowl nuclei segmentation dataset.
    """

    def __init__(self, phase='train_val', ds_root=_DSB18_DATASET, options=_DEFAULT_DS_NUCLEI_TRAIN_OPTIONS):
        """Initialize the Dataset object
        Args:
            phase: Possible options: 'train_noval', 'val', 'train_val' or 'test'
            ds_root: Path to the root of the dataset
            options: see below
        Options:
            in_memory: True loads all the training images upfront, False loads images in small batches
            data_aug: Adds augmented data to training set
            compute_bboxes: True enables on-th-fly computation of bounding boxes of of instance masks
            unique_img_size: True if all input images are to be scale to the same HxW size (False for DSB 18)
            random_seed: Random seed used for sampling
            input_channels: Sets it input images are RGB nuclei or RGB+nuclei sem seg 4-channel images
            contour_thickness: Sets the size of the contours to generate for instance masks at training time
            random_seed: what it says on the can
            val_split: Portion of data reserved for the validation split
        """
        # Only options supported in this initial implementation
        assert (phase in ['train_noval', 'val', 'train_val', 'test'])
        assert(options['mode'] in ['instance_masks', 'semantic_labels', 'instance_contours', 'semantic_contours', 'dwt_direction'])
        self._phase = phase
        self.options = options

        # Set paths and file names
        self._ds_root = ds_root

        self._train_folder = self._ds_root + '/stage1_train'
        self._val_folder = self._ds_root + '/stage1_train'
        self._test_folder = self._ds_root + '/stage1_test'

        self._train_IDs_file = self._ds_root + '/train.txt'
        self._val_IDs_file = self._ds_root + '/val.txt'
        self._test_IDs_file = self._ds_root + '/test.txt'

        # Setup sub-paths and thresholding levels (see model.py for the hypertuning results)
        if self.options['mode'] in ['semantic_labels', 'instance_masks']:
            self._label_folder = '/labels/'
            self._pred_label_folder = '/pred_sem_label/'
            self._pred_label_folder_raw = '/pred_sem_label_raw/'
            if self.options['bin_threshold'] == -1:
                self.options['bin_threshold'] = 234
        elif self.options['mode'] in ['instance_contours', 'semantic_contours']:
            self._label_folder = '/contours_{}px/'.format(self.options['contour_thickness'])
            self._pred_label_folder = '/pred_contours_{}px/'.format(self.options['contour_thickness'])
            self._pred_label_folder_raw = '/pred_contours_{}px_raw/'.format(self.options['contour_thickness'])
            if self.options['bin_threshold'] == -1:
                self.options['bin_threshold'] = 234 # 52
        elif self.options['mode'] == 'dwt_direction':
            self._label_folder = '/normed_gradients/'
            self._pred_label_folder = '/pred_normed_gradients/'

        # Load ID files
        if not self._load_ID_files():
            self.prepare()

        # Instantiate augmenter, if requested
        if self.options['data_aug']:
            assert (phase != 'test' and phase != 'val')
            assert (self.options['data_aug'] in ['basic', 'heavy'])
            self._aug_options = _DEFAULT_AUG_OPTIONS
            if self.options['data_aug'] is 'basic':
                self._aug_options['piecewise_affine'] = (0., 0.)
            elif self.options['data_aug'] is 'heavy':
                self._aug_options['piecewise_affine'] = (0.01, 0.05)
            self._aug_options['input_channels'] = options['input_channels']
            # self._aug_options['crop_first'] = options['crop_first']
            self._aug = Augmenter(self._aug_options)

        # Load all data in memory, if requested
        if self.options['in_memory']:
            if self.options['mode'] in ['semantic_labels', 'instance_contours', 'semantic_contours', 'dwt_direction']:
                 self._preload_all_samples_with_sem_labels()
            else:
                self._preload_all_samples_with_inst_masks()

        np.random.seed(self.options['random_seed'])
        if self._phase == 'train_noval':
            # Train over the original training set
            self._train_ptr = 0
            # self.train_size = len(self._train_IDs) if _DBG_TRAIN_SET == -1 else _DBG_TRAIN_SET
            self.train_size = len(self._train_IDs)
            self._train_idx = np.arange(self.train_size)
            np.random.shuffle(self._train_idx)

        elif self._phase == 'val':
            # Validate over the validation split
            self._val_ptr = 0
            # self.val_size = len(self._val_IDs) if _DBG_TRAIN_SET == -1 else _DBG_TRAIN_SET
            self.val_size = len(self._val_IDs)
            self._val_idx = np.arange(self.val_size)
            np.random.shuffle(self._val_idx)

        elif self._phase == 'train_val':
            # Train over the training split, validate over the validation split
            self._train_ptr = 0
            # self.train_size = len(self._train_IDs) if _DBG_TRAIN_SET == -1 else _DBG_TRAIN_SET
            self.train_size = len(self._train_IDs)
            self._train_idx = np.arange(self.train_size)
            np.random.shuffle(self._train_idx)
            self._val_ptr = 0
            # self.val_size = len(self._val_IDs) if _DBG_TRAIN_SET == -1 else _DBG_TRAIN_SET
            self.val_size = len(self._val_IDs)
            self._val_idx = np.arange(self.val_size)
            np.random.shuffle(self._val_idx)

        # elif self._phase == 'test':
        # Test over the entire testing set
        self._test_ptr = 0
        # self.test_size = len(self._test_IDs) if _DBG_TRAIN_SET == -1 else _DBG_TRAIN_SET
        self.test_size = len(self._test_IDs)
        self._test_idx = np.arange(self.test_size)

    @property
    def unique_img_size(self):
        return self.options['unique_img_size']

    @property
    def root(self):
        return self._ds_root

    @property
    def pred_label_folder(self):
        return self._pred_label_folder

    @property
    def pred_label_folder_raw(self):
        return self._pred_label_folder_raw

    @property
    def bin_threshold(self):
        return self.options['bin_threshold']


    ###
    ### Input Samples and Labels Prep
    ###
    def prepare(self):
        """Do all the preprocessing needed before training/val/test samples can be used.
        """
        # Create paths files and load them back in
        self._create_ID_files()

        # Combine masks in a single label
        self._combine_masks()

        # Combine instance mask contours in a single label
        self._combine_contours()

        # Compute deep watershed transform maps (discretized distance transform + normalized gradients)
        self._generate_dwt_maps()

        # Load ID files
        self._load_ID_files()


    def _create_ID_files(self):
        """Create the ID files for each split of the dataset
        """
        # Instantiate sampler with the proper options
        sampler_options = {}
        for k, v in _DEFAULT_SAMPLING_OPTIONS.items():
            sampler_options[k] = self.options[k]
        smplr = Sampler(options=sampler_options)

        # Get train/val/test splits
        self._train_IDs = smplr.train_IDs.copy()
        self._val_IDs = smplr.val_IDs.copy()
        self._test_IDs = next(os.walk(self._test_folder))[1]

        # Save ID files
        with open(self._train_IDs_file, 'w') as f:
            for ID in self._train_IDs:
                f.write('{}\n'.format(str(ID)))
        with open(self._val_IDs_file, 'w') as f:
            for ID in self._val_IDs:
                f.write('{}\n'.format(str(ID)))
        with open(self._test_IDs_file, 'w') as f:
            for ID in self._test_IDs:
                f.write('{}\n'.format(str(ID)))
    
    
    def _load_ID_files(self):
        """Load the ID files and build the file paths associated with those IDs
        Returns:
              True if ID files were loaded, False if ID files weren't found
        """
        ext = '.npy' if self.options['mode'] == 'dwt_direction' else '.png'
        if self._phase == 'train_noval':
            # Train over the original training set
            if not os.path.exists(self._train_IDs_file) or not os.path.exists(self._val_IDs_file):
                return False
            with open(self._train_IDs_file, 'r') as f:
                self._train_IDs = f.readlines()
                self._train_IDs = [ID.rstrip() for ID in self._train_IDs]
            with open(self._val_IDs_file, 'r') as f:
                self._val_IDs = f.readlines()
                self._val_IDs = [ID.rstrip() for ID in self._val_IDs]
            self._images_train_path = [self._train_folder + '/' + idx + '/images/' + idx + '.png' for idx in self._train_IDs]
            self._images_train_path +=  [self._val_folder + '/' + idx + '/images/' + idx + '.png' for idx in self._val_IDs]
            self._labels_train_path = [self._train_folder + '/' + idx + self._label_folder + idx + ext for idx in self._train_IDs]
            self._labels_train_path += [self._val_folder + '/' + idx + self._label_folder + idx + ext for idx in self._val_IDs]
            self._train_IDs += self._val_IDs
            if _DBG_TRAIN_VAL_TEST_SETS != -1: # Debug mode only
                self._images_train_path = self._images_train_path[0:_DBG_TRAIN_VAL_TEST_SETS]
                self._labels_train_path = self._labels_train_path[0:_DBG_TRAIN_VAL_TEST_SETS]
                self._train_IDs = self._train_IDs[0:_DBG_TRAIN_VAL_TEST_SETS]

        elif self._phase == 'train_val':
            # Train over the training split, validate over the validation split
            if not os.path.exists(self._train_IDs_file):
                return False
            with open(self._train_IDs_file, 'r') as f:
                self._train_IDs = f.readlines()
                self._train_IDs = [ID.rstrip() for ID in self._train_IDs]
            self._images_train_path = [self._train_folder + '/' + idx + '/images/' + idx + '.png' for idx in self._train_IDs]
            self._labels_train_path = [self._train_folder + '/' + idx + self._label_folder + idx + ext for idx in self._train_IDs]
            if not os.path.exists(self._val_IDs_file):
                return False
            with open(self._val_IDs_file, 'r') as f:
                self._val_IDs = f.readlines()
                self._val_IDs = [ID.rstrip() for ID in self._val_IDs]
            self._images_val_path = [self._val_folder + '/' + idx + '/images/' + idx + '.png' for idx in self._val_IDs]
            self._labels_val_path = [self._val_folder + '/' + idx + self._label_folder + idx + ext for idx in self._val_IDs]
            if _DBG_TRAIN_VAL_TEST_SETS != -1: # Debug mode only
                self._images_train_path = self._images_train_path[0:_DBG_TRAIN_VAL_TEST_SETS]
                self._labels_train_path = self._labels_train_path[0:_DBG_TRAIN_VAL_TEST_SETS]
                self._train_IDs = self._train_IDs[0:_DBG_TRAIN_VAL_TEST_SETS]
                self._images_val_path = self._images_val_path[0:_DBG_TRAIN_VAL_TEST_SETS]
                self._labels_val_path = self._labels_val_path[0:_DBG_TRAIN_VAL_TEST_SETS]
                self._val_IDs = self._val_IDs[0:_DBG_TRAIN_VAL_TEST_SETS]

        elif self._phase == 'val':
            # Validate over the validation split
            if not os.path.exists(self._val_IDs_file):
                return False
            with open(self._val_IDs_file, 'r') as f:
                self._val_IDs = f.readlines()
                self._val_IDs = [ID.rstrip() for ID in self._val_IDs]
            self._images_val_path = [self._val_folder + '/' + idx + '/images/' + idx + '.png' for idx in self._val_IDs]
            self._labels_val_path = [self._val_folder + '/' + idx + self._label_folder + idx + ext for idx in self._val_IDs]
            if _DBG_TRAIN_VAL_TEST_SETS != -1: # Debug mode only
                self._images_val_path = self._images_val_path[0:_DBG_TRAIN_VAL_TEST_SETS]
                self._labels_val_path = self._labels_val_path[0:_DBG_TRAIN_VAL_TEST_SETS]
                self._val_IDs = self._val_IDs[0:_DBG_TRAIN_VAL_TEST_SETS]

        # elif self._phase == 'test':
        # Test over the entire testing set
        if not os.path.exists(self._test_IDs_file):
            return False
        with open(self._test_IDs_file, 'r') as f:
            self._test_IDs = f.readlines()
            self._test_IDs = [ID.rstrip() for ID in self._test_IDs]
        self._images_test_path = [self._test_folder + '/' + idx + '/images/' + idx + '.png' for idx in self._test_IDs]
        if _DBG_TRAIN_VAL_TEST_SETS != -1:  # Debug mode only
            self._images_test_path = self._images_test_path[0:_DBG_TRAIN_VAL_TEST_SETS]

        return True

    
    def _combine_masks(self):
        """Combine masks into a single label.
        A PNG file is created in the samples's base folder.
        """
        if self._phase == 'train_noval':
            IDs = self._train_IDs
        else:
            IDs = self._train_IDs + self._val_IDs

        assert(self._train_folder == self._val_folder) # Code below will fail otherwise

        for ID in tqdm(IDs, total=len(IDs), ascii=True, ncols=100, desc='Combining masks into single label'):
            path = self._train_folder + '/' + ID

            # Build semantic label out of instance masks
            masks = load_inst_masks(path + '/masks')  # np array [#, H, W]
            label = inst_masks_to_sem_label(masks)  # (H, W)

            # Save the mask in PNG format to mask folder
            if not os.path.exists(path + '/labels/'):
                os.makedirs(path + '/labels/')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                output_file = path + '/labels/' + ID + '.png'
                if os.path.exists(output_file):
                    os.remove(output_file)
                imsave(output_file, label)

            # Combine image with label (for error analysis)
            # combined = self._combine_image_with_label(path + '/images/' + ID + '.png', path + '/labels/' + ID + '.png')
            # if not os.path.exists(path + '/combined/'):
            #     os.makedirs(path + '/combined/')
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            #     imsave(path + '/combined/' + ID + '.png', combined)


    def _combine_contours(self):
        """Combine instance mask contours in a single label
        A PNG file is created in the samples's base folder.
        """
        if self._phase == 'train_noval':
            IDs = self._train_IDs
        else:
            IDs = self._train_IDs + self._val_IDs

        assert(self._train_folder == self._val_folder) # Code below will fail otherwise

        thickness = self.options['contour_thickness']
        desc='Combining {}px-wide mask contours into single label'.format(thickness)
        for ID in tqdm(IDs, total=len(IDs), ascii=True, ncols=100, desc=desc):
            path = self._train_folder + '/' + ID

            # Build semantic label out of instance masks
            masks = load_inst_masks(path + '/masks')  # np array [#, H, W]
            label = inst_masks_contours_to_label(masks, thickness)  # (H, W)

            # Save the mask in PNG format to mask folder
            sub_folder = '/contours_{}px/'.format(thickness)
            if not os.path.exists(path + sub_folder):
                os.makedirs(path + sub_folder)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                output_file = path + sub_folder + ID + '.png'
                if os.path.exists(output_file):
                    os.remove(output_file)
                imsave(output_file, label)

            # Combine image with label (for error analysis)
            # img_path = path + '/images/' + ID + '.png'
            # label_path = path + '/contours/' + ID + '.png'
            # combined_path = path + '/combined/' + ID + '.png'
            # if not os.path.exists(path + '/combined/'):
            #     os.makedirs(path + '/combined/')
            # else:
            #     img_path = combined_path
            # combined = self._combine_image_with_label(img_path, label_path, color=(255,0,0))
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            #     imsave(combined_path, combined)

    def _generate_dwt_maps(self):
        """Compute deep watershed transform maps (discretized distance transform + normalized gradients).
        Four additiona PNG files are created in the training inmages individual folders.
        Two are for the discretized distance transform (aka, watershed energy) using different discretization bins).
        Two more contain the "direction" vector (one per component, ie x or y).
            Original code:
            In the original MATLAB code, they refer to distance transform as "depth_map" and angular vector as "dir_map"
        """
        if self._phase == 'train_noval':
            IDs = self._train_IDs
        else:
            IDs = self._train_IDs + self._val_IDs

        assert(self._train_folder == self._val_folder) # Code below will fail otherwise

        desc='Generating DWT angular and energy maps'
        for ID in tqdm(IDs, total=len(IDs), ascii=True, ncols=100, desc=desc):
            path = self._train_folder + '/' + ID

            # Load instance masks and the semantic label
            masks = load_inst_masks(path + '/masks')  # np array [#, H, W]

            # Allocate accumulators
            dist_fine = np.zeros(masks[0].shape, dtype=np.float32)
            dist_coarse = dist_fine.copy()

            # Compute Euclidean distances transform for each mask and combine the results in a single image
            # https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.ndimage.distance_transform_edt.html
            for mask in masks:
                dist_fine += ndi.distance_transform_edt(mask)

            # Measure gradients of distance transform image and scale them
            # https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.ndimage.sobel.html
            # See https://stackoverflow.com/questions/26827391/why-the-orientation-of-image-convolution-operators-are-not-intuitive
            sobel_gradient_magnitude = ndi.generic_gradient_magnitude(dist_fine, ndi.sobel)
            safe_divide_sobel_gradient_magnitude = sobel_gradient_magnitude.copy()
            safe_divide_sobel_gradient_magnitude[safe_divide_sobel_gradient_magnitude == 0] = 1
            normalized_gradient_x = ndi.sobel(dist_fine, axis=1) / safe_divide_sobel_gradient_magnitude
            normalized_gradient_y = ndi.sobel(dist_fine, axis=0) / safe_divide_sobel_gradient_magnitude
            normalized_gradients = np.concatenate([np.expand_dims(normalized_gradient_y, axis=-1), np.expand_dims(normalized_gradient_x, axis=-1)], axis=-1)

            # Zero-out any gradient information that is not part of the semantic segmentation
            bins_fine = [0, 1, 2, 4, 6, 9, 12, 16, np.inf]
            bins_coarse = [bin*2 for bin in bins_fine]
            for idx in range(len(bins_coarse)-1):
                dist_coarse[(dist_fine > bins_coarse[idx]) & (dist_fine <= bins_coarse[idx + 1])] = bins_coarse[idx]
            for idx in range(len(bins_fine)-1):
                dist_fine[(dist_fine > bins_fine[idx]) & (dist_fine <= bins_fine[idx+1])] = bins_fine[idx]
            dist_coarse, dist_fine = dist_coarse.astype(np.uint8), dist_fine.astype(np.uint8)

            # Save the discretized distance transforms and normalized gradients in PNG format
            folders = ['/distance_coarse/', '/distance_fine/', '/normed_gradient_x/', '/normed_gradient_y/']
            images = [dist_coarse, dist_fine, normalized_gradient_x, normalized_gradient_y]
            for folder, image in zip(folders, images):
                if not os.path.exists(path + folder):
                    os.makedirs(path + folder)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    output_file = path + folder + ID + '.png'
                    if os.path.exists(output_file):
                        os.remove(output_file)
                    imsave(output_file, image)

            # Save the normalized gradients in NPY format
            folder = '/normed_gradients/'
            if not os.path.exists(path + folder):
                os.makedirs(path + folder)
            output_file = path + folder + ID + '.npy'
            if os.path.exists(output_file):
                os.remove(output_file)
            np.save(output_file, normalized_gradients)


    ###
    ### Batch Management
    ###
    def _preload_all_samples_with_sem_labels(self):
        """Preload all samples (input sampleRGB image + associated semantic label) in memory
        In training and validation mode, there is a label; in testimg mode, there isn't.
        """
        if self._phase == 'train_noval':
            self._images_train, self._labels_train = [], []
            with tqdm(total=len(self._images_train_path), desc="Loading train samples & semantic labels", ascii=True, ncols=100) as pbar:
                for image_path, label_path in zip(self._images_train_path, self._labels_train_path):
                    pbar.update(1)
                    image, label = self._load_sample(image_path, label_path)
                    self._images_train.append(image)
                    self._labels_train.append(label)

        elif self._phase == 'train_val':
            self._images_train, self._labels_train = [], []
            with tqdm(total=len(self._images_train_path), desc="Loading train samples & semantic labels", ascii=True, ncols=100) as pbar:
                for image_path, label_path in zip(self._images_train_path, self._labels_train_path):
                    pbar.update(1)
                    image, label = self._load_sample(image_path, label_path)
                    self._images_train.append(image)
                    self._labels_train.append(label)
            self._images_val, self._labels_val = [], []
            with tqdm(total=len(self._images_val_path), desc="Loading val samples & semantic labels", ascii=True, ncols=100) as pbar:
                for image_path, label_path in zip(self._images_val_path, self._labels_val_path):
                    pbar.update(1)
                    image, label = self._load_sample(image_path, label_path, never_aug=True)
                    self._images_val.append(image)
                    self._labels_val.append(label)

        elif self._phase == 'val':
            self._images_val, self._labels_val = [], []
            with tqdm(total=len(self._images_val_path), desc="Loading val samples & semantic labels", ascii=True, ncols=100) as pbar:
                for image_path, label_path in zip(self._images_val_path, self._labels_val_path):
                    pbar.update(1)
                    image, label = self._load_sample(image_path, label_path, never_aug=True)
                    self._images_val.append(image)
                    self._labels_val.append(label)

        elif self._phase == 'test':
            self._images_test = []
            with tqdm(total=len(self._images_test_path), desc="Loading test samples", ascii=True, ncols=100) as pbar:
                for image_path in self._images_test_path:
                    pbar.update(1)
                    self._images_test.append(self._load_sample(image_path, never_aug=True))

    def _preload_all_samples_with_inst_masks(self):
        """Preload all samples (input sampleRGB image + associated bboxes and instance masks) in memory
        In training and validation mode, there are instance masks; in testimg mode, there aren't.
        """
        if self._phase == 'train_noval':
            self._images_train, self._inst_masks_train, self._inst_bboxes_train = [], [], []
            with tqdm(total=len(self._images_train_path), desc="Loading train samples & instance masks", ascii=True, ncols=100) as pbar:
                for image_path in self._images_train_path:
                    pbar.update(1)
                    image = self._load_sample(image_path)
                    self._images_train.append(image)
                    masks_folder = os.path.dirname(image_path).replace('images', 'masks')
                    masks = load_inst_masks(masks_folder) # np array [#, H, W]
                    self._inst_masks_train.append(masks) # list([#, height, width])
                    if self.options['compute_bboxes']:
                        bboxes = extract_bboxes(masks)
                        self._inst_bboxes_train.append(bboxes) # list([#, (y1, x1, y2, x2)])

        elif self._phase == 'train_val':

            self._images_train, self._inst_masks_train, self._inst_bboxes_train = [], [], []
            with tqdm(total=len(self._images_train_path), desc="Loading train samples & instance masks", ascii=True, ncols=100) as pbar:
                for image_path in self._images_train_path:
                    pbar.update(1)
                    image = self._load_sample(image_path)
                    self._images_train.append(image)
                    masks_folder = os.path.dirname(image_path).replace('images', 'masks')
                    masks = load_inst_masks(masks_folder) # np array [#, H, W]
                    self._inst_masks_train.append(masks) # list([#, height, width])
                    if self.options['compute_bboxes']:
                        bboxes = extract_bboxes(masks)
                        self._inst_bboxes_train.append(bboxes) # list([#, (y1, x1, y2, x2)])

            self._images_val, self._inst_masks_val, self._inst_bboxes_val = [], [], []
            with tqdm(total=len(self._images_val_path), desc="Loading val samples & instance masks", ascii=True, ncols=100) as pbar:
                for image_path in self._images_val_path:
                    pbar.update(1)
                    image = self._load_sample(image_path, never_aug=True)
                    self._images_val.append(image)
                    masks_folder = os.path.dirname(image_path).replace('images', 'masks')
                    masks = load_inst_masks(masks_folder) # np array [#, H, W]
                    self._inst_masks_val.append(masks) # list([#, height, width])
                    if self.options['compute_bboxes']:
                        bboxes = extract_bboxes(masks)
                        self._inst_bboxes_val.append(bboxes) # list([#, (y1, x1, y2, x2)])

        elif self._phase == 'val':

            self._images_val, self._inst_masks_val, self._inst_bboxes_val = [], [], []
            with tqdm(total=len(self._images_val_path), desc="Loading val samples & instance masks", ascii=True, ncols=100) as pbar:
                for image_path in self._images_val_path:
                    pbar.update(1)
                    image = self._load_sample(image_path, never_aug=True)
                    self._images_val.append(image)
                    masks_folder = os.path.dirname(image_path).replace('images', 'masks')
                    masks = load_inst_masks(masks_folder) # np array [#, H, W]
                    self._inst_masks_val.append(masks) # list([#, height, width])
                    if self.options['compute_bboxes']:
                        bboxes = extract_bboxes(masks)
                        self._inst_bboxes_val.append(bboxes) # list([#, (y1, x1, y2, x2)])

        elif self._phase == 'test':

            self._images_test = []
            with tqdm(total=len(self._images_test_path), desc="Loading test samples", ascii=True, ncols=100) as pbar:
                for image_path in self._images_test_path:
                    pbar.update(1)
                    self._images_test.append(self._load_sample(image_path, never_aug=True))

    def _load_sample(self, image_path=None, label_path=None, never_aug=False):
        """Load a propertly formatted sample (input sampleRGB image + associated label, if any label)
        Args:
            image_path: Path to RGB image, if any
            label_path: Path to label, if any label
            never_aug: Never apply augmentation to val and test samples
        Returns:
            image: RGB image in format [H, W, 3]
            label: Label in format [W, H, 1], if any label
        """
        # Read in RGB image, if any
        if image_path:
            image = imread(image_path)

            # Some input images are 32bit. Just discard the last dimension in that case.
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = image[:,:,:3]

            # Concatenate the semantic label onto the RGB image
            if self.options['input_channels'] == 4:
                # if self.options['mode'] == 'instance_contours':
                #     extra_channel_path = image_path.replace('images', 'labels')
                # elif self.options['mode'] == 'semantic_contours':
                #     extra_channel_path = image_path.replace('images', 'pred_sem_label')
                if self._phase in ['train_noval', 'train_val', 'val']:
                    extra_channel_path = image_path.replace('images', 'labels')
                elif self._phase == 'test':
                    extra_channel_path = image_path.replace('images', 'pred_sem_label')
                extra_channel = imread(extra_channel_path)
                extra_channel = np.expand_dims(extra_channel, axis=-1)
                assert (len(extra_channel.shape) == 3 and extra_channel.shape[2] == 1)
                assert (extra_channel.shape[0] == image.shape[0] and extra_channel.shape[1] == image.shape[1])
                if self.options['mode'] == 'dwt_direction':
                    # Zero-out anything in the RGB image that isn't part of the semantic segmentation
                    sem_seg_mask = np.where(extra_channel > 0, 1, 0)
                    image = image * sem_seg_mask
                image = np.concatenate([image, extra_channel], axis=-1)

        # Read in label, if any
        if label_path:
            if self.options['mode'] == 'dwt_direction':
                # label_path_x, label_path_y = label_path.format('x'), label_path.format('y')
                label = np.load(label_path)
                # label_x, label_y = np.expand_dims(label_x, axis=-1), np.expand_dims(label_y, axis=-1)
                # label = np.concatenate([label_y, label_x], axis=-1)
                assert (len(label.shape) == 3 and label.shape[2] == 2)
            else:
                label = imread(label_path)
                label = np.expand_dims(label, axis=-1)
                assert (len(label.shape) == 3 and label.shape[2] == 1)
        else:
            label = None

        # Use augmentation, if requested
        # if self.options['in_memory'] is False and self.options['data_aug'] and never_aug is False:
        #     if label is None:
        #         assert(self._aug_options['aug_labels'] is False)
        #         aug_image = self._aug.augment([image])
        #         image = aug_image[0]
        #     else:
        #         aug_image, aug_label = self._aug.augment([image], [label])
        #         image, label = aug_image[0], aug_label[0]

        # Return image and/or label
        if label_path:
            if image_path:
                return image, label
            else:
                return label
        else:
            if image_path:
                return image


    def _augment_sample(self, image=None, label=None):
        """Augment an in-memory sample (input sampleRGB image + associated label, if any label)
        Args:
            image: RGB or RGB+4th channel image
            label: Label, if any label
        Returns:
            image: Augmented image in format [H, W, 3] or [H, W, 4]
            label: Label in format [W, H, 1], if any label
        """
        # Use augmentation, if requested
        assert(self.options['data_aug'] in ['basic', 'heavy'])
        if label is None:
            assert(self._aug_options['aug_labels'] is False)
            aug_image = self._aug.augment([image])
            image = aug_image[0]
        else:
            aug_image, aug_label = self._aug.augment([image], [label])
            image, label = aug_image[0], aug_label[0]

        # Return image and label
        return image, label


    def next_batch_sem_labels(self, batch_size, split='train', with_IDs=False):
        """Get next batch of image (path) and labels
        In training and validation mode, there is a label; in testimg mode, there isn't.
        In 'val_predpaths' mode, also return a destination file where to save a label prediction.
        Args:
            batch_size: Size of the batch
            split: Possible options: 'train', 'val', 'test', or 'val_predpaths'
            with_IDs: If True, also return IDs
        Returns in training and validation mode:
            images: Batch of RGB images in format [batch_size, H, W, 3]
            labels: Batch of labels in format [batch_size, W, H, 1] or file paths to predicted label
        Returns in testing:
            images: Batch of RGB images in format [batch_size, H, W, 3]
            output_files: List of output file names that match the input file names
        """
        # assert (self.options['in_memory'] is False)  # Only option supported at this point
        if split == 'train':
            assert(self._phase == 'train_val' or self._phase == 'train_noval')
            if self._train_ptr + batch_size < self.train_size:
                idx = np.array(self._train_idx[self._train_ptr:self._train_ptr + batch_size])
                new_ptr = self._train_ptr + batch_size
            else:
                old_idx = np.array(self._train_idx[self._train_ptr:])
                np.random.shuffle(self._train_idx)
                new_ptr = (self._train_ptr + batch_size) % self.train_size
                idx = np.concatenate((old_idx, np.array(self._train_idx[:new_ptr])))

            images, labels, IDs = self._get_train_samples_with_sem_labels(idx)

            self._train_ptr = new_ptr

            if with_IDs:
                if batch_size > 1 and self.options['crop_first'] is None:
                    return images, labels, IDs
                else:
                    return np.asarray(images), np.asarray(labels), np.asarray(IDs)
            else:
                if batch_size > 1 and self.options['crop_first'] is None:
                    return images, labels
                else:
                    return np.asarray(images), np.asarray(labels)

        elif split == 'val':
            assert(self._phase == 'train_val' or self._phase == 'val')
            if self._val_ptr + batch_size < self.val_size:
                idx = np.array(self._val_idx[self._val_ptr:self._val_ptr + batch_size])
                new_ptr = self._val_ptr + batch_size
            else:
                old_idx = np.array(self._val_idx[self._val_ptr:])
                # np.random.shuffle(self._val_idx)
                new_ptr = (self._val_ptr + batch_size) % self.val_size
                idx = np.concatenate((old_idx, np.array(self._val_idx[:new_ptr])))

            images, labels, IDs = self._get_val_samples_with_sem_labels(idx)

            self._val_ptr = new_ptr

            if with_IDs:
                if batch_size > 1 and self.options['crop_first'] is None:
                    return images, labels, IDs
                else:
                    return np.asarray(images), np.asarray(labels), np.asarray(IDs)
            else:
                if batch_size > 1 and self.options['crop_first'] is None:
                    return images, labels
                else:
                    return np.asarray(images), np.asarray(labels)

        elif split == 'val_with_preds':
            assert(self._phase == 'train_val' or self._phase == 'val')
            if self._val_ptr + batch_size < self.val_size:
                idx = np.array(self._val_idx[self._val_ptr:self._val_ptr + batch_size])
                new_ptr = self._val_ptr + batch_size
            else:
                old_idx = np.array(self._val_idx[self._val_ptr:])
                # np.random.shuffle(self._val_idx)
                new_ptr = (self._val_ptr + batch_size) % self.val_size
                idx = np.concatenate((old_idx, np.array(self._val_idx[:new_ptr])))

            images, labels, pred_labels, IDs = self._get_val_samples_with_sem_labels_and_preds(idx)

            self._val_ptr = new_ptr

            if with_IDs:
                if batch_size > 1:
                    return images, labels, pred_labels, IDs
                else:
                    return np.asarray(images), np.asarray(labels), np.asarray(pred_labels), np.asarray(IDs)
            else:
                if batch_size > 1:
                    return images, labels, pred_labels
                else:
                    return np.asarray(images), np.asarray(labels), np.asarray(pred_labels)

        elif split == 'val_with_pred_paths':
            assert(self._phase == 'train_val' or self._phase == 'val')
            if self._val_ptr + batch_size < self.val_size:
                idx = np.array(self._val_idx[self._val_ptr:self._val_ptr + batch_size])
                new_ptr = self._val_ptr + batch_size
            else:
                old_idx = np.array(self._val_idx[self._val_ptr:])
                # np.random.shuffle(self._val_idx)
                new_ptr = (self._val_ptr + batch_size) % self.val_size
                idx = np.concatenate((old_idx, np.array(self._val_idx[:new_ptr])))

            pred_paths = []
            images, labels, IDs = self._get_val_samples_with_sem_labels(idx)
            for l in idx:
                pred_paths.append(self._images_val_path[l].replace('/images/', self._pred_label_folder))

            self._val_ptr = new_ptr

            if with_IDs:
                if batch_size > 1:
                    return images, labels, pred_paths, IDs
                else:
                    return np.asarray(images), np.asarray(labels), pred_paths, np.asarray(IDs)
            else:
                if batch_size > 1:
                    return images, labels, pred_paths
                else:
                    return np.asarray(images), np.asarray(labels), pred_paths

        elif split == 'test':
            if self._test_ptr + batch_size < self.test_size:
                new_ptr = self._test_ptr + batch_size
                idx = list(range(self._test_ptr, self._test_ptr + batch_size))
            else:
                new_ptr = (self._test_ptr + batch_size) % self.test_size
                idx = list(range(self._test_ptr, self.test_size)) + list(range(0, new_ptr))

            images, IDs = [], []
            for l in idx:
                image = self._images_test[l] if self.options['in_memory'] else self._load_sample(self._images_test_path[l])
                images.append(image)
                IDs.append(self._test_IDs[l])

            self._test_ptr = new_ptr

            if with_IDs:
                if batch_size > 1:
                    return images, IDs
                else:
                    return np.asarray(images), np.asarray(IDs)
            else:
                if batch_size > 1:
                    return images
                else:
                    return np.asarray(images)

        elif split == 'test_with_pred_paths':
            if self._test_ptr + batch_size < self.test_size:
                new_ptr = self._test_ptr + batch_size
                idx = list(range(self._test_ptr, self._test_ptr + batch_size))
            else:
                new_ptr = (self._test_ptr + batch_size) % self.test_size
                idx = list(range(self._test_ptr, self.test_size)) + list(range(0, new_ptr))

            images, pred_paths, IDs = [], [], []
            for l in idx:
                image = self._images_test[l] if self.options['in_memory'] else self._load_sample(
                    self._images_test_path[l])
                images.append(image)
                pred_paths.append(self._images_test_path[l].replace('/images/', self._pred_label_folder))
                IDs.append(self._test_IDs[l])

            self._test_ptr = new_ptr

            if with_IDs:
                if batch_size > 1:
                    return images, pred_paths, IDs
                else:
                    return np.asarray(images), pred_paths, np.asarray(IDs)
            else:
                if batch_size > 1:
                    return images, pred_paths
                else:
                    return np.asarray(images), pred_paths

        else:
            return None, None

    def next_batch_inst_masks(self, batch_size, split='train', with_IDs=False):
        """Get next batch of images (path) and associated bboxes and instance masks
        In training and validation mode, there are instance masks; in testimg mode, there aren't.
        In 'val_predpaths' mode, also return a destination folder where to save predicted instance masks.
        Args:
            batch_size: Size of the batch
            split: Possible options: 'train', 'val', 'test', or 'val_predpaths'
            with_IDs: If True, also return IDs
        Returns in training and validation mode:
            images: Batch of RGB images in format [batch_size, H, W, 3]
            labels: Batch of labels in format [batch_size, W, H, 1] or file paths to predicted label
        Returns in testing:
            images: Batch of RGB images in format [batch_size, H, W, 3]
            pred_folder: List of output folders where to save the predicted instance masks
        """
        assert(split in ['train', 'val', 'val_with_preds', 'val_with_pred_paths', 'test', 'test_with_pred_paths'])

        # Come up with list of indices to load
        if split == 'train':
            assert(self._phase == 'train_val' or self._phase == 'train_noval')
            if self._train_ptr + batch_size < self.train_size:
                idx = np.array(self._train_idx[self._train_ptr:self._train_ptr + batch_size])
                new_ptr = self._train_ptr + batch_size
            else:
                old_idx = np.array(self._train_idx[self._train_ptr:])
                np.random.shuffle(self._train_idx)
                new_ptr = (self._train_ptr + batch_size) % self.train_size
                idx = np.concatenate((old_idx, np.array(self._train_idx[:new_ptr])))

        elif split in ['val', 'val_with_preds', 'val_with_pred_paths']:
            assert(self._phase == 'train_val' or self._phase == 'val')
            if self._val_ptr + batch_size < self.val_size:
                idx = np.array(self._val_idx[self._val_ptr:self._val_ptr + batch_size])
                new_ptr = self._val_ptr + batch_size
            else:
                old_idx = np.array(self._val_idx[self._val_ptr:])
                # np.random.shuffle(self._val_idx)
                new_ptr = (self._val_ptr + batch_size) % self.val_size
                idx = np.concatenate((old_idx, np.array(self._val_idx[:new_ptr])))

        elif split in ['test', 'test_with_pred_paths']:
            assert (self._phase == 'test')
            if self._test_ptr + batch_size < self.test_size:
                new_ptr = self._test_ptr + batch_size
                idx = list(range(self._test_ptr, self._test_ptr + batch_size))
            else:
                new_ptr = (self._test_ptr + batch_size) % self.test_size
                idx = list(range(self._test_ptr, self.test_size)) + list(range(0, new_ptr))

        # Load samples
        if split == 'train':

            images, masks, bboxes, IDs = self._get_train_samples_with_inst_masks(idx)
            self._train_ptr = new_ptr
            if with_IDs:
                return np.asarray(images), masks, bboxes, IDs
            else:
                return np.asarray(images), masks, bboxes

        elif split == 'val':

            images, masks, bboxes, IDs = self._get_val_samples_with_inst_masks(idx)
            self._val_ptr = new_ptr
            if with_IDs:
                return np.asarray(images), masks, bboxes, IDs
            else:
                return np.asarray(images), masks, bboxes

        elif split == 'val_with_preds':

            images, masks, bboxes, pred_masks, pred_bboxes = self._get_val_samples_with_inst_masks_and_preds(idx)
            self._val_ptr = new_ptr
            return np.asarray(images), masks, bboxes, pred_masks, pred_bboxes

        elif split == 'val_with_pred_paths':

            images, masks, bboxes = self._get_val_samples_with_inst_masks(idx)
            pred_paths = []
            for l in idx:
                pred_paths.append(os.path.dirname(self._images_val_path[l]).replace('images', 'pred_inst_masks'))
            self._val_ptr = new_ptr
            return np.asarray(images), masks, bboxes, pred_paths

        elif split == 'test':

            images = []
            for l in idx:
                image = self._images_test[l] if self.options['in_memory'] else self._load_sample(self._images_test_path[l])
                images.append(image)
            self._test_ptr = new_ptr
            return np.asarray(images)

        elif split == 'test_with_pred_paths':

            images, pred_paths = [], []
            for l in idx:
                image = self._images_test[l] if self.options['in_memory'] else self._load_sample(self._images_test_path[l])
                images.append(image)
                pred_paths.append(os.path.dirname(self._images_test_path[l]).replace('images', 'pred_inst_masks'))
            self._test_ptr = new_ptr
            return np.asarray(images), pred_paths


    ###
    ### Semantic labels
    ###

    def _get_train_samples_with_sem_labels(self, idx):
        """Get training images with associated semantic labels
        Args:
            idx: List of sample indices to return
        Returns:
            images: List of RGB images in format list([H, W, 3])
            labels: List of labels in format list([H, W, 1])
        """
        images, labels, IDs = [], [], []
        for l in idx:
            if self.options['in_memory']:
                image, label = self._images_train[l], self._labels_train[l]
            else:
                image, label = self._load_sample(self._images_train_path[l], self._labels_train_path[l])

            # Crop images and labels to a fixed size, if requested
            if self.options['crop_first']:
                h, w = image.shape[:2]
                h_max, w_max = self.options['crop_first']
                assert (h >= h_max and w >= w_max)
                max_y_offset, max_x_offset = h - h_max, w - w_max
                if max_y_offset > 0 or max_x_offset > 0:
                    y_offset = np.random.randint(max_y_offset + 1)
                    x_offset = np.random.randint(max_x_offset + 1)
                    image = image[y_offset:y_offset + h_max, x_offset:x_offset + w_max]
                    label = label[y_offset:y_offset + h_max, x_offset:x_offset + w_max]

            # Augment image and label if requested
            if self.options['data_aug'] in ['basic', 'heavy']:
                image, label = self._augment_sample(image, label)

            images.append(image)
            labels.append(label)
            IDs.append(self._train_IDs[l])

        return images, labels, IDs


    def _get_val_samples_with_sem_labels(self, idx):
        """Get validation images with associated semantic labels
        Args:
            idx: List of sample indices to return
        Returns:
            images: List of RGB images in format list([H, W, 3])
            labels: List of labels in format list([H, W, 1])
        """
        images, labels, IDs = [], [], []
        for l in idx:
            if self.options['in_memory']:
                image, label = self._images_val[l], self._labels_val[l]
            else:
                image, label = self._load_sample(self._images_val_path[l], self._labels_val_path[l], never_aug=True)

            # Crop images and labels to a fixed size, if requested
            # Why do this with validation data? Because we are currently stuck with having the same
            # batch size between the training and validation data sets.  Since our images all have
            # different sizes, if we want to batch validation samples, they must all have the same size
            if self.options['crop_first']:
                h, w = image.shape[:2]
                h_max, w_max = self.options['crop_first']
                assert (h >= h_max and w >= w_max)
                max_y_offset, max_x_offset = h - h_max, w - w_max
                if max_y_offset > 0 or max_x_offset > 0:
                    y_offset = np.random.randint(max_y_offset + 1)
                    x_offset = np.random.randint(max_x_offset + 1)
                    image = image[y_offset:y_offset + h_max, x_offset:x_offset + w_max]
                    label = label[y_offset:y_offset + h_max, x_offset:x_offset + w_max]

            images.append(image)
            labels.append(label)
            IDs.append(self._val_IDs[l])

        return images, labels, IDs

    def _get_val_samples_with_sem_labels_and_preds(self, idx):
        """Get validation images with associated semantic labels and predictions
        Args:
            idx: List of sample indices to return
        Returns:
            images: List of RGB images in format list([H, W, 3])
            labels: List of labels in format list([H, W, 1])
            pred_labels: List of predicted labels in format list([H, W, 1])
        """
        images, labels, IDs = self._get_val_samples_with_sem_labels(idx)
        pred_labels = []
        for l in idx:
            pred_label_path = self._images_val_path[l].replace('/images/', self._pred_label_folder)
            if os.path.exists(pred_label_path):
                pred_label = imread(pred_label_path)
                pred_label = np.expand_dims(pred_label, axis=-1)
                if self.options['crop_first']:
                    h, w = pred_label.shape[:2]
                    h_max, w_max = self.options['crop_first']
                    assert (h >= h_max and w >= w_max)
                    max_y_offset, max_x_offset = h - h_max, w - w_max
                    if max_y_offset > 0 or max_x_offset > 0:
                        y_offset = np.random.randint(max_y_offset + 1)
                        x_offset = np.random.randint(max_x_offset + 1)
                        pred_label = pred_label[y_offset:y_offset + h_max, x_offset:x_offset + w_max]
                pred_labels.append(pred_label)
        return images, labels, pred_labels, IDs

    def _get_test_samples_with_pred_sem_labels(self, idx):
        """Get test images with predicted semantic labels
        Args:
            idx: List of sample indices to return
        Returns:
            images: List of RGB images in format list([H, W, 3])
            pred_labels: List of predicted labels in format list([H, W, 1])
        """
        images, pred_labels, IDs = [], [], []
        for l in idx:
            image = self._images_test[l] if self.options['in_memory'] else self._load_sample(self._images_test_path[l], never_aug=True)
            images.append(image)
            pred_label_path = self._images_test_path[l].replace('/images/', self._pred_label_folder)
            if os.path.exists(pred_label_path):
                pred_label = imread(pred_label_path)
                pred_label = np.expand_dims(pred_label, axis=-1)
                pred_labels.append(pred_label)
            IDs.append(self._test_IDs[l])
        return images, pred_labels, IDs

    def get_rand_samples_with_sem_labels(self, num_samples, mode='val', as_list=True, return_IDs=False, deterministic=False):
        """Get a few (or all) random (or ordered) samples from the dataset.
        Used for debugging purposes (testing how the model is improving over time, for instance).
        If sampling from the training/validation set, there is a label; otherwise, there isn't.
        Note that this doesn't return a valid np array if the images don't have the same size.
        Args:
            num_samples: Number of samples to return
            mode: Possible options: 'train', 'val', 'test'
            as_list: Return as list or np array?
            return_IDs: If True, also return ID of the sample
            deterministic: If True, return samples in order
        Returns in training and validation mode:
            images: Batch of RGB images in format [num_samples, H, W, 3] or list([H, W, 3])
            labels: Batch of labels in format [num_samples, W, H, 1]
        Returns in testing:
            images: Batch of RGB images in format [num_samples, H, W, 3]
            output_files: List of output file names that match the input file names
        """
        if num_samples == 0:
            return None, None

        if mode == 'train':
            assert(self._phase == 'train_val' or self._phase == 'train_noval')
            if deterministic:
                idx = self._train_idx[0:num_samples]
            else:
                idx = np.random.choice(self._train_idx, size=num_samples, replace=False)
            images, labels, IDs = self._get_train_samples_with_sem_labels(idx)
            if as_list:
                if return_IDs:
                    return images, labels, IDs
                else:
                    return images, labels
            else:
                if return_IDs:
                    return np.asarray(images), np.asarray(labels), np.asarray(IDs)
                else:
                    return np.asarray(images), np.asarray(labels)

        elif mode == 'val':
            assert(self._phase == 'train_val' or self._phase == 'val')
            if deterministic:
                idx = self._val_idx[0:num_samples]
            else:
                idx = np.random.choice(self._val_idx, size=num_samples, replace=False)
            images, labels, IDs = self._get_val_samples_with_sem_labels(idx)
            if as_list:
                if return_IDs:
                    return images, labels, IDs
                else:
                    return images, labels
            else:
                if return_IDs:
                    return np.asarray(images), np.asarray(labels), np.asarray(IDs)
                else:
                    return np.asarray(images), np.asarray(labels)

        elif mode == 'val_with_preds':
            assert(self._phase == 'train_val' or self._phase == 'val')
            if deterministic:
                idx = self._val_idx[0:num_samples]
            else:
                idx = np.random.choice(self._val_idx, size=num_samples, replace=False)
            images, labels, pred_labels, IDs = self._get_val_samples_with_sem_labels_and_preds(idx)
            if as_list:
                if return_IDs:
                    return images, labels, pred_labels, IDs
                else:
                    return images, labels, pred_labels
            else:
                if return_IDs:
                    return np.asarray(images), np.asarray(labels), np.asarray(pred_labels), np.asarray(IDs)
                else:
                    return np.asarray(images), np.asarray(labels), np.asarray(pred_labels)

        elif mode == 'test':
            if deterministic:
                idx = self._test_idx[0:num_samples]
            else:
                idx = np.random.choice(self._test_idx, size=num_samples, replace=False)
            images, IDs = [], []
            for l in idx:
                image = self._images_test[l] if self.options['in_memory'] else self._load_sample(
                    self._images_test_path[l])
                images.append(image)
                IDs.append(self._test_IDs[l])
            if as_list:
                if return_IDs:
                    return images, IDs
                else:
                    return images
            else:
                if return_IDs:
                    return np.asarray(images), np.asarray(IDs)
                else:
                    return np.asarray(images)

        elif mode == 'test_with_preds':
            if deterministic:
                idx = self._test_idx[0:num_samples]
            else:
                idx = np.random.choice(self._test_idx, size=num_samples, replace=False)
            images, pred_labels, IDs = self._get_test_samples_with_pred_sem_labels(idx)
            if as_list:
                if return_IDs:
                    return images, pred_labels, IDs
                else:
                    return images, pred_labels
            else:
                if return_IDs:
                    return np.asarray(images), np.asarray(pred_labels), np.asarray(IDs)
                else:
                    return np.asarray(images), np.asarray(pred_labels)

        else:
            return None, None


    ###
    ### Instance masks
    ###

    def _get_train_samples_with_inst_masks(self, idx):
        """Get training images with associated bboxes and instance masks
        Args:
            idx: List of sample indices to return
        Returns:
            images: List of RGB images in format list([H, W, 3])
            masks: List of list of instance masks in format list(list([num_instances, H, W, 1]))
            bboxes: List of list of instance bboxes in format list(list([num_instances, (y1, x1, y2, x2)]))
        """
        images, masks, bboxes, IDs = [], [], [], []
        for l in idx:
            if self.options['in_memory']:
                if self.options['compute_bboxes']:
                    _image, _masks, _bboxes = self._images_train[l], self._inst_masks_train[l], self._inst_bboxes_train[l]
                else:
                    _image, _masks = self._images_train[l], self._inst_masks_train[l]
            else:
                _image = self._load_sample(self._images_train_path[l])
                masks_folder = os.path.dirname(self._images_train_path[l]).replace('images', 'masks')
                _masks = load_inst_masks(masks_folder)  # np array [#, H, W]
                if self.options['compute_bboxes']:
                    _bboxes = extract_bboxes(_masks)
            images.append(_image)
            masks.append(_masks)
            if self.options['compute_bboxes']:
                bboxes.append(_bboxes)
            IDs.append(self._train_IDs[l])
        return images, masks, bboxes, IDs

    def _get_val_samples_with_inst_masks(self, idx):
        """Get validation images with associated bboxes and instance masks
        Args:
            idx: List of sample indices to return
        Returns:
            images: List of RGB images in format list([H, W, 3])
            masks: List of list of instance masks in format list(list([num_instances, H, W, 1]))
            bboxes: List of list of instance bboxes in format list(list([num_instances, (y1, x1, y2, x2)]))
        """
        images, masks, bboxes, IDs = [], [], [], []
        for l in idx:
            if self.options['in_memory']:
                if self.options['compute_bboxes']:
                    _image, _masks, _bboxes = self._images_val[l], self._inst_masks_val[l], self._inst_bboxes_val[l]
                else:
                    _image, _masks = self._images_val[l], self._inst_masks_val[l]
            else:
                _image = self._load_sample(self._images_val_path[l], never_aug=True)
                masks_folder = os.path.dirname(self._images_val_path[l]).replace('images', 'masks')
                _masks = load_inst_masks(masks_folder)  # np array [#, H, W]
                if self.options['compute_bboxes']:
                    _bboxes = extract_bboxes(_masks)
            images.append(_image)
            masks.append(_masks)
            if self.options['compute_bboxes']:
                bboxes.append(_bboxes)
            IDs.append(self._val_IDs[l])
        return images, masks, bboxes, IDs

    def _get_val_samples_with_inst_masks_and_preds(self, idx):
        """Get validation images with associated bboxes, instance masks, and predictions
        Args:
            idx: List of sample indices to return
        Returns:
            images: List of RGB images in format list([H, W, 3])
            masks: List of list of instance masks in format list(list([num_instances, H, W, 1]))
            bboxes: List of list of instance bboxes in format list(list([num_instances, (y1, x1, y2, x2)]))
            pred_masks: List of list of predicted instance masks in format list(list([num_instances, H, W, 1]))
            pred_bboxes: List of list of predicted instance bboxes in format list(list([num_instances, (y1, x1, y2, x2)]))
        """
        images, masks, bboxes, IDs = self._get_val_samples_with_inst_masks(idx)
        pred_masks, pred_bboxes = [], []
        for l in idx:
            pred_masks_folder = os.path.dirname(self._images_val_path[l]).replace('images', 'pred_inst_masks')
            if os.path.exists(pred_masks_folder):
                _masks = load_inst_masks(pred_masks_folder)  # np array [#, H, W]
                pred_masks.append(_masks)
                if self.options['compute_bboxes']:
                    _bboxes = extract_bboxes(_masks)
                    pred_bboxes.append(_bboxes)
        return images, masks, bboxes, pred_masks, pred_bboxes, IDs

    def _get_test_samples_with_pred_inst_masks(self, idx):
        """Get test images with predicted bboxes and instance masks
        Args:
            idx: List of sample indices to return
        Returns:
            images: List of RGB images in format list([H, W, 3])
            masks: List of list of instance masks in format list(list([num_instances, H, W, 1]))
            bboxes: List of list of instance bboxes in format list(list([num_instances, (y1, x1, y2, x2)]))
        """
        images, pred_masks, pred_bboxes, IDs = [], [], [], []
        for l in idx:
            image = self._images_test[l] if self.options['in_memory'] else self._load_sample(self._images_test_path[l], never_aug=True)
            images.append(image)
            pred_masks_folder = os.path.dirname(self._images_test_path[l]).replace('images', 'pred_inst_masks')
            if os.path.exists(pred_masks_folder):
                _masks = load_inst_masks(pred_masks_folder)  # np array [#, H, W]
                if self.options['compute_bboxes']:
                    _bboxes = extract_bboxes(_masks)
                pred_masks.append(_masks)
                if self.options['compute_bboxes']:
                    pred_bboxes.append(_bboxes)
            IDs.append(self._test_IDs[l])
        return images, pred_masks, pred_bboxes, IDs

    def get_rand_samples_with_inst_masks(self, num_samples, mode='val', as_list=True, return_IDs=False, deterministic=False):
        """Get a few (or all) random (or ordered) samples from the dataset.
        Used for debugging purposes (testing how the model is improving over time, for instance).
        If sampling from the training/validation set, there are instance masks/bboxes; otherwise, there aren't.
        Note that this doesn't return a valid np array if the images don't have the same size.
        Args:
            num_samples: Number of samples to return
            mode: Possible options: 'train', 'val', 'val_with_preds', 'test', 'test_with_preds'
            as_list: Return as list or np array?
            return_IDs: If True, also return ID of the sample
            deterministic: If True, return samples in order
        Returns in training and validation mode:
            images: Batch of RGB images in format [num_samples, H, W, 3] or list([H, W, 3])
            labels: Batch of instance bboxes in format [num_samples, num_instances, (y1, x1, y2, x2)] or list([num_instances, (y1, x1, y2, x2)])
            masks: Batch of instance masks in format [num_samples, num_instances, H, W, 1] or list [num_instances, H, W, 1]
        Returns in testing:
            images: Batch of RGB images in format [num_samples, H, W, 3]
            output_files: List of output file names that match the input file names
        """
        if num_samples == 0:
            return None, None

        if mode == 'train':
            assert(self._phase == 'train_val' or self._phase == 'train_noval')
            if deterministic:
                idx = self._train_idx[0:num_samples]
            else:
                idx = np.random.choice(self._train_idx, size=num_samples, replace=False)
            images, masks, bboxes, IDs = self._get_train_samples_with_inst_masks(idx)
            if as_list:
                if return_IDs:
                    return images, masks, bboxes, IDs
                else:
                    return images, masks, bboxes
            else:
                if return_IDs:
                    return np.asarray(images), np.asarray(masks), np.asarray(bboxes), np.asarray(IDs)
                else:
                    return np.asarray(images), np.asarray(masks), np.asarray(bboxes)

        elif mode == 'val':
            assert(self._phase == 'train_val' or self._phase == 'val')
            if deterministic:
                idx = self._val_idx[0:num_samples]
            else:
                idx = np.random.choice(self._val_idx, size=num_samples, replace=False)
            images, masks, bboxes, IDs = self._get_val_samples_with_inst_masks(idx)
            if as_list:
                if return_IDs:
                    return images, masks, bboxes, IDs
                else:
                    return images, masks, bboxes
            else:
                if return_IDs:
                    return np.asarray(images), np.asarray(masks), np.asarray(bboxes), np.asarray(IDs)
                else:
                    return np.asarray(images), np.asarray(masks), np.asarray(bboxes)

        elif mode == 'val_with_preds':
            assert(self._phase == 'train_val' or self._phase == 'val')
            if deterministic:
                idx = self._val_idx[0:num_samples]
            else:
                idx = np.random.choice(self._val_idx, size=num_samples, replace=False)
            images, masks, bboxes, pred_masks, pred_bboxes, IDs = self._get_val_samples_with_inst_masks_and_preds(idx)
            if as_list:
                if return_IDs:
                    return images, masks, bboxes, pred_masks, pred_bboxes, IDs
                else:
                    return images, masks, bboxes, pred_masks, pred_bboxes
            else:
                if return_IDs:
                    return np.asarray(images), np.asarray(masks), np.asarray(bboxes), np.asarray(pred_masks), np.asarray(pred_bboxes), np.asarray(IDs)
                else:
                    return np.asarray(images), np.asarray(masks), np.asarray(bboxes), np.asarray(pred_masks), np.asarray(pred_bboxes)

        elif mode == 'test':
            if deterministic:
                idx = self._test_idx[0:num_samples]
            else:
                idx = np.random.choice(self._test_idx, size=num_samples, replace=False)
            images, IDs = [], []
            for l in idx:
                image = self._images_test[l] if self.options['in_memory'] else self._load_sample(self._images_test_path[l])
                images.append(image)
                IDs.append(self._test_IDs[l])
            if as_list:
                if return_IDs:
                    return images, IDs
                else:
                    return images
            else:
                if return_IDs:
                    return np.asarray(images), np.asarray(IDs)
                else:
                    return np.asarray(images)

        elif mode == 'test_with_preds':
            if deterministic:
                idx = self._test_idx[0:num_samples]
            else:
                idx = np.random.choice(self._test_idx, size=num_samples, replace=False)
            images, pred_masks, pred_bboxes, IDs = self._get_test_samples_with_pred_inst_masks(idx)
            if as_list:
                if return_IDs:
                    return images, pred_masks, pred_bboxes, IDs
                else:
                    return images, pred_masks, pred_bboxes
            else:
                if return_IDs:
                    return np.asarray(images), np.asarray(pred_masks), np.asarray(pred_bboxes), np.asarray(IDs)
                else:
                    return np.asarray(images), np.asarray(pred_masks), np.asarray(pred_bboxes)

        else:
            return None, None

    ###
    ### Debug utils
    ###
    def print_config(self):
        """Display configuration values."""
        print("\nDataset Configuration:")
        for k, v in self.options.items():
            print("  {:20} {}".format(k, v))
        print("  {:20} {}".format('phase', self._phase))
        if self._phase == 'train_noval':
            print("  {:20} {}".format('train size', self.train_size))
        elif self._phase == 'train' or self._phase == 'train_val':
            print("  {:20} {}".format('train size', self.train_size))
            print("  {:20} {}".format('val size', self.val_size))
        elif self._phase == 'val':
            print("  {:20} {}".format('val size', self.val_size))
        elif self._phase == 'test':
            print("  {:20} {}".format('test size', self.test_size))


    def _combine_image_with_label(self, img_path, label_path, color=(0, 255, 0)):
        """Overlay label on image."""
        image, label = imread(img_path), imread(label_path)
        # Some input images are 32bit. Just discard the last dimension in that case.
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:,:,:3]
        label = np.expand_dims(label, axis=-1)
        return draw_mask(image, label, color, alpha=0.2)

    ###
    ### TODO TFRecords helpers
    ### See:
    ### https://github.com/fperazzi/davis-2017/blob/master/python/lib/davis/dataset/base.py
    ### https://github.com/fperazzi/davis-2017/blob/master/python/lib/davis/dataset/loader.py
    ### https://github.com/kwotsin/create_tfrecords
    ### https://kwotsin.github.io/tech/2017/01/29/tfrecords.html
    ### http://yeephycho.github.io/2016/08/15/image-data-in-tensorflow/
    ### E:\repos\models-master\research\inception\inception\data\build_imagenet_data.py
    ### E:\repos\models-master\research\object_detection\dataset_tools\create_kitti_tf_record.py
    ###
    def _load_from_tfrecords(self):
        # TODO _load_from_tfrecords
        pass

    def _write_to_tfrecords(self):
        # TODO _write_to_tfrecords
        pass

# def test():

#     from visualize import archive_images_with_labels, archive_images

#     # Load dataset (using contours)
#     options = _DEFAULT_DSB18_OPTIONS
#     options['mode'] = 'instance_contours'
#     options['in_memory'] = False
#     options['data_aug'] = 'basic'
#     options['input_channels'] = 4
#     options['contour_thickness'] = 2
#     ds = DSB18Dataset(phase='train_val', options=options)

#     # Display dataset configuration
#     ds.print_config()
#     # assert (ds.train_size == 536)
#     # assert (ds.val_size == 134)

#     # Parameters
#     num_samples = 4
#     save_folder1 = "c:\\temp\\visualizations1"
#     save_folder2 = "c:\\temp\\visualizations2"
#     save_folder3 = "c:\\temp\\visualizations3"
#     save_folder4 = "c:\\temp\\visualizations4"
#     save_folder5 = "c:\\temp\\visualizations5"
#     save_folder6 = "c:\\temp\\visualizations6"

#     # Inspect original dataset (with semantic labels)
#     images, labels = ds.get_rand_samples_with_sem_labels(num_samples, 'train')
#     assert (type(images) is list and type(labels) is list)
#     assert (len(images) == num_samples and len(labels) == num_samples)
#     for image, label in zip(images, labels):
#         print("Image [{}], Label [{}]".format(image.shape, label.shape))
#         assert (len(image.shape) == 3 and image.shape[2] == options['input_channels'])
#         assert (len(label.shape) == 3 and label.shape[2] == 1)
#         assert (image.shape[0] == label.shape[0] and image.shape[1] == label.shape[1])
#     archive_images_with_labels(images, labels, save_folder1)
#     labels = [np.squeeze(label) for label in labels]
#     archive_images(labels, save_folder2)

#     images, labels = ds.get_rand_samples_with_sem_labels(num_samples, 'val')
#     assert (type(images) is list and type(labels) is list)
#     assert (len(images) == num_samples and len(labels) == num_samples)
#     for image, label in zip(images, labels):
#         print("Image [{}], Label [{}]".format(image.shape, label.shape))
#         assert (len(image.shape) == 3 and image.shape[2] == options['input_channels'])
#         assert (len(label.shape) == 3 and label.shape[2] == 1)
#         assert (image.shape[0] == label.shape[0] and image.shape[1] == label.shape[1])
#     archive_images_with_labels(images, labels, save_folder3)
#     labels = [np.squeeze(label) for label in labels]
#     archive_images(labels, save_folder4)

#     images, labels, pred_labels = ds.get_rand_samples_with_sem_labels(num_samples, 'val_with_preds')
#     assert (type(images) is list and type(labels) is list and type(pred_labels) is list)
#     assert (len(images) == num_samples and len(labels) == num_samples)
#     assert (len(pred_labels) == num_samples or len(pred_labels) == 0)
#     if len(pred_labels) == 0:
#         for image, label in zip(images, labels):
#             print("Image [{}], Label [{}], No Pred Label".format(image.shape, label.shape))
#             assert (len(image.shape) == 3 and image.shape[2] == options['input_channels'])
#             assert (len(label.shape) == 3 and label.shape[2] == 1)
#             assert (image.shape[0] == label.shape[0] and image.shape[1] == label.shape[1])
#     else:
#         for image, label, pred_label in zip(images, labels, pred_labels):
#             print("Image [{}], Label [{}], Pred Label [{}]".format(image.shape, label.shape, pred_label.shape))
#             assert (len(image.shape) == 3 and image.shape[2] == options['input_channels'])
#             assert (len(label.shape) == 3 and label.shape[2] == 1)
#             assert (len(pred_label.shape) == 3 and pred_label.shape[2] == 1)
#             assert (image.shape[0] == label.shape[0] and image.shape[1] == label.shape[1])
#             assert (image.shape[0] == pred_label.shape[0] and image.shape[1] == pred_label.shape[1])
#     archive_images_with_labels(images, labels, save_folder5)
#     labels = [np.squeeze(label) for label in labels]
#     archive_images(labels, save_folder6)

#     # Load dataset (using contours)
#     options = _DEFAULT_DSB18_OPTIONS
#     options['mode'] = 'instance_contours'
#     options['in_memory'] = False
#     ds_inst_contours = DSB18Dataset(phase='test', options=options)

#     # Display dataset configuration
#     ds_inst_contours.print_config()
#     assert (ds_inst_contours.test_size == 65)

#     # Inspect original dataset (with semantic labels)
#     images = ds_inst_contours.get_rand_samples_with_sem_labels(num_samples, 'test')
#     assert (type(images) and len(images) == num_samples)
#     for image in images:
#         print("Image [{}]".format(image.shape))
#         assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))

#     images, pred_labels = ds_inst_contours.get_rand_samples_with_sem_labels(num_samples, 'test_with_preds')
#     assert (type(images) is list and type(pred_labels) is list)
#     assert (len(images) == num_samples and (len(pred_labels) == num_samples or len(pred_labels) == 0))
#     if len(pred_labels) == 0:
#         for image in images:
#             print("Image [{}], No Pred Label".format(image.shape))
#             assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#     else:
#         for image, pred_label in zip(images, pred_labels):
#             print("Image [{}], Pred Label [{}]".format(image.shape, pred_label.shape))
#             assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#             assert (len(pred_label.shape) == 3 and pred_label.shape[2] == 1)
#             assert (image.shape[0] == pred_label.shape[0] and image.shape[1] == pred_label.shape[1])

#     # Load dataset (using semantic labels)
#     options = _DEFAULT_DSB18_OPTIONS
#     options['mode'] = 'semantic_labels'
#     options['in_memory'] = False
#     ds_sem_labels = DSB18Dataset(phase='train_val', options=options)

#     # Display dataset configuration
#     ds_sem_labels.print_config()
#     assert (ds_sem_labels.train_size == 536)
#     assert (ds_sem_labels.val_size == 134)

#     # Inspect original dataset (with semantic labels)
#     num_samples = 4
#     images, labels = ds_sem_labels.get_rand_samples_with_sem_labels(num_samples, 'train')
#     assert (type(images) is list and type(labels) is list)
#     assert (len(images) == num_samples and len(labels) == num_samples)
#     for image, label in zip(images, labels):
#         print("Image [{}], Label [{}]".format(image.shape, label.shape))
#         assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#         assert (len(label.shape) == 3 and label.shape[2] == 1)
#         assert (image.shape[0] == label.shape[0] and image.shape[1] == label.shape[1])

#     images, labels = ds_sem_labels.get_rand_samples_with_sem_labels(num_samples, 'val')
#     assert (type(images) is list and type(labels) is list)
#     assert (len(images) == num_samples and len(labels) == num_samples)
#     for image, label in zip(images, labels):
#         print("Image [{}], Label [{}]".format(image.shape, label.shape))
#         assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#         assert (len(label.shape) == 3 and label.shape[2] == 1)
#         assert (image.shape[0] == label.shape[0] and image.shape[1] == label.shape[1])

#     images, labels, pred_labels = ds_sem_labels.get_rand_samples_with_sem_labels(num_samples, 'val_with_preds')
#     assert (type(images) is list and type(labels) is list and type(pred_labels) is list)
#     assert (len(images) == num_samples and len(labels) == num_samples)
#     assert (len(pred_labels) == num_samples or len(pred_labels) == 0)
#     if len(pred_labels) == 0:
#         for image, label in zip(images, labels):
#             print("Image [{}], Label [{}], No Pred Label".format(image.shape, label.shape))
#             assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#             assert (len(label.shape) == 3 and label.shape[2] == 1)
#             assert (image.shape[0] == label.shape[0] and image.shape[1] == label.shape[1])
#     else:
#         for image, label, pred_label in zip(images, labels, pred_labels):
#             print("Image [{}], Label [{}], Pred Label [{}]".format(image.shape, label.shape, pred_label.shape))
#             assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#             assert (len(label.shape) == 3 and label.shape[2] == 1)
#             assert (len(pred_label.shape) == 3 and pred_label.shape[2] == 1)
#             assert (image.shape[0] == label.shape[0] and image.shape[1] == label.shape[1])
#             assert (image.shape[0] == pred_label.shape[0] and image.shape[1] == pred_label.shape[1])

#     # Load dataset (using semantic labels)
#     options = _DEFAULT_DSB18_OPTIONS
#     options['mode'] = 'semantic_labels'
#     options['in_memory'] = False
#     ds_sem_labels = DSB18Dataset(phase='test', options=options)

#     # Display dataset configuration
#     ds_sem_labels.print_config()
#     assert (ds_sem_labels.test_size == 65)

#     # Inspect original dataset (with semantic labels)
#     images = ds_sem_labels.get_rand_samples_with_sem_labels(num_samples, 'test')
#     assert (type(images) and len(images) == num_samples)
#     for image in images:
#         print("Image [{}]".format(image.shape))
#         assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))

#     images, pred_labels = ds_sem_labels.get_rand_samples_with_sem_labels(num_samples, 'test_with_preds')
#     assert (type(images) is list and type(pred_labels) is list)
#     assert (len(images) == num_samples and (len(pred_labels) == num_samples or len(pred_labels) == 0))
#     if len(pred_labels) == 0:
#         for image in images:
#             print("Image [{}], No Pred Label".format(image.shape))
#             assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#     else:
#         for image, pred_label in zip(images, pred_labels):
#             print("Image [{}], Pred Label [{}]".format(image.shape, pred_label.shape))
#             assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#             assert (len(pred_label.shape) == 3 and pred_label.shape[2] == 1)
#             assert (image.shape[0] == pred_label.shape[0] and image.shape[1] == pred_label.shape[1])

#     # Load dataset (using semantic labels)
#     options = _DEFAULT_DSB18_OPTIONS
#     options['mode'] = 'semantic_labels'
#     options['in_memory'] = True
#     ds_sem_labels = DSB18Dataset(phase='train_val', options=options)

#     # Display dataset configuration
#     ds_sem_labels.print_config()
#     assert (ds_sem_labels.train_size == 536)
#     assert (ds_sem_labels.val_size == 134)

#     # Inspect original dataset (with semantic labels)
#     num_samples = 4
#     images, labels = ds_sem_labels.get_rand_samples_with_sem_labels(num_samples, 'train')
#     assert (type(images) is list and type(labels) is list)
#     assert (len(images) == num_samples and len(labels) == num_samples)
#     for image, label in zip(images, labels):
#         print("Image [{}], Label [{}]".format(image.shape, label.shape))
#         assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#         assert (len(label.shape) == 3 and label.shape[2] == 1)
#         assert (image.shape[0] == label.shape[0] and image.shape[1] == label.shape[1])

#     images, labels = ds_sem_labels.get_rand_samples_with_sem_labels(num_samples, 'val')
#     assert (type(images) is list and type(labels) is list)
#     assert (len(images) == num_samples and len(labels) == num_samples)
#     for image, label in zip(images, labels):
#         print("Image [{}], Label [{}]".format(image.shape, label.shape))
#         assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#         assert (len(label.shape) == 3 and label.shape[2] == 1)
#         assert (image.shape[0] == label.shape[0] and image.shape[1] == label.shape[1])

#     images, labels, pred_labels = ds_sem_labels.get_rand_samples_with_sem_labels(num_samples, 'val_with_preds')
#     assert (type(images) is list and type(labels) is list and type(pred_labels) is list)
#     assert (len(images) == num_samples and len(labels) == num_samples)
#     assert (len(pred_labels) == num_samples or len(pred_labels) == 0)
#     if len(pred_labels) == 0:
#         for image, label in zip(images, labels):
#             print("Image [{}], Label [{}]".format(image.shape, label.shape))
#             assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#             assert (len(label.shape) == 3 and label.shape[2] == 1)
#             assert (image.shape[0] == label.shape[0] and image.shape[1] == label.shape[1])
#     else:
#         for image, label, pred_label in zip(images, labels, pred_labels):
#             print("Image [{}], Label [{}], Pred Label [{}]".format(image.shape, label.shape, pred_label.shape))
#             assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#             assert (len(label.shape) == 3 and label.shape[2] == 1)
#             assert (len(pred_label.shape) == 3 and pred_label.shape[2] == 1)
#             assert (image.shape[0] == label.shape[0] and image.shape[1] == label.shape[1])
#             assert (image.shape[0] == pred_label.shape[0] and image.shape[1] == pred_label.shape[1])

#     # Load dataset (using semantic labels)
#     options = _DEFAULT_DSB18_OPTIONS
#     options['mode'] = 'semantic_labels'
#     options['in_memory'] = True
#     ds_sem_labels = DSB18Dataset(phase='test', options=options)

#     # Display dataset configuration
#     ds_sem_labels.print_config()
#     assert (ds_sem_labels.test_size == 65)

#     # Inspect original dataset (with semantic labels)
#     images = ds_sem_labels.get_rand_samples_with_sem_labels(num_samples, 'test')
#     assert (type(images) and len(images) == num_samples)
#     for image in images:
#         print("Image [{}]".format(image.shape))
#         assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))

#     images, pred_labels = ds_sem_labels.get_rand_samples_with_sem_labels(num_samples, 'test_with_preds')
#     assert (type(images) is list and type(pred_labels) is list)
#     assert (len(images) == num_samples and (len(pred_labels) == num_samples or len(pred_labels) == 0))
#     if len(pred_labels) == 0:
#         for image in images:
#             print("Image [{}]".format(image.shape))
#             assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#     else:
#         for image, pred_label in zip(images, pred_labels):
#             print("Image [{}], Pred Label [{}]".format(image.shape, pred_label.shape))
#             assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#             assert (len(pred_label.shape) == 3 and pred_label.shape[2] == 1)
#             assert (image.shape[0] == pred_label.shape[0] and image.shape[1] == pred_label.shape[1])

#     # Load dataset (using instance masks)
#     options = _DEFAULT_DSB18_OPTIONS
#     options['mode'] = 'instance_masks'
#     options['in_memory'] = False
#     ds_inst_masks = DSB18Dataset(phase='train_val', options=options)

#     # Display dataset configuration
#     ds_inst_masks.print_config()
#     assert (ds_inst_masks.train_size == 536)
#     assert (ds_inst_masks.val_size == 134)

#     # Inspect original dataset (with instance masks)
#     num_samples = 4
#     images, inst_masks, inst_bboxes = ds_inst_masks.get_rand_samples_with_inst_masks(num_samples, 'train')
#     assert (type(images) is list and type(inst_masks) is list and type(inst_bboxes) is list)
#     assert (len(images) == num_samples and len(inst_masks) == num_samples and len(inst_bboxes) == num_samples)
#     for image, masks, bboxes in zip(images, inst_masks, inst_bboxes):
#         print("Image [{}]".format(image.shape))
#         assert (type(masks) is np.ndarray and type(bboxes) is np.ndarray)
#         assert (masks.shape[2] == bboxes.shape[0] and bboxes.shape[1] == 4)
#         assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#         for idx in range(masks.shape[-1]):
#             mask = masks[:, :, idx]
#             bbox = bboxes[idx]
#             print("  Mask [{}], Bbox [{}]".format(mask.shape, bbox.shape))
#             assert (len(mask.shape) == 2 and len(bbox.shape) == 1)
#             assert (image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1])

#     images, inst_masks, inst_bboxes = ds_inst_masks.get_rand_samples_with_inst_masks(num_samples, 'val')
#     assert (type(images) is list and type(inst_masks) is list and type(inst_bboxes) is list)
#     assert (len(images) == num_samples and len(inst_masks) == num_samples and len(inst_bboxes) == num_samples)
#     for image, masks, bboxes in zip(images, inst_masks, inst_bboxes):
#         print("Image [{}]".format(image.shape))
#         assert (type(masks) is np.ndarray and type(bboxes) is np.ndarray)
#         assert (masks.shape[2] == bboxes.shape[0] and bboxes.shape[1] == 4)
#         assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#         for idx in range(masks.shape[-1]):
#             mask = masks[:, :, idx]
#             bbox = bboxes[idx]
#             print("  Mask [{}], Bbox [{}]".format(mask.shape, bbox.shape))
#             assert (len(mask.shape) == 2 and len(bbox.shape) == 1)
#             assert (image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1])

#     images, inst_masks, inst_bboxes, pred_inst_masks, pred_inst_bboxes = ds_inst_masks.get_rand_samples_with_inst_masks(
#         num_samples, 'val_with_preds')
#     assert (type(images) is list and type(inst_masks) is list and type(inst_bboxes) is list)
#     assert (type(pred_inst_masks) is list and type(pred_inst_bboxes) is list)
#     assert (len(images) == num_samples and len(inst_masks) == num_samples and len(inst_bboxes) == num_samples)
#     assert (len(pred_inst_masks) == len(pred_inst_bboxes))
#     if len(pred_inst_masks) == 0:
#         for image, masks, bboxes in zip(images, inst_masks, inst_bboxes):
#             print("Image [{}]".format(image.shape))
#             assert (type(masks) is np.ndarray and type(bboxes) is np.ndarray)
#             assert (masks.shape[2] == bboxes.shape[0] and bboxes.shape[1] == 4)
#             assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#             for idx in range(masks.shape[-1]):
#                 mask = masks[:, :, idx]
#                 bbox = bboxes[idx]
#                 print("  Mask [{}], Bbox [{}]".format(mask.shape, bbox.shape))
#                 assert (len(mask.shape) == 2 and len(bbox.shape) == 1)
#                 assert (image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1])
#     else:
#         for image, masks, bboxes, pred_masks, pred_bboxes in zip(images, inst_masks, inst_bboxes, pred_inst_masks,
#                                                                  pred_inst_bboxes):
#             print("Image [{}]".format(image.shape))
#             assert (type(masks) is np.ndarray and type(bboxes) is np.ndarray)
#             assert (type(pred_masks) is np.ndarray and type(pred_bboxes) is np.ndarray)
#             assert (masks.shape[2] == bboxes.shape[0] and bboxes.shape[1] == 4)
#             assert (pred_masks.shape[2] == pred_bboxes.shape[0] and pred_bboxes.shape[1] == 4)
#             assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#             for idx in range(masks.shape[-1]):
#                 mask = masks[:, :, idx]
#                 bbox = bboxes[idx]
#                 print("  Mask [{}], Bbox [{}]".format(mask.shape, bbox.shape))
#                 assert (len(mask.shape) == 2 and len(bbox.shape) == 1)
#                 assert (image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1])
#             for idx in range(pred_masks.shape[-1]):
#                 mask = pred_masks[:, :, idx]
#                 bbox = pred_bboxes[idx]
#                 print("  Mask [{}], Bbox [{}]".format(mask.shape, bbox.shape))
#                 assert (len(mask.shape) == 2 and len(bbox.shape) == 1)
#                 assert (image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1])

#     # Load dataset (using instance masks)
#     options = _DEFAULT_DSB18_OPTIONS
#     options['mode'] = 'instance_masks'
#     options['in_memory'] = False
#     ds_inst_masks = DSB18Dataset(phase='test', options=options)

#     # Display dataset configuration
#     ds_inst_masks.print_config()
#     assert (ds_inst_masks.test_size == 65)

#     # Inspect original dataset (with instance masks)
#     num_samples = 4
#     images = ds_inst_masks.get_rand_samples_with_inst_masks(num_samples, 'test')
#     assert (type(images) is list)
#     assert (len(images) == num_samples)
#     for image in images:
#         print("Image [{}]".format(image.shape))
#         assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))

#     images, pred_inst_masks, pred_inst_bboxes = ds_inst_masks.get_rand_samples_with_inst_masks(num_samples,
#                                                                                                'test_with_preds')
#     assert (type(images) is list and type(pred_inst_masks) is list and type(pred_inst_bboxes) is list)
#     assert (len(images) == num_samples and len(pred_inst_masks) == num_samples and len(pred_inst_bboxes) == num_samples)
#     for image, masks, bboxes in zip(images, pred_inst_masks, pred_inst_bboxes):
#         print("Image [{}]".format(image.shape))
#         assert (type(masks) is np.ndarray and type(bboxes) is np.ndarray)
#         assert (masks.shape[2] == bboxes.shape[0] and bboxes.shape[1] == 4)
#         assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#         for idx in range(masks.shape[-1]):
#             mask = masks[:, :, idx]
#             bbox = bboxes[idx]
#             print("  Mask [{}], Bbox [{}]".format(mask.shape, bbox.shape))
#             assert (len(mask.shape) == 2 and len(bbox.shape) == 1)
#             assert (image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1])

#     # Load dataset (using instance masks)
#     options = _DEFAULT_DSB18_OPTIONS
#     options['mode'] = 'instance_masks'
#     options['in_memory'] = True
#     ds_inst_masks = DSB18Dataset(phase='train_val', options=options)

#     # Display dataset configuration
#     ds_inst_masks.print_config()
#     assert (ds_inst_masks.train_size == 536)
#     assert (ds_inst_masks.val_size == 134)

#     # Inspect original dataset (with instance masks)
#     num_samples = 4
#     images, inst_masks, inst_bboxes = ds_inst_masks.get_rand_samples_with_inst_masks(num_samples, 'train')
#     assert (type(images) is list and type(inst_masks) is list and type(inst_bboxes) is list)
#     assert (len(images) == num_samples and len(inst_masks) == num_samples and len(inst_bboxes) == num_samples)
#     for image, masks, bboxes in zip(images, inst_masks, inst_bboxes):
#         print("Image [{}]".format(image.shape))
#         assert (type(masks) is np.ndarray and type(bboxes) is np.ndarray)
#         assert (masks.shape[2] == bboxes.shape[0] and bboxes.shape[1] == 4)
#         assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#         for idx in range(masks.shape[-1]):
#             mask = masks[:, :, idx]
#             bbox = bboxes[idx]
#             print("  Mask [{}], Bbox [{}]".format(mask.shape, bbox.shape))
#             assert (len(mask.shape) == 2 and len(bbox.shape) == 1)
#             assert (image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1])

#     images, inst_masks, inst_bboxes = ds_inst_masks.get_rand_samples_with_inst_masks(num_samples, 'val')
#     assert (type(images) is list and type(inst_masks) is list and type(inst_bboxes) is list)
#     assert (len(images) == num_samples and len(inst_masks) == num_samples and len(inst_bboxes) == num_samples)
#     for image, masks, bboxes in zip(images, inst_masks, inst_bboxes):
#         print("Image [{}]".format(image.shape))
#         assert (type(masks) is np.ndarray and type(bboxes) is np.ndarray)
#         assert (masks.shape[2] == bboxes.shape[0] and bboxes.shape[1] == 4)
#         assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#         for idx in range(masks.shape[-1]):
#             mask = masks[:, :, idx]
#             bbox = bboxes[idx]
#             print("  Mask [{}], Bbox [{}]".format(mask.shape, bbox.shape))
#             assert (len(mask.shape) == 2 and len(bbox.shape) == 1)
#             assert (image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1])

#     images, inst_masks, inst_bboxes, pred_inst_masks, pred_inst_bboxes = ds_inst_masks.get_rand_samples_with_inst_masks(
#         num_samples, 'val_with_preds')
#     assert (type(images) is list and type(inst_masks) is list and type(inst_bboxes) is list)
#     assert (type(pred_inst_masks) is list and type(pred_inst_bboxes) is list)
#     assert (len(images) == num_samples and len(inst_masks) == num_samples and len(inst_bboxes) == num_samples)
#     assert (len(pred_inst_masks) == len(pred_inst_bboxes))
#     if len(pred_inst_masks) == 0:
#         for image, masks, bboxes in zip(images, inst_masks, inst_bboxes):
#             print("Image [{}]".format(image.shape))
#             assert (type(masks) is np.ndarray and type(bboxes) is np.ndarray)
#             assert (masks.shape[2] == bboxes.shape[0] and bboxes.shape[1] == 4)
#             assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#             for idx in range(masks.shape[-1]):
#                 mask = masks[:, :, idx]
#                 bbox = bboxes[idx]
#                 print("  Mask [{}], Bbox [{}]".format(mask.shape, bbox.shape))
#                 assert (len(mask.shape) == 2 and len(bbox.shape) == 1)
#                 assert (image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1])
#     else:
#         for image, masks, bboxes, pred_masks, pred_bboxes in zip(images, inst_masks, inst_bboxes, pred_inst_masks,
#                                                                  pred_inst_bboxes):
#             print("Image [{}]".format(image.shape))
#             assert (type(masks) is np.ndarray and type(bboxes) is np.ndarray)
#             assert (type(pred_masks) is np.ndarray and type(pred_bboxes) is np.ndarray)
#             assert (masks.shape[2] == bboxes.shape[0] and bboxes.shape[1] == 4)
#             assert (pred_masks.shape[2] == pred_bboxes.shape[0] and pred_bboxes.shape[1] == 4)
#             assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#             for idx in range(masks.shape[-1]):
#                 mask = masks[:, :, idx]
#                 bbox = bboxes[idx]
#                 print("  Mask [{}], Bbox [{}]".format(mask.shape, bbox.shape))
#                 assert (len(mask.shape) == 2 and len(bbox.shape) == 1)
#                 assert (image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1])
#             for idx in range(pred_masks.shape[-1]):
#                 mask = pred_masks[:, :, idx]
#                 bbox = pred_bboxes[idx]
#                 print("  Mask [{}], Bbox [{}]".format(mask.shape, bbox.shape))
#                 assert (len(mask.shape) == 2 and len(bbox.shape) == 1)
#                 assert (image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1])

#     # Load dataset (using instance masks)
#     options = _DEFAULT_DSB18_OPTIONS
#     options['mode'] = 'instance_masks'
#     options['in_memory'] = True
#     ds_inst_masks = DSB18Dataset(phase='test', options=options)

#     # Display dataset configuration
#     ds_inst_masks.print_config()
#     assert (ds_inst_masks.test_size == 65)

#     # Inspect original dataset (with instance masks)
#     num_samples = 4
#     images = ds_inst_masks.get_rand_samples_with_inst_masks(num_samples, 'test')
#     assert (type(images) is list)
#     assert (len(images) == num_samples)
#     for image in images:
#         print("Image [{}]".format(image.shape))
#         assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))

#     images, pred_inst_masks, pred_inst_bboxes = ds_inst_masks.get_rand_samples_with_inst_masks(num_samples,
#                                                                                                'test_with_preds')
#     assert (type(images) is list and type(pred_inst_masks) is list and type(pred_inst_bboxes) is list)
#     assert (len(images) == num_samples and len(pred_inst_masks) == num_samples and len(pred_inst_bboxes) == num_samples)
#     for image, masks, bboxes in zip(images, pred_inst_masks, pred_inst_bboxes):
#         print("Image [{}]".format(image.shape))
#         assert (type(masks) is np.ndarray and type(bboxes) is np.ndarray)
#         assert (masks.shape[2] == bboxes.shape[0] and bboxes.shape[1] == 4)
#         assert (len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4))
#         for idx in range(masks.shape[-1]):
#             mask = masks[:, :, idx]
#             bbox = bboxes[idx]
#             print("  Mask [{}], Bbox [{}]".format(mask.shape, bbox.shape))
#             assert (len(mask.shape) == 2 and len(bbox.shape) == 1)
#             assert (image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1])

# if __name__ == '__main__':
#     test()
