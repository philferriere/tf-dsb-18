"""
submission.py

Kaggle submission utility functions and classes.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, warnings
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread, imsave
from segment import sem_label_to_inst_masks, save_inst_masks, load_inst_masks

# from post import Post
from dataset import DSB18Dataset, _DSB18_DATASET

_DEFAULT_DSB18_SUBMISSION = {'key' : 'value'}

class DSB18Submission(object):
    """2018 Data Science Bowl submission object.
    """

    def __init__(self, instance_masks_csv=None, ds_root=_DSB18_DATASET, options=_DEFAULT_DSB18_SUBMISSION):
        """Initialize the Submission object
        Args:
            instance_masks_csv: Path to the instance masks CSV. If None, we make one up.
            # semantic_labels_csv: Path to the semantic labels CSV. If None, we make one up.
            ds_root: Path to the root of the dataset
            options: see below
        Options:
            data_aug: If TRUE, use predictions generated using the model trained with augmented data
            use_cache: If TRUE, use predictions post-processed with CRF
        """
        # Only options supported in this initial implementation
        assert (options == _DEFAULT_DSB18_SUBMISSION)
        self._options = options

        # Set paths and file names
        self._ds_root = ds_root
        self._test_folder = self._ds_root + '/stage1_test'
        self._submissions_folder = self._ds_root + '/submissions'
        self._test_IDs_file = self._ds_root + '/test.txt'
        if instance_masks_csv is None:
            self._instance_masks_csv = '{}/inst_masks_sub_{}.csv'.format(self._submissions_folder, time.strftime("%Y-%m-%d_%Hh-%Mm-%Ss"))
        else:
            self._instance_masks_csv = instance_masks_csv
        # if semantic_labels_csv is None:
        #     self.semantic_labels_csv = '{}/sem_labels_sub_{}.csv'.format(self._submissions_folder, time.strftime("%Y-%m-%d_%Hh-%Mm-%Ss"))
        # else:
        #     self.semantic_labels_csv = semantic_labels_csv

        # Load ID file
        self._load_ID_file()

        # Do all the post-processing needed before a submission can be made
        self._prepare()

    @property
    def instance_masks_csv(self):
        return self._instance_masks_csv

    ###
    ### Labels Prep
    ###
    def _prepare(self):
        """Do all the post-processing needed before a submission can be made.
        """
        if not os.path.exists(self._submissions_folder):
            os.makedirs(self._submissions_folder)

    def _load_ID_file(self):
        """Load the ID file and create the instance mask file paths
        Returns:
              True if ID file was loaded, False if ID files wasn't found
        """
        if not os.path.exists(self._test_IDs_file):
            return False

        # Read in the IDs
        with open(self._test_IDs_file, 'r') as f:
            self._test_IDs = f.readlines()
            self._test_IDs = [ID.rstrip() for ID in self._test_IDs]

        # build paths to predicted instance masks
        self._pred_inst_masks_folders = [self._test_folder + '/' + idx + '/pred_inst_masks' for idx in self._test_IDs]

        return True

    ###
    ### RLE Encoding
    ###

    def to_csv(self):
        """Create submission DataFrame and generate submission CSV
        """
        if os.path.exists(self._instance_masks_csv):
            os.remove(self._instance_masks_csv)

        # print(self.submission_test_ids)
        # print(self.rles)
        sub = pd.DataFrame()
        sub['ImageId'] = self.submission_test_ids
        sub['EncodedPixels'] = pd.Series(self.rles).apply(lambda x: ' '.join(str(y) for y in x))
        sub.to_csv(self._instance_masks_csv, index=False)

    def rle_encoding(self, x):
        """
        RLE encode a b
        """
        dots = np.where(x.T.flatten() == 1)[0]
        run_lengths = []
        prev = -2
        for b in dots:
            if (b > prev + 1): run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        return run_lengths

    def encode_predicted_instance_masks(self):
        """RLE encode predicted instance masks.
        Each predicted instance mask is stored in a `<inage_id>/pred_inst_masks/<mask id>.png` file
        """
        self.submission_test_ids = []
        self.rles = []

        with tqdm(total=len(self._test_IDs), desc="RLE encode predicted instance masks", ascii=True, ncols=100) as pbar:
            for n, id_ in enumerate(self._test_IDs):
                pbar.update(1)
                rle = []
                inst_masks = load_inst_masks(self._pred_inst_masks_folders[n])
                for inst_mask in inst_masks:
                    rle.append(self.rle_encoding(inst_mask == 255))
                self.rles.extend(rle)
                self.submission_test_ids.extend([id_] * len(rle))

    ###
    ### Debug utils
    ###
    def print_config(self):
        """Display configuration values."""
        print("\nConfiguration:")
        for k, v in self._options.items():
            print("  {:20} {}".format(k, v))
        print("  {:20} {}".format('samples', len(self._test_IDs)))
        print("  {:20} {}".format('output csv', self._instance_masks_csv))

    ###
    ### Functions DISABLED in this project
    ### They're the functions related to semantic labels instead of instance masks
    ###
    def DISABLED_semantic_label_to_instance_masks_rles(self, x, cutoff=128):
        """Convert a semantic label into individual instance masks and RLE encode them
        """
        lab_img = label(x > cutoff)
        for i in range(1, lab_img.max() + 1):
            yield self.rle_encoding(lab_img == i)

    def DISABLED_encode_predicted_semantic_labels(self):
        """RLE encode predicted semantic labels.
        Each predicted semantic label is stored in a `<inage_id>/pred_sem_label/<inage_id>.png` file
        """
        assert("this function shouldn't be called in this project!")
        assert("this code is untested!")
        self.submission_test_ids = []
        self.rles = []

        with tqdm(total=len(self._test_IDs), desc="RLE encode predicted semantic labels", ascii=True, ncols=100) as pbar:
            for n, id_ in enumerate(self._test_IDs):
                pbar.update(1)
                self.submission_test_ids.append(id_)
                self.rles.append(self.rle_encoding(imread(self._label_paths[n])))

    def DISABLED_encode_predicted_semantic_labels_as_instance_masks(self):
        """RLE encode predicted instance masks.
        Each predicted instance mask is stored in a `<inage_id>/pred_inst_masks/<mask id>.png` file
        """
        assert("this function shouldn't be called in this project!")
        self.submission_test_ids = []
        self.rles = []

        with tqdm(total=len(self._label_paths), desc="RLE encode predicted instance masks", ascii=True, ncols=100) as pbar:
            for n, id_ in enumerate(self._test_IDs):
                pbar.update(1)
                rle = list(self.semantic_label_to_instance_masks_rles(imread(self._label_paths[n])))
                self.rles.extend(rle)
                self.submission_test_ids.extend([id_] * len(rle))

    def DISABLED_semantic_labels_to_instance_masks(self):
        """Convert semantic labels to instance masks.
        Each generated instance mask is stored in a `<inage_id>/pred_inst_masks/<mask id>.png` file
        """
        assert("this function shouldn't be called in this project!")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with tqdm(total=len(self._test_IDs), desc="Split semantic labels into instance masks", ascii=True, ncols=100) as pbar:
                for n, idx in enumerate(self._test_IDs):
                    pbar.update(1)
                    # Split semantic label into instance masks using connected components
                    inst_masks = sem_label_to_inst_masks(imread(self._label_paths[n]))
                    # Create output folder
                    folder_path = "{}/{}/pred_inst_masks/".format(self._test_folder, idx)
                    # Save created instance masks to disk
                    save_inst_masks(inst_masks, folder_path)


if __name__ == '__main__':

    #
    # Generate a submission
    #
    # Create a Submission object
    submit = DSB18Submission()
    submit.print_config()

    # RLE encode predicted instance masks
    submit.encode_predicted_instance_masks()

    # Create submission DataFrame and generate submission CSV
    submit.to_csv()
    print(submit.instance_masks_csv)

