"""
post_stage2.py

Post-processing utility functions and classes.

Modifications by Phil Ferriere licensed under the MIT License (see LICENSE for details)

Based on:
  - https://github.com/neptune-ml/data-science-bowl-2018/blob/master/postprocessing.py
    Written by https://github.com/neptune-ml
    MIT code license

Comparison at various settings on gt data used as predictions (results computed over 670 samples):
    connected_components mAP: 0.7611403272506709
    watershed_centers_basic mAP: 0.9067145585678406
    watershed_contours_basic 4px-wide contours mAP: 0.8470611056580025
    watershed_contours_basic 3px-wide mAP: 0.8470611056580025
    watershed_contours_basic 2px-wide mAP: 0.912181779569604

Worst offenders at each setting:
    connected_components top 20 worst offenders:
    ID:e5aeb5b3577abbebe8982b5dd7d22c4257250ad3000661a42f38bf9248d291fd - AP:0.05
    ID:93cfd412c7de5210bbd262ec3a602cfea65072e9272e9fce9b5339a5b9436eb7 - AP:0.18928571428571428
    ID:e49fc2b4f1f39d481a6525225ab3f688be5c87f56884456ad54c953315efae83 - AP:0.20707719935129573
    ID:a31deaf0ac279d5f34fb2eca80cc2abce6ef30bd64e7aca40efe4b2ba8e9ad3d - AP:0.21408777491117864
    ID:55f98f43c152aa0dc8bea513f8ba558cc57494b81ae4ee816977816e79629c50 - AP:0.2160098973607038
    ID:9ebcfaf2322932d464f15b5662cae4d669b2d785b8299556d73fffcae8365d32 - AP:0.23260577224293896
    ID:fa73f24532b3667718ede7ac5c2e24ad7d3cae17b0a42ed17bbb81b15c28f4ae - AP:0.25
    ID:a0325cb7aa59e9c0a75e64ba26855d8032c46161aa4bca0c01bac5e4a836485e - AP:0.25160085058327425
    ID:88d5a03f8ecd459f076a06e0d5035149193bfdd727c30905de19054dcb9018ae - AP:0.2667811059019442
    ID:5b0bde771bc67c505d1b59405cbcad0a2766ec3ee4e35852e959552c1b454233 - AP:0.2857808857808858
    ID:a7f767ca9770b160f234780e172aeb35a50830ba10dc49c526f4712451abe1d2 - AP:0.2905677655677656
    ID:58c593bcb98386e7fd42a1d34e291db93477624b164e83ab2afa3caa90d1d921 - AP:0.29166666666666663
    ID:4d14a3629b6af6de86d850be236b833a7bfcbf6d8665fd73c6dc339e06c14607 - AP:0.3205734732348355
    ID:5c6eb9a47852754d4e45eceb9a696c64c7cfe304afc5ea491cdfef11d55c17f3 - AP:0.3224218164445098
    ID:c53326fe49fc26b7fe602b9d8c0c2da2cb157690b44c2b9351a93f8d9bd8043d - AP:0.3515979635559222
    ID:00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e - AP:0.3571732356628037
    ID:a7f6194ddbeaefb1da571226a97785d09ccafc5893ce3c77078d2040bccfcb77 - AP:0.35896399421657654
    ID:8a65e41c630d85c0004ce1772ff66fbc87aca34cb165f695255b39343fcfc832 - AP:0.3633699633699634
    ID:4590d7d47f521df62f3bcb0bf74d1bca861d94ade614d8afc912d1009d607b94 - AP:0.3643580243649534
    ID:708eb41a3fc8f2b6cd1f529cdf38dc4ad5d5f00ad30bdcba92884f37ff78d614 - AP:0.36458634885291574

    watershed_centers_basic top 20 worst offenders:
    ID:a7f767ca9770b160f234780e172aeb35a50830ba10dc49c526f4712451abe1d2 - AP:0.22396802282385808
    ID:a7f6194ddbeaefb1da571226a97785d09ccafc5893ce3c77078d2040bccfcb77 - AP:0.3410766963499296
    ID:93cfd412c7de5210bbd262ec3a602cfea65072e9272e9fce9b5339a5b9436eb7 - AP:0.35492063492063497
    ID:55f98f43c152aa0dc8bea513f8ba558cc57494b81ae4ee816977816e79629c50 - AP:0.3771687710022113
    ID:58c593bcb98386e7fd42a1d34e291db93477624b164e83ab2afa3caa90d1d921 - AP:0.4
    ID:a0325cb7aa59e9c0a75e64ba26855d8032c46161aa4bca0c01bac5e4a836485e - AP:0.40781079236194706
    ID:45f059cf21d85ecfce0eb93260516f1e2443d210e9a52f9ae2271d604aa3fcc5 - AP:0.4240860215053764
    ID:589f86dee5b480a88dd4f77eeaffe2c4d70aefdf879a4096dde1fa4d41055b8f - AP:0.43999999999999995
    ID:8d29c5a03e0560c8f9338e8eb7bccf47930149c8173f9ba4b9279fb87d86cf6d - AP:0.44716286214176515
    ID:a31deaf0ac279d5f34fb2eca80cc2abce6ef30bd64e7aca40efe4b2ba8e9ad3d - AP:0.47415170796598394
    ID:e49fc2b4f1f39d481a6525225ab3f688be5c87f56884456ad54c953315efae83 - AP:0.4828496955569509
    ID:4c032609d377bd980e01f888e0b298600bf8af0e33c4271a1f3aaf76964dce06 - AP:0.5060908084163899
    ID:693bc64581275f04fc456da74f031d583733360a1f6032fa38b3fbf592ff4352 - AP:0.5200884123706315
    ID:e5384c905e9879cb6e8ff5250fb03155bc1db035d8dde458eece9078b7de8ff1 - AP:0.5449468085106383
    ID:29780b28e6a75fac7b96f164a1580666513199794f1b19a5df8587fe0cb59b67 - AP:0.5482974971365064
    ID:0d3640c1f1b80f24e94cc9a5f3e1d9e8db7bf6af7d4aba920265f46cadc25e37 - AP:0.5507161172161171
    ID:bc115ff727e997a88f7cfe4ce817745731a6c753cb9fab6a36e7e66b415a1d3d - AP:0.5509506833036244
    ID:4d14a3629b6af6de86d850be236b833a7bfcbf6d8665fd73c6dc339e06c14607 - AP:0.5533807690998788
    ID:947c0d94c8213ac7aaa41c4efc95d854246550298259cf1bb489654d0e969050 - AP:0.5720247329948899
    ID:708eb41a3fc8f2b6cd1f529cdf38dc4ad5d5f00ad30bdcba92884f37ff78d614 - AP:0.5857612840719435

    watershed_contours_basic 3px-wide contours top 20 worst offenders:
    ID:7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80 - AP:0.0
    ID:e1bcb583985325d0ef5f3ef52957d0371c96d4af767b13e48102bca9d5351a9b - AP:0.2678571428571429
    ID:93cfd412c7de5210bbd262ec3a602cfea65072e9272e9fce9b5339a5b9436eb7 - AP:0.2894444444444445
    ID:58c593bcb98386e7fd42a1d34e291db93477624b164e83ab2afa3caa90d1d921 - AP:0.29166666666666663
    ID:9c95eae11da041189e84cda20bdfb75716a6594684de4b6ce12a9aaadbb874c9 - AP:0.29944099378881994
    ID:9cbc0700317361236a9fca2eb1f8f79e3a7da17b1970c179cf453921a6136001 - AP:0.33333333333333337
    ID:d1dbc6ee7c44a7027e935d040e496793186b884a1028d0e26284a206c6f5aff0 - AP:0.40584795321637435
    ID:b560dba92fbf2af785739efced50d5866c86dc4dada9be3832138bef4c3524d2 - AP:0.4128205128205128
    ID:fa73f24532b3667718ede7ac5c2e24ad7d3cae17b0a42ed17bbb81b15c28f4ae - AP:0.4128571428571429
    ID:c6216cdc42f61bc345434986db42e2ef9b9741aee3210b7a808e952e319d2305 - AP:0.43420710801024925
    ID:a0325cb7aa59e9c0a75e64ba26855d8032c46161aa4bca0c01bac5e4a836485e - AP:0.4429149709480275
    ID:947c0d94c8213ac7aaa41c4efc95d854246550298259cf1bb489654d0e969050 - AP:0.4605670404252712
    ID:55f98f43c152aa0dc8bea513f8ba558cc57494b81ae4ee816977816e79629c50 - AP:0.4694874701368345
    ID:308084bdd358e0bd3dc7f2b409d6f34cc119bce30216f44667fc2be43ff31722 - AP:0.47643798147342287
    ID:4bf6a5ec42032bb8dbbb10d25fdc5211b2fe1ce44b6e577ef89dbda17697d819 - AP:0.4870394265232975
    ID:f4b7c24baf69b8752c49d0eb5db4b7b5e1524945d48e54925bff401d5658045d - AP:0.4927884615384615
    ID:c53326fe49fc26b7fe602b9d8c0c2da2cb157690b44c2b9351a93f8d9bd8043d - AP:0.4960640194351136
    ID:feffce59a1a3eb0a6a05992bb7423c39c7d52865846da36d89e2a72c379e5398 - AP:0.5139402935840343
    ID:0121d6759c5adb290c8e828fc882f37dfaf3663ec885c663859948c154a443ed - AP:0.5187255802328857
    ID:094afe36759e7daffe12188ab5987581d405b06720f1d5acf3f2614f404df380 - AP:0.5198763483373454

    watershed_contours_basic 2px-wide contours top 20 worst offenders:
    ID:7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80 - AP:0.0
    ID:58c593bcb98386e7fd42a1d34e291db93477624b164e83ab2afa3caa90d1d921 - AP:0.29166666666666663
    ID:f4b7c24baf69b8752c49d0eb5db4b7b5e1524945d48e54925bff401d5658045d - AP:0.4575
    ID:93cfd412c7de5210bbd262ec3a602cfea65072e9272e9fce9b5339a5b9436eb7 - AP:0.47619047619047616
    ID:a0325cb7aa59e9c0a75e64ba26855d8032c46161aa4bca0c01bac5e4a836485e - AP:0.5163147007943455
    ID:9cbc0700317361236a9fca2eb1f8f79e3a7da17b1970c179cf453921a6136001 - AP:0.5261904761904762
    ID:55f98f43c152aa0dc8bea513f8ba558cc57494b81ae4ee816977816e79629c50 - AP:0.5405391123739638
    ID:947c0d94c8213ac7aaa41c4efc95d854246550298259cf1bb489654d0e969050 - AP:0.5750936773042936
    ID:0121d6759c5adb290c8e828fc882f37dfaf3663ec885c663859948c154a443ed - AP:0.5807226696186871
    ID:29780b28e6a75fac7b96f164a1580666513199794f1b19a5df8587fe0cb59b67 - AP:0.59639018266153
    ID:c04fa1a74a980d790ba6f3e595fd9851f14370bb71c7cbb7846c33ca9d72687f - AP:0.6115091426350652
    ID:c53326fe49fc26b7fe602b9d8c0c2da2cb157690b44c2b9351a93f8d9bd8043d - AP:0.6133285172451963
    ID:693bc64581275f04fc456da74f031d583733360a1f6032fa38b3fbf592ff4352 - AP:0.6213309206829727
    ID:d1dbc6ee7c44a7027e935d040e496793186b884a1028d0e26284a206c6f5aff0 - AP:0.6374166554493652
    ID:9c95eae11da041189e84cda20bdfb75716a6594684de4b6ce12a9aaadbb874c9 - AP:0.6408375054896794
    ID:e81c758e1ca177b0942ecad62cf8d321ffc315376135bcbed3df932a6e5b40c0 - AP:0.6431770173544276
    ID:8e507d58f4c27cd2a82bee79fe27b069befd62a46fdaed20970a95a2ba819c7b - AP:0.6482142857142857
    ID:e1bcb583985325d0ef5f3ef52957d0371c96d4af767b13e48102bca9d5351a9b - AP:0.6482142857142857
    ID:ad473063dab4bf4f2461d9a99a9c0166d4871f156516d9e0a523484e7cf2258d - AP:0.6554818622591201
    ID:3d0ca3498d97edebd28dbc7035eced40baa4af199af09cbb7251792accaa69fe - AP:0.6666666666666665

Notes:

    - Method kainz_miccai15: Given nuclei sem seg (foreground/background) and cell center, generate instance masks
    - Method 2: Given nuclei sem seg (foreground/background) and 4-pel-wide contours, generate instance masks
Refs:
    Scikit-Image Morphology library @ http://scikit-image.org/docs/0.13.x/api/skimage.morphology.html?highlight=morphology#module-skimage.morphology
    - morph.watershed
        http://scikit-image.org/docs/0.13.x/api/skimage.morphology.html?highlight=morphology#skimage.morphology.watershed

    SciPy Multi-dimensional image processing (ndimage) @ https://docs.scipy.org/doc/scipy-0.19.1/reference/ndimage.html
    - ndi.label
        https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.ndimage.label.html#scipy.ndimage.label
    - ndi.binary_fill_holes
        https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.ndimage.binary_fill_holes.html#scipy.ndimage.binary_fill_holes
    - ndi.distance_transform_edt
        https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.ndimage.distance_transform_edt.html#scipy.ndimage.distance_transform_edt
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# TODO: Add two more methods:
# https://www.kaggle.com/c/data-science-bowl-2018/discussion/47690#279721
# 'watershed_segmentation' and 'random_walker_segmentation'
# http://www.scipy-lectures.org/packages/scikit-image/index.html#marker-based-methods

# TODO: See https://www.kaggle.com/c/data-science-bowl-2018/discussion/47690#270974

# TODO: https://www.kaggle.com/c/data-science-bowl-2018/discussion/47690#272017 for distance transform approach

# TODO: See https://www.kaggle.com/c/data-science-bowl-2018/discussion/47690#284980

# TODO: See https://www.kaggle.com/c/data-science-bowl-2018/discussion/48825#278139

# TODO: See http://blog.wolfram.com/2012/01/04/how-to-count-cells-annihilate-sailboats-and-warp-the-mona-lisa/

import os
import numpy as np
from tqdm import tqdm
import cv2
from skimage.io import imread
from scipy import ndimage as ndi
from scipy.stats import itemfreq
from skimage import morphology as morph # import skimage.morphology as morph ?
from skimage.filters import threshold_otsu
from itertools import product

from contours import inst_masks_centroids_to_label, inst_masks_contours_to_label
from metrics import average_precision
from segment import save_inst_masks, save_labeled_mask
#TODO Remove the import below that is only used for debugging
from visualize import archive_images, archive_instances, archive_images_with_contours_and_centers
from dataset import DSB18Dataset, _DSB18_DATASET, _DEFAULT_DSB18_OPTIONS

# Untuned hyperparams
_UNTUNED_MIN_NUCLEUS_SIZE = 20

_DEBUG_SAVE_STEPS = False
_DEBUG_SAVE_STEPS_FOLDER = "c:/temp/steps"

_DEFAULT_PROC_OPTIONS = { # See constructor for explanation of options
    'method': 'watershed_contours_basic',
    'mode': 'both' ,
    'contour_thickness': 2, # Because of the downsampling occuring in an FCN, use at least 4-pixel-thick contours.
    'min_nucleus_size': _UNTUNED_MIN_NUCLEUS_SIZE,
    'save_preds': True
    }

class Post(object):
    """Post-processor.
    """

    def __init__(self, ds_root=_DSB18_DATASET, options=_DEFAULT_PROC_OPTIONS):
        """Initialize the post-processing object
        Args:
            options: see below
        Options:
            method: Method to use ['connected_components' | 'watershed_centers_basic' | 'watershed_contours_basic']
            min_nucleus_size: Minimum nucleus size (anything smaller will be discarded as noise)
            save_preds: If True, preds are returned and saved to disk; they are just returned otherwise
            mode: Save individual instance masks or single labeled mask? ['instance_masks' | 'labeled_mask' | 'both']
        """
        # Only options supported in this initial implementation
        self.options = options
        assert(self.options['method'] in ['connected_components', 'watershed_centers_basic', 'watershed_contours_basic'])
        assert(self.options['mode'] in ['instance_masks', 'labeled_mask', 'both'])

        # Set paths and file names
        self._ds_root = ds_root
        self._test_folder = self._ds_root + '/stage2_test_final'
        self._test_IDs_file = self._ds_root + '/test.txt'

        # Load ID file
        self._load_ID_file()


    def _load_ID_file(self):
        """Load the ID file and create all the nuclei and contour file paths
        Returns:
              True if ID file was loaded, False if ID files wasn't found
        """
        if not os.path.exists(self._test_IDs_file):
            return False

        # Read in the IDs and build paths to predicted labels
        with open(self._test_IDs_file, 'r') as f:
            self._test_IDs = f.readlines()
            self._test_IDs = [ID.rstrip() for ID in self._test_IDs]

        # Build file and folder paths
        self._sem_labels_paths = [self._test_folder + '/' + idx + '/pred_sem_label/' + idx + '.png' for idx in self._test_IDs]
        sub_folder = '/pred_contours_{}px/'.format(self.options['contour_thickness'])
        self._sem_contours_paths = [self._test_folder + '/' + idx + sub_folder + idx + '.png' for idx in self._test_IDs]
        self._labeled_mask_paths = [self._test_folder + '/' + idx + '/pred_labeled_mask/' + idx + '.png' for idx in self._test_IDs]
        self._inst_masks_folders = [self._test_folder + '/' + idx + '/pred_inst_masks/' for idx in self._test_IDs]

        return True


    def process(self):
        """Run the post-processing.
        Args:
            bin_sem_labels: List of thresholded predicted semantic segmentations [(H, W uint8)]
            bin_center_markers: List of center markers to be used by watershed seg [(H, W uint8)]
            bin_contour_markers: List of contour markers to be used by watershed seg [(H, W uint8)]
            raw_sem_labels: List of unthresholded predicted semantic segmentations (#, (H, W, 1)) or (#, (H, W)) ?
            raw_contours: List of unthresholded contours to be used by watershed seg (#, (H, W, 1)) or (#, (H, W)) ?
        Returns:
            inst_maskss: List of list of generated instance masks [[(H, W int32)]
            or/and
            labeled_masks: List of images with labeled instances (#, (H, W, 1)) or (#, H, W)) ?
        """
        # Load segmentations
        bin_sem_labels, bin_sem_contours = [], []
        with tqdm(total=len(self._test_IDs), desc="Loading nuclei+contour segmentations", ascii=True, ncols=100) as pbar:
            for n, id_ in enumerate(self._test_IDs):
                pbar.update(1)
                bin_sem_labels.append(imread(self._sem_labels_paths[n]))
                bin_sem_contours.append(imread(self._sem_contours_paths[n]))

        # Process the samples
        inst_masks, labeled_masks = self.process_samples(bin_sem_labels, bin_contour_markers=bin_sem_contours)

        # Save the results to disk
        if self.options['save_preds']:
            with tqdm(total=len(self._test_IDs), desc="Saving results to disk", ascii=True, ncols=100) as pbar:
                for n in range(len(self._test_IDs)):
                    pbar.update(1)
                    if self.options['mode'] in ['instance_masks', 'both']:
                        save_inst_masks(inst_masks[n], self._inst_masks_folders[n])
                    if self.options['mode'] in ['labeled_mask', 'both']:
                        save_labeled_mask(labeled_masks[n], self._labeled_mask_paths[n])

    def process_samples(self, bin_sem_labels, bin_center_markers=None, bin_contour_markers=None,
                        raw_sem_labels=None, raw_contours=None):
        """Run the post-processing on selected samples.
        Args:
            bin_sem_labels: List of thresholded predicted semantic segmentations [(H, W uint8)]
            bin_center_markers: List of center markers to be used by watershed seg [(H, W uint8)]
            bin_contour_markers: List of contour markers to be used by watershed seg [(H, W uint8)]
            raw_sem_labels: List of unthresholded predicted semantic segmentations (#, (H, W, 1)) or (#, (H, W)) ?
            raw_contours: List of unthresholded contours to be used by watershed seg (#, (H, W, 1)) or (#, (H, W)) ?
        Returns:
            inst_masks: List of list of generated instance masks [[(H, W int32)]
            or
            labeled_masks: List of images with labeled instances (#, (H, W, 1)) or (#, H, W)) ?
        """
        # TODO Add support for raw semantic labels and contours?
        if self.options['method'] == 'watershed_centers_basic':
            return self.watershed_centers_basic(bin_sem_labels, bin_center_markers)
        elif self.options['method'] == 'watershed_contours_basic':
            return self.watershed_contours_basic(bin_sem_labels, bin_contour_markers)
        elif self.options['method'] == 'connected_components':
            return self.connected_components(bin_sem_labels)
        elif self.options['method'] == 'kainz_miccai15':
            return self.kainz_miccai15()
        elif self.options['method'] == 'bhargava_17':
            return self.bhargava_17()

    ###
    ### Post-processors
    ###
    def kainz_miccai15(self, sem_labels):
        """Process instance masks.
        Args:
            inst_masks: List of list of unprocessed instance masks (#, (#, H, W)) or (#, (#, H, W), 1) ?
        Returns:
            post_inst_masks: List of list of processed instance masks (#, (#, H, W)) or (#, (#, H, W), 1) ?
        Refs:
            https://github.com/pkainz/MICCAI2015
        Abstract:
            Automated cell detection in histopathology images is a hard
            problem due to the large variance of cell shape and appearance. We show
            that cells can be detected reliably in images by predicting, for each pixel
            location, a monotonous function of the distance to the center of the clos-
            est cell. Cell centers can then be identified by extracting local extremums
            of the predicted values.
        """
        # TODO bhargava_17
        return NotImplemented


    def bhargava_17(self, sem_labels):
        """Process instance masks.
        Args:
            inst_masks: List of list of unprocessed instance masks (#, (#, H, W)) or (#, (#, H, W), 1) ?
        Returns:
            post_inst_masks: List of list of processed instance masks (#, (#, H, W)) or (#, (#, H, W), 1) ?
        Refs:
            Two different code repos:
                https://github.com/neerajkumarvaid/NucleiSegmentation
                https://drive.google.com/open?id=0ByERBiBsEbuTRkFpeHpmUENPRjQ
            https://surabhibhargava.github.io/Detection-and-Segmentation-of-Nuclei-in-Computational-Pathology/
            http://www.iitg.ernet.in/amitsethi/posters/16.SharmaBhargava.pdf
        Abstract:
            Automated cell detection in histopathology images is a hard
            problem due to the large variance of cell shape and appearance. We show
            that cells can be detected reliably in images by predicting, for each pixel
            location, a monotonous function of the distance to the center of the clos-
            est cell. Cell centers can then be identified by extracting local extremums
            of the predicted values.
        """
        # TODO bhargava_17
        return NotImplemented


    def connected_components(self, bin_sem_labels):
        """Compute instance masks using connected components and no post-processing whatsoever.
        Args:
            bin_sem_labels: List of thresholded predicted semantic segmentations [(H,W) uint8]
        Returns:
            inst_masks, labeled_masks: List of list of instance masks [[(H,W) int32]] and list of labeled masks
        """
        # Compute instance masks using connected components
        inst_masks, labeled_masks = [], []
        with tqdm(total=len(bin_sem_labels), desc="Extracting connected components", ascii=True, ncols=100) as pbar:
            for bin_sem_label in bin_sem_labels:
                pbar.update(1)
                labeled_mask, _ = self.label(bin_sem_label)
                labeled_masks.append(labeled_mask)
                inst_masks.append(self.labeled_mask_to_inst_masks(labeled_mask))
        return inst_masks, labeled_masks


    def watershed_centers_basic(self, bin_sem_labels, bin_center_markers):
        """Compute instance masks using watershed of thresholded predicted labels and center markers.
        Args:
            bin_sem_labels: List of thresholded predicted semantic segmentations [(H,W) uint8]
            bin_center_markers: List of center markers to be used by watershed seg [(H,W) uint8]
        Returns:
            inst_masks: List of list of instance masks [[(H, W, 1) int32]]
            or
            labeled_masks: List of images with labeled instances (#, (H, W, 1)) or (#, H, W)) ?
        Refs:
            From pipelines.py:
            def watershed_centers(mask, center, config, save_output=True):
                watershed_center = Step(name='watershed_centers',
                                        transformer=WatershedCenter(),
                                        input_steps=[mask, center],
                                        adapter={'images': ([(mask.name, 'binarized_images')]),
                                                 'contours': ([(center.name, 'binarized_images')]),
                                                 },
                                        cache_dirpath=config.env.cache_dirpath,
                                        save_output=save_output)

                drop_smaller = Step(name='drop_smaller',
                                    transformer=Dropper(**config.dropper),
                                    input_steps=[watershed_center],
                                    adapter={'labels': ([('watershed_center', 'detached_images')]),
                                             },
                                    cache_dirpath=config.env.cache_dirpath,
                                    save_output=save_output)

                binary_fill = Step(name='binary_fill',
                                   transformer=BinaryFillHoles(),
                                   input_steps=[drop_smaller],
                                   adapter={'images': ([('drop_smaller', 'labels')]),
                                            },
                                   cache_dirpath=config.env.cache_dirpath,
                                   save_output=save_output)

                return binary_fill
        """
        # Compute instance masks using watershed of thresholded predicted labels and center markers
        inst_masks, labeled_masks = [], []
        with tqdm(total=len(bin_sem_labels), desc="Extracting instances with watershed and center markers", ascii=True, ncols=100) as pbar:
            for bin_sem_label, bin_centers in zip(bin_sem_labels, bin_center_markers):
                pbar.update(1)
                res = self.watershed_from_centers(bin_sem_label, bin_centers)
                res = self.drop_small(res)
                labeled_mask = self.fill_holes_per_blob(res)
                labeled_masks.append(labeled_mask)
                inst_masks.append(self.labeled_mask_to_inst_masks(labeled_mask))
        return inst_masks, labeled_masks


    def watershed_contours_basic(self, bin_sem_labels, bin_contour_markers):
        """Compute instance masks using watershed of thresholded predicted labels and contour markers.
        Args:
            bin_sem_labels: List of thresholded predicted semantic segmentations [(H,W) uint8]
            bin_contour_markers: List of contour markers to be used by watershed seg [(H,W) uint8]
        Returns:
            inst_masks: List of np arrays of generated instance masks [nps(#,H,W) int32)]
            or
            labeled_masks: List of images with labeled instances [(H, W, 1) int32]
        Refs:
            From pipelines.py:
            def watershed_contours(mask, contour, config, save_output=True):
                watershed_contour = Step(name='watershed_contour',
                                         transformer=WatershedContour(),
                                         input_steps=[mask, contour],
                                         adapter={'images': ([(mask.name, 'binarized_images')]),
                                                  'contours': ([(contour.name, 'binarized_images')]),
                                                  },
                                         cache_dirpath=config.env.cache_dirpath,
                                         save_output=save_output)

                drop_smaller = Step(name='drop_smaller',
                                    transformer=Dropper(**config.dropper),
                                    input_steps=[watershed_contour],
                                    adapter={'labels': ([('watershed_contour', 'detached_images')]),
                                             },

                                    cache_dirpath=config.env.cache_dirpath,
                                    save_output=save_output)
                return drop_smaller
        """
        # Compute instance masks using watershed of thresholded predicted labels and contour markers
        inst_masks, labeled_masks = [], []
        with tqdm(total=len(bin_sem_labels), desc="Extracting instances with watershed and contours", ascii=True, ncols=100) as pbar:
            for bin_sem_label, bin_contours in zip(bin_sem_labels, bin_contour_markers):
                pbar.update(1)
                if _DEBUG_SAVE_STEPS:
                    _DEBUG_IMAGES = [bin_sem_label.copy(), bin_contours.copy()]
                    _DEBUG_TITLES = ["bin_sem_label", "bin_contours"]
                res = self.watershed_from_contours(bin_sem_label, bin_contours)
                if _DEBUG_SAVE_STEPS:
                    _DEBUG_IMAGES.append(res.copy())
                    _DEBUG_TITLES.append("after_watershed_from_contours")
                res = self.drop_small(res)
                if _DEBUG_SAVE_STEPS:
                    _DEBUG_IMAGES.append(res.copy())
                    _DEBUG_TITLES.append("after_drop_small")
                if _DEBUG_SAVE_STEPS:
                    archive_images(_DEBUG_IMAGES, _DEBUG_SAVE_STEPS_FOLDER, _DEBUG_TITLES)
                labeled_masks.append(res)
                inst_masks.append(self.labeled_mask_to_inst_masks(res))
        return inst_masks, labeled_masks


    def UNUSED_watershed_contours_raw(self, raw_sem_labels, raw_contours):
        """Process semantic segmentatiorn using watershed (heavier implementation).
        Args:
            raw_sem_labels: List of unthresholded predicted semantic segmentations (#, (H, W, 1)) or (#, (H, W)) ?
            raw_contours: List of unthresholded contours to be used by watershed seg (#, (H, W, 1)) or (#, (H, W)) ?
        Returns:
            inst_masks: List of list of generated instance masks (#, (#, H, W)) or (#, (#, H, W, 1)) ?
            or
            labeled_masks: List of images with labeled instances (#, (H, W, 1)) or (#, H, W)) ?
        Refs:
            From postprocessing.py:
            def postprocess_samples(image, contour):
                cleaned_mask = UNUSED_clean_mask(image, contour)
                good_markers = UNUSED_get_markers(cleaned_mask, contour)
                good_distance = get_distance(cleaned_mask)

                labels = morph.watershed(-good_distance, good_markers, mask=cleaned_mask)

                labels = UNUSED_add_dropped_water_blobs(labels, cleaned_mask)

                m_thresh = threshold_otsu(image)
                initial_mask_binary = (image > m_thresh).astype(np.uint8)
                labels = UNUSED_drop_artifacts_per_label(labels, initial_mask_binary)

                labels = drop_small(labels, min_size=20)
                labels = fill_holes_per_blob(labels)

                return labels
        """
        # TODO
        pred_inst_masks = []
        for sem_label, contours in zip(raw_sem_labels, raw_contours):
            res = self.UNUSED_watershed_from_raw(sem_label, contours)
            if self.options['return_inst_masks']:
                res = self.labeled_mask_to_inst_masks(res)
            pred_inst_masks.append(res)
        return pred_inst_masks

    #
    # Watershed helpers
    #
    def watershed_from_centers(self, bin_sem_label, bin_centers):
        """Segment thresholded predicted labels using watershed segmentation from centers
        Args:
            bin_sem_label: Thresholded predicted semantic segmentations (H,W,1)
            bin_centers: Thresholded centers to be used by watershed seg (H,W)
        """
        # Distance and markers must have the same shape: choose (H,W)
        if len(bin_sem_label.shape)==3 and bin_sem_label.shape[2]==1:
            bin_sem_label = np.squeeze(bin_sem_label, axis=-1)
        if len(bin_centers.shape)==3 and bin_centers.shape[2]==1:
            bin_centers = np.squeeze(bin_centers, axis=-1)

        # Get the exact euclidean distance to background (0) for the predicted nuclei areas
        distance = self.get_distance(bin_sem_label) # (H,W float 64)

        # Label each connected component in the image of predicted cell centers
        markers, nr_blobs = self.label(bin_centers) # (W,H int32), count

        # Find the watershed basins in the distance image flooded from the given markers
        # Because with morph.watershed the **lowest** value points are labeled first, we pass in neg(distance) instead.
        # With the third param, we tell the watershed module to ignore everything outside the predicted nuclei areas
        labeled = morph.watershed(-distance, markers, mask=bin_sem_label)

        # NOTE: The following was dropped as it resulted in a significant drop in AP:
        #
        # Worst offenders without the removed code:
        # ID:308084bdd358e0bd3dc7f2b409d6f34cc119bce30216f44667fc2be43ff31722 - AP:0.47643798147342287 - idx:14
        # ID:4217e25defac94ff465157d53f5a24b8a14045b763d8606ec4a97d71d99ee381 - AP:0.5503200228958354 - idx:20
        # ID:1d02c4b5921e916b9ddfb2f741fd6cf8d0e571ad51eb20e021c826b5fb87350e - AP:0.6323873517786561 - idx:16
        # ID:4e07a653352b30bb95b60ebc6c57afbc7215716224af731c51ff8d430788cd40 - AP:0.6939171809775554 - idx:8
        # ID:8f6e49e474ebb649a1e99662243d51a46cc9ba0c9c8f1efe2e2b662a81b48de1 - AP:0.8155863253191699 - idx:5
        # ID:ed5be4b63e9506ad64660dd92a098ffcc0325195298c13c815a73773f1efc279 - AP:0.8286707266933908 - idx:3
        # ID:353ab00e964f71aa720385223a9078b770b7e3efaf5be0f66e670981f68fe606 - AP:0.8446176828276029 - idx:10
        # ID:abd8dde78f8d37b68b28da67459371ed65f0a575523e94bc4ecbc88e6fedf0d0 - AP:0.8666666666666669 - idx:13
        # ID:b4d902d42c93dea77b541456f8d905f35eeb24fc3a5b0b15b5678d78e0aabe0c - AP:0.8823529411764708 - idx:4
        # ID:a7f767ca9770b160f234780e172aeb35a50830ba10dc49c526f4712451abe1d2 - AP:0.8930402930402929 - idx:18
        #
        # Worst offenders with the "add back" code
        # ID:308084bdd358e0bd3dc7f2b409d6f34cc119bce30216f44667fc2be43ff31722 - AP:0.37068899207755457 - idx:14
        # ID:1d02c4b5921e916b9ddfb2f741fd6cf8d0e571ad51eb20e021c826b5fb87350e - AP:0.42729286655373605 - idx:16
        # ID:4217e25defac94ff465157d53f5a24b8a14045b763d8606ec4a97d71d99ee381 - AP:0.5232783157781324 - idx:20
        # ID:ed5be4b63e9506ad64660dd92a098ffcc0325195298c13c815a73773f1efc279 - AP:0.6324095421038027 - idx:3
        # ID:4e07a653352b30bb95b60ebc6c57afbc7215716224af731c51ff8d430788cd40 - AP:0.6619212410657151 - idx:8
        # ID:07fb37aafa6626608af90c1e18f6a743f29b6b233d2e427dcd1102df6a916cf5 - AP:0.6956153758410263 - idx:6
        # ID:353ab00e964f71aa720385223a9078b770b7e3efaf5be0f66e670981f68fe606 - AP:0.7090760717590181 - idx:10
        # ID:175dbb364bfefc9537931144861c9b6e08934df3992782c669c6fe4234319dfc - AP:0.7349640039729471 - idx:9
        # ID:8f6e49e474ebb649a1e99662243d51a46cc9ba0c9c8f1efe2e2b662a81b48de1 - AP:0.7366945430619456 - idx:5
        # ID:6b61ab2e3ff0e2c7a55fd71e290b51e142555cf82bc7574fc27326735e8acbd1 - AP:0.7540038738142536 - idx:1
        #
        # Add back areas of the predicted nuclei areas that were dropped during the whole process
        # dropped, nr_blobs = self.label(bin_sem_label - (labeled > 0))
        # dropped = np.where(dropped > 0, dropped + nr_blobs, 0)
        # correct_labeled = dropped + labeled
        #
        # return self.relabel(correct_labeled)
        return labeled


    def watershed_from_contours(self, bin_sem_label, bin_contours):
        """Segment thresholded predicted labels using watershed segmentation from contours
        Args:
            bin_sem_label: Thresholded predicted semantic segmentations nparray(H,W uint8)
            bin_contours: Thresholded contours nparray(H,W uint8)
            open_closed_contours: Thresholded contours nparray(H,W uint8)
        """
        # Mask out areas that are not part of the contours in the thresholded predicted labels
        # We should be left with the centers of the nuclei
        centers = np.where(bin_contours == bin_contours.max(), 0, bin_sem_label)

        # Run the regular watershed from centers
        return self.watershed_from_centers(bin_sem_label, centers)


    def UNUSED_watershed_from_raw(self, raw_sem_label, raw_contours):
        """Segment unthresholded predicted labels using watershed segmentation from contours
        Args:
            raw_sem_label: Unthresholded predicted semantic segmentations (H, W, 1) or (H, W) ?
            raw_contours: Untesholed contours to be used by watershed seg (H, W, 1) or (H, W) ?
        """
        # Mask out areas that are not part of the contours in the thresholded predicted labels
        # We should be left with the centers of the nuclei
        cleaned_mask = self.UNUSED_clean_mask(raw_sem_label, raw_contours)
        good_markers = self.UNUSED_get_markers(cleaned_mask, raw_contours)
        good_distance = self.get_distance(cleaned_mask)

        labels = morph.watershed(-good_distance, good_markers, mask=cleaned_mask)

        labels = self.UNUSED_add_dropped_water_blobs(labels, cleaned_mask)

        m_thresh = threshold_otsu(raw_sem_label)
        initial_mask_binary = (raw_sem_label > m_thresh).astype(np.uint8)
        labels = self.UNUSED_drop_artifacts_per_label(labels, initial_mask_binary)

        labels = self.drop_small(labels)
        labels = self.fill_holes_per_blob(labels)

        return labels

    #
    # Various helpers
    #
    def get_distance(self, bin_sem_label):
        distance = ndi.distance_transform_edt(bin_sem_label)
        return distance


    def labeled_mask_to_inst_masks(self, labeled_mask):
        """"Convert an image of labeled masks to indivudual instance masks
            Args:
                labeled_mask: Labeled instances in a single mask image (H, W, 1) int32
            Returns:
                masks:  List of instance masks [(H, W, 1) uint32]
        """
        nr_true = labeled_mask.max()
        masks = []
        for i in range(1, nr_true + 1):
            msk = labeled_mask.copy()
            msk[msk != i] = 0.
            msk[msk == i] = 255.
            masks.append(msk)
        if not masks:
            return np.asarray([labeled_mask])
        else:
            return np.asarray(masks)


    def fill_holes_per_blob(self, labeled_mask):
        """Close holes in image labels fixing labels one at a time (not the image as a single whole)."""
        image_cleaned = np.zeros_like(labeled_mask)
        for i in range(1, labeled_mask.max() + 1):
            mask = np.where(labeled_mask == i, 1, 0)
            mask = ndi.binary_fill_holes(mask)
            image_cleaned = image_cleaned + mask * i
        return image_cleaned


    def relabel(self, labeled_mask):
        """Relabel non-contiguous image labels. Necessary when small objects are dropped."""
        h, w = labeled_mask.shape

        relabel_dict = {}

        for i, k in enumerate(np.unique(labeled_mask)):
            if k == 0:
                relabel_dict[k] = 0
            else:
                relabel_dict[k] = i
        for i, j in product(range(h), range(w)):
            labeled_mask[i, j] = relabel_dict[labeled_mask[i, j]]
        return labeled_mask


    def drop_small(self, labeled_mask):
        """Remove small instance masks"""
        labeled_mask = morph.remove_small_objects(labeled_mask, min_size=self.options['min_nucleus_size'])
        return self.relabel(labeled_mask)


    def label(self, bin_sem_label):
        """Get connected components
            Args:
                bin_sem_label: Tnresholded semantic segmentations (H, W, 1) uint8
            Returns:
                labeled_masks: Labeled instances in a single mask image (H, W, 1) int32
                nr_blobs: How many objects were found
        """
        # TODO Make sure you use the same implementation of "label" connected compnents everywhere
        labeled, nr_blobs = ndi.label(bin_sem_label)
        return labeled, nr_blobs

    #
    # Unused
    #
    def UNUSED_relabel_random_colors(self, labeled_mask, max_colors=1000):
        """Relabel image labels with random colors."""
        keys = list(range(1, max_colors, 1))
        np.random.shuffle(keys)
        values = list(range(1, max_colors, 1))
        np.random.shuffle(values)
        funky_dict = {k: v for k, v in zip(keys, values)}
        funky_dict[0] = 0

        h, w = labeled_mask.shape

        for i, j in product(range(h), range(w)):
            labeled_mask[i, j] = funky_dict[labeled_mask[i, j]]
        return labeled_mask


    def UNUSED_drop_artifacts_per_label(self, labels, initial_mask):
        labels_cleaned = np.zeros_like(labels)
        for i in range(1, labels.max() + 1):
            component = np.where(labels == i, 1, 0)
            component_initial_mask = np.where(labels == i, initial_mask, 0)
            component = self.UNUSED_drop_artifacts(component, component_initial_mask)
            labels_cleaned = labels_cleaned + component * i
        return labels_cleaned


    def UNUSED_clean_mask(self, sem_label, contours):
        """Threshold raw semantic segmentation and raw contour predictions."""
        # threshold
        sem_label_thresh = threshold_otsu(sem_label)
        contours_thresh = threshold_otsu(contours)
        bin_sem_label = sem_label > sem_label_thresh
        bin_contours = contours > contours_thresh

        # combine thresholded contours and thresholded predicted cell segmentation and fill the cells
        m_ = np.where(bin_sem_label | bin_contours, 1, 0)
        m_ = ndi.binary_fill_holes(m_)

        # close what wasn't closed before
        area, radius = self.UNUSED_mean_blob_size(bin_sem_label)
        struct_size = int(1.25 * radius)
        struct_el = morph.disk(struct_size)
        m_padded = self.UNUSED_pad_mask(m_, pad=struct_size)
        m_padded = morph.binary_closing(m_padded, selem=struct_el)
        m_ = self.UNUSED_crop_mask(m_padded, crop=struct_size)

        # open to cut the real cells from the artifacts
        area, radius = self.UNUSED_mean_blob_size(bin_sem_label)
        struct_size = int(0.75 * radius)
        struct_el = morph.disk(struct_size)
        m_ = np.where(bin_contours & (~bin_sem_label), 0, m_)
        m_padded = self.UNUSED_pad_mask(m_, pad=struct_size)
        m_padded = morph.binary_opening(m_padded, selem=struct_el)
        m_ = self.UNUSED_crop_mask(m_padded, crop=struct_size)

        # join the connected cells with what we had at the beginning
        m_ = np.where(bin_sem_label | m_, 1, 0)
        m_ = ndi.binary_fill_holes(m_)

        # drop all the cells that weren't present at least in 25% of area in the initial mask
        m_ = self.UNUSED_drop_artifacts(m_, bin_sem_label, min_coverage=0.25)

        return m_


    def UNUSED_get_markers(self, bin_sem_label, raw_contours):
        # threshold
        c_thresh = threshold_otsu(raw_contours)
        c_b = raw_contours > c_thresh

        mk_ = np.where(c_b, 0, bin_sem_label)

        area, radius = self.UNUSED_mean_blob_size(bin_sem_label)
        struct_size = int(0.25 * radius)
        struct_el = morph.disk(struct_size)
        m_padded = self.UNUSED_pad_mask(mk_, pad=struct_size)
        m_padded = morph.erosion(m_padded, selem=struct_el)
        mk_ = self.UNUSED_crop_mask(m_padded, crop=struct_size)
        mk_, _ = self.label(mk_)
        return mk_


    def UNUSED_add_dropped_water_blobs(self, water, mask_cleaned):
        water_mask = (water > 0).astype(np.uint8)
        dropped = mask_cleaned - water_mask
        dropped, _ = self.label(dropped)
        dropped = np.where(dropped, dropped + water.max(), 0)
        water = water + dropped
        return water


    def UNUSED_drop_artifacts(self, mask_after, mask_pre, min_coverage=0.5):
        connected, nr_connected = self.label(mask_after)
        mask = np.zeros_like(mask_after)
        for i in range(1, nr_connected + 1):
            conn_blob = np.where(connected == i, 1, 0)
            initial_space = np.where(connected == i, mask_pre, 0)
            blob_size = np.sum(conn_blob)
            initial_blob_size = np.sum(initial_space)
            coverage = float(initial_blob_size) / float(blob_size)
            if coverage > min_coverage:
                mask = mask + conn_blob
            else:
                mask = mask + initial_space
        return mask


    def UNUSED_mean_blob_size(self, mask):
        labels, labels_nr = self.label(mask)
        if labels_nr < 2:
            mean_area = 1
            mean_radius = 1
        else:
            mean_area = int(itemfreq(labels)[1:, 1].mean())
            mean_radius = int(np.round(np.sqrt(mean_area / np.pi)))
        return mean_area, mean_radius


    def UNUSED_pad_mask(self, mask, pad):
        if pad <= 1:
            pad = 2
        h, w = mask.shape
        h_pad = h + 2 * pad
        w_pad = w + 2 * pad
        mask_padded = np.zeros((h_pad, w_pad))
        mask_padded[pad:pad + h, pad:pad + w] = mask
        mask_padded[pad - 1, :] = 1
        mask_padded[pad + h + 1, :] = 1
        mask_padded[:, pad - 1] = 1
        mask_padded[:, pad + w + 1] = 1

        return mask_padded


    def UNUSED_crop_mask(self, mask, crop):
        if crop <= 1:
            crop = 2
        h, w = mask.shape
        mask_cropped = mask[crop:h - crop, crop:w - crop]
        return mask_cropped


    ###
    ### Evaluation
    ###
    def UNUSED_eval_all_processors(self, gt_inst_masks, pred_labels, center_markers, contour_markers):
        """Evaluate mean average precision of all the post-processing technique.
        Args:
            gt_inst_masks: List of list of groundtruth instance masks (#, (#, H, W)) or (#, (#, H, W), 1) ?
            fake_pred_label: List of thesholded predicted semantic segmentations (#, H, W) or (#, H, W, 1) ?
        Returns:
            mAP: mean average precision.
        Notes:
            Available processors: # ['watershed_centers_basic' | 'watershed_contours_basic' | 'watershed_contours_heavy']
        """
        # Assume
        mAPs = {}
        mAP['watershed_centers_basic'] = self.eval_map(gt_inst_masks, self.watershed_centers_basic(pred_labels, center_markers))
        mAP['watershed_contours_basic'] = self.eval_map(gt_inst_masks, self.watershed_contours_basic(pred_labels, contour_markers))
        mAP['UNUSED_watershed_contours_raw'] = self.eval_map(gt_inst_masks, self.UNUSED_watershed_contours_raw(pred_labels, contour_markers))
        print("\nPost-processor mAPs:")
        for k, v in self.options.items():
            print("  {:20} {}".format(k, v))
        return mAPs

    def eval_map(self, gt_inst_masks, pred_inst_masks, track_offenders=False):
        """Evaluate mean average precision of a post-processing technique.
        Args:
            gt_inst_masks: List of np arrays of groundtruth instance masks [np(#, H, W).
            pred_inst_masks: List of np arrays of predicted instance masks [np(#, H, W, 1).
        Returns:
            mAP: mean average precision.
            worst_APs_idx: List of indices of the worst APs.
            worst_APs_val: List of values of the worst APs.
        """
        mAP = []
        with tqdm(total=len(gt_inst_masks), desc="Evaluating post-proc mAP", ascii=True, ncols=100) as pbar:
            for gt_masks, pred_masks in zip(gt_inst_masks, pred_inst_masks):
                pbar.update(1)
                mAP.append(average_precision(gt_masks, pred_masks)) # gt_masks (61, H, W) pred_inst_masks (44, H, W)
        if track_offenders:
            worst_APs_idx = sorted(range(len(mAP)), key=lambda k: mAP[k])
            worst_APs_val = [mAP[idx] for idx in worst_APs_idx]
            return np.mean(mAP), worst_APs_idx, worst_APs_val
        else:
            return np.mean(mAP)

    ###
    ### Debug utils
    ###
    def print_config(self):
        """Display configuration values."""
        print("\nPost-processor Configuration:")
        for k, v in self.options.items():
            print("  {:20} {}".format(k, v))


def test_processors_with_fake_preds():
    """Test post-processors using groundtruths as the predictions.
        This should give us an upper bound on the efficiency of each post-processor
        Results on 670 training samples:
        connected_components mAP: 0.7611403272506709
    """

    from dataset import DSB18Dataset, _DEFAULT_DSB18_OPTIONS

    # Load dataset using instance masks
    ds_options = _DEFAULT_DSB18_OPTIONS
    ds_options['mode'] = 'instance_masks'
    ds_options['in_memory'] = True
    ds_inst_masks = DSB18Dataset(phase='train_noval', options=ds_options)
    ds_inst_masks.print_config()

    # Get the gt instance masks
    _, gt_inst_masks, _ = ds_inst_masks.get_rand_samples_with_inst_masks(ds_inst_masks.train_size, 'train', deterministic=True)

    # Load dataset using semantic labels
    ds_options = _DEFAULT_DSB18_OPTIONS
    ds_options['mode'] = 'semantic_labels'
    ds_options['in_memory'] = True
    ds_sem_labels = DSB18Dataset(phase='train_noval', options=ds_options)
    ds_sem_labels.print_config()

    # Get the gt semantic labels (and their IDs so we can list worst offenders)
    _, bin_sem_labels, IDs = ds_sem_labels.get_rand_samples_with_sem_labels(ds_sem_labels.train_size, 'train', return_IDs=True, deterministic=True)
    bin_sem_labels = [np.squeeze(bin_sem_label) for bin_sem_label in bin_sem_labels]

    # Get the centroids of each instance mask and combine them in a single bin label
    bin_center_markers = []
    with tqdm(total=len(gt_inst_masks), desc="Compute instance mask centroids", ascii=True, ncols=100) as pbar:
        for gt_masks in gt_inst_masks:
            pbar.update(1)
            bin_center_markers.append(inst_masks_centroids_to_label(gt_masks))

    # Get the contours of each instance mask and combine them in a single bin label
    bin_contour_markers_4px = []
    with tqdm(total=len(gt_inst_masks), desc="Compute 4px-wide instance mask contours", ascii=True, ncols=100) as pbar:
        for gt_masks in gt_inst_masks:
            pbar.update(1)
            bin_contour_markers_4px.append(inst_masks_contours_to_label(gt_masks, thickness=4))
    bin_contour_markers_3px = []
    with tqdm(total=len(gt_inst_masks), desc="Compute 3px-wide instance mask contours", ascii=True, ncols=100) as pbar:
        for gt_masks in gt_inst_masks:
            pbar.update(1)
            bin_contour_markers_3px.append(inst_masks_contours_to_label(gt_masks, thickness=3))
    bin_contour_markers_2px = []
    with tqdm(total=len(gt_inst_masks), desc="Compute 2px-wide instance mask contours", ascii=True, ncols=100) as pbar:
        for gt_masks in gt_inst_masks:
            pbar.update(1)
            bin_contour_markers_2px.append(inst_masks_contours_to_label(gt_masks, thickness=2))

    # Parameters
    # print(bin_sem_labels[0].shape)
    # print(bin_contour_markers[0].shape)
    # print(bin_center_markers[0].shape)
    # num_samples = min(10, len(gt_inst_masks))
    # save_folder1 = "c:/temp/visualizations1"
    # save_folder2 = "c:/temp/visualizations2"
    # save_folder3 = "c:/temp/visualizations3"
    # save_folder4 = "c:/temp/visualizations4_ws_cent"
    # save_folder5 = "c:/temp/visualizations5_ws_cont"
    # save_folder5_offenders_gt = "c:/temp/visualizations5_ws_cont_offenders_gt"
    # save_folder5_offenders = "c:/temp/visualizations5_ws_cont_offenders"
    # archive_images_with_contours_and_centers(bin_sem_labels[0:num_samples], # [(H,W,1)]
    #                                          bin_contour_markers[0:num_samples], # [(H,W)]
    #                                          bin_center_markers[0:num_samples], # [(H,W)]
    #                                          save_folder1)
    # archive_images(bin_sem_labels[0:num_samples], save_folder2)

    # Number of worst offenders to track
    num_offenders = 20

    #
    # Evaluate connected_components
    #
    method = 'connected_components'

    # Instantiate post-processor
    post_options = _DEFAULT_PROC_OPTIONS
    post_options['method'] = method
    post = Post(post_options)
    post.print_config()

    # Run the post-processor
    pred_inst_masks, _ = post.process_samples(bin_sem_labels)

    # Evaluate the performance of the post-processor
    mAP, worst_APs_idx, worst_APs_val = post.eval_map(gt_inst_masks, pred_inst_masks, track_offenders=True)
    print("{} mAP: {}\nTop {} worst offenders:".format(method, mAP, num_offenders))
    bad_bin_sem_labels, bad_pred_inst_masks, bad_gt_inst_masks = [], [], []
    for n in range(num_offenders):
        print("ID:{} - AP:{}".format(IDs[worst_APs_idx[n]], worst_APs_val[n]))
        bad_bin_sem_labels.append(bin_sem_labels[worst_APs_idx[n]])
        bad_pred_inst_masks.append(pred_inst_masks[worst_APs_idx[n]])
        bad_gt_inst_masks.append(gt_inst_masks[worst_APs_idx[n]])

    #
    # Evaluate watershed_centers_basic
    #
    method = 'watershed_centers_basic'

    # Instantiate post-processor
    post_options = _DEFAULT_PROC_OPTIONS
    post_options['method'] = method
    post = Post(post_options)
    post.print_config()

    # Run the post-processor
    pred_inst_masks, _ = post.process_samples(bin_sem_labels, bin_center_markers)

    # Evaluate the performance of the post-processor
    mAP, worst_APs_idx, worst_APs_val = post.eval_map(gt_inst_masks, pred_inst_masks, track_offenders=True)
    print("{} mAP: {}\nTop {} worst offenders:".format(method, mAP, num_offenders))
    bad_bin_sem_labels, bad_pred_inst_masks, bad_gt_inst_masks = [], [], []
    for n in range(num_offenders):
        print("ID:{} - AP:{}".format(IDs[worst_APs_idx[n]], worst_APs_val[n]))
        bad_bin_sem_labels.append(bin_sem_labels[worst_APs_idx[n]])
        bad_pred_inst_masks.append(pred_inst_masks[worst_APs_idx[n]])
        bad_gt_inst_masks.append(gt_inst_masks[worst_APs_idx[n]])

    #
    # Evaluate watershed_contours_basic with 4px-wide contours
    #
    method = 'watershed_contours_basic'
    contour_width = '4px-wide'

    # Instantiate post-processor
    post_options = _DEFAULT_PROC_OPTIONS
    post_options['method'] = method
    post = Post(post_options)
    post.print_config()

    # Run the post-processor
    pred_inst_masks, _ = post.process_samples(bin_sem_labels, bin_contour_markers=bin_contour_markers_4px)

    # Evaluate the performance of the post-processor
    mAP, worst_APs_idx, worst_APs_val = post.eval_map(gt_inst_masks, pred_inst_masks, track_offenders=True)
    print("{} {} mAP: {}\nTop {} worst offenders:".format(method, contour_width, mAP, num_offenders))
    bad_bin_sem_labels, bad_pred_inst_masks, bad_gt_inst_masks = [], [], []
    for n in range(num_offenders):
        print("ID:{} - AP:{}".format(IDs[worst_APs_idx[n]], worst_APs_val[n]))
        bad_bin_sem_labels.append(bin_sem_labels[worst_APs_idx[n]])
        bad_pred_inst_masks.append(pred_inst_masks[worst_APs_idx[n]])
        bad_gt_inst_masks.append(gt_inst_masks[worst_APs_idx[n]])

    #
    # Evaluate watershed_contours_basic with 3px-wide contours
    #
    method = 'watershed_contours_basic'
    contour_width = '3px-wide'

    # Instantiate post-processor
    post_options = _DEFAULT_PROC_OPTIONS
    post_options['method'] = method
    post = Post(post_options)
    post.print_config()

    # Run the post-processor
    pred_inst_masks, _ = post.process_samples(bin_sem_labels, bin_contour_markers=bin_contour_markers_3px)

    # Evaluate the performance of the post-processor
    mAP, worst_APs_idx, worst_APs_val = post.eval_map(gt_inst_masks, pred_inst_masks, track_offenders=True)
    print("{} {} mAP: {}\nTop {} worst offenders:".format(method, contour_width, mAP, num_offenders))
    bad_bin_sem_labels, bad_pred_inst_masks, bad_gt_inst_masks = [], [], []
    for n in range(num_offenders):
        print("ID:{} - AP:{}".format(IDs[worst_APs_idx[n]], worst_APs_val[n]))
        bad_bin_sem_labels.append(bin_sem_labels[worst_APs_idx[n]])
        bad_pred_inst_masks.append(pred_inst_masks[worst_APs_idx[n]])
        bad_gt_inst_masks.append(gt_inst_masks[worst_APs_idx[n]])

    #
    # Evaluate watershed_contours_basic with 2px-wide contours
    #
    method = 'watershed_contours_basic'
    contour_width = '2px-wide'

    # Instantiate post-processor
    post_options = _DEFAULT_PROC_OPTIONS
    post_options['method'] = method
    post = Post(post_options)
    post.print_config()

    # Run the post-processor
    pred_inst_masks, _ = post.process_samples(bin_sem_labels, bin_contour_markers=bin_contour_markers_2px)

    # Evaluate the performance of the post-processor
    mAP, worst_APs_idx, worst_APs_val = post.eval_map(gt_inst_masks, pred_inst_masks, track_offenders=True)
    print("{} {} mAP: {}\nTop {} worst offenders:".format(method, contour_width, mAP, num_offenders))
    bad_bin_sem_labels, bad_pred_inst_masks, bad_gt_inst_masks = [], [], []
    for n in range(num_offenders):
        print("ID:{} - AP:{}".format(IDs[worst_APs_idx[n]], worst_APs_val[n]))
        bad_bin_sem_labels.append(bin_sem_labels[worst_APs_idx[n]])
        bad_pred_inst_masks.append(pred_inst_masks[worst_APs_idx[n]])
        bad_gt_inst_masks.append(gt_inst_masks[worst_APs_idx[n]])

        # archive_instances(bin_sem_labels[0:num_samples], gt_inst_masks[0:num_samples], save_folder3)
        # archive_instances(bin_sem_labels[0:num_samples], pred_inst_masks[0:num_samples], save_folder4)
        # archive_instances(bin_sem_labels[0:num_samples], pred_inst_masks[0:num_samples], save_folder5)

        # archive_instances(bad_bin_sem_labels, bad_pred_inst_masks, save_folder5_offenders)
        # archive_instances(bad_bin_sem_labels, bad_gt_inst_masks, save_folder5_offenders_gt)
        # archive_instances(bin_sem_labels[0:num_samples], pred_inst_masks[0:num_samples], save_folder4)

        # Worst offenders:
        # ID:308084bdd358e0bd3dc7f2b409d6f34cc119bce30216f44667fc2be43ff31722 - AP:0.37068899207755457 - idx:14
        # ID:1d02c4b5921e916b9ddfb2f741fd6cf8d0e571ad51eb20e021c826b5fb87350e - AP:0.42729286655373605 - idx:16
        # ID:4217e25defac94ff465157d53f5a24b8a14045b763d8606ec4a97d71d99ee381 - AP:0.5232783157781324 - idx:20
        # ID:ed5be4b63e9506ad64660dd92a098ffcc0325195298c13c815a73773f1efc279 - AP:0.6324095421038027 - idx:3
        # ID:4e07a653352b30bb95b60ebc6c57afbc7215716224af731c51ff8d430788cd40 - AP:0.6619212410657151 - idx:8
        # ID:07fb37aafa6626608af90c1e18f6a743f29b6b233d2e427dcd1102df6a916cf5 - AP:0.6956153758410263 - idx:6
        # ID:353ab00e964f71aa720385223a9078b770b7e3efaf5be0f66e670981f68fe606 - AP:0.7090760717590181 - idx:10
        # ID:175dbb364bfefc9537931144861c9b6e08934df3992782c669c6fe4234319dfc - AP:0.7349640039729471 - idx:9
        # ID:8f6e49e474ebb649a1e99662243d51a46cc9ba0c9c8f1efe2e2b662a81b48de1 - AP:0.7366945430619456 - idx:5
        # ID:6b61ab2e3ff0e2c7a55fd71e290b51e142555cf82bc7574fc27326735e8acbd1 - AP:0.7540038738142536 - idx:1

def test_processors_with_real_preds():

    # Instantiate default post-processor
    post_options = _DEFAULT_PROC_OPTIONS
    post = Post(options=post_options)
    post.print_config()

    # Run the post-processor
    post.process()

# if __name__ == '__main__':
#     # test_processors_with_fake_preds()
#     # test_processors_with_real_preds()
