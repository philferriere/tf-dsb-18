"""
sampling.py

Helpers to identify image clusters, perform stratified sampling, and oversample the minority classes.

Modifications by Phil Ferriere licensed under the MIT License (see LICENSE for details)

Based on:
    -https://www.kaggle.com/mpware/stage1-eda-microscope-image-types-clustering/notebook
        Written by MPWARE Team
        Licensed under the MIT License

Stats:
        All IDs:
           cluster   id  percentage
        0        0  546   81.492537
        1        1  108   16.119403
        2        2   16    2.388060

    With 'normal' sampling of stratified splits (strats according to HSV clusters):

        Val IDs:
           cluster   id  percentage
        0        0  109   81.343284
        1        1   22   16.417910
        2        2    3    2.238806

        Train IDs:
           cluster   id  percentage
        0        0  437   81.529851
        1        1   86   16.044776
        2        2   13    2.425373

    With 'oversampling' of training split (strats according to HSV clusters):

        Val IDs:
           cluster   id  percentage
        0        0  109   81.343284
        1        1   22   16.417910
        2        2    3    2.238806

        Train IDs:
           cluster   id  percentage
        0        0  437   33.333333
        1        1  437   33.333333
        2        2  437   33.333333

Refs:
    The Right Way to Oversample in Predictive Modeling
    @ https://beckernick.github.io/oversampling-modeling/

    sklearn.model_selection.train_test_split
    @ http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

    skimage.color.rgb2hsv
    RGB to HSV color space conversion
    @ http://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.rgb2hsv

    skimage.color.rgb2hed
    RGB to Haematoxylin-Eosin-DAB (HED) color space conversion
    @ http://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.rgb2hed
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from skimage.io import imread, imsave
from skimage.color import rgb2hsv, rgb2hed
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from visualize import archive_images

if sys.platform.startswith("win"):
    _DSB18_DATASET = "E:/datasets/dsb18"
else:
    _DSB18_DATASET = '/media/EDrive/datasets/dsb18'

_DEFAULT_SAMPLING_OPTIONS = { # See constructor for explanation of options
    'sampling': 'oversampling', # ['normal' | 'oversampling']
    'train_val_split': 'stratified', # ['random' | 'stratified']
    'clustering': 'hsv_clusters' , # None or ['hsv_clusters' | 'hed_clusters' | rgb_clusters]
    'num_clusters': 3,
    'num_colors': 1,
    'random_seed' : 1969,
    'val_split' : 0.2
    }

class Sampler(object):
    """Post-processor.
    """

    def __init__(self, ds_root=_DSB18_DATASET, options=_DEFAULT_SAMPLING_OPTIONS):
        """Initialize the Sampler object
        Args:
            options: see below
        Options:
            sampling: Sampling technique ['random_split' | 'stratified_split']
            clustering: Cluster in HSV or HED space
            num_clusters: Number of clusters to categorize images in
        """
        # Only options supported in this initial implementation
        assert(options['sampling'] in ['normal', 'oversampling'])
        assert(options['train_val_split'] in ['random', 'stratified'])
        assert(options['clustering'] in ['hsv_clusters', 'hed_clusters', 'rgb_clusters'])

        # Set paths and file names
        self._ds_root = ds_root

        self._train_folder = self._ds_root + '/stage1_train'
        self._val_folder = self._ds_root + '/stage1_train'

        if options['clustering'] is None or options['train_val_split'] == 'random':
            file = 'random_split'
        else:
            file = '{}_{}{}_{}colors'.format(options['train_val_split'], options['num_clusters'], options['clustering'], options['num_colors'])
        self._IDs_csv = self._ds_root + '/' + file + '.csv'

        # Read in image IDs
        self._IDs = self.image_ids_in(self._train_folder)
        self._train_IDs = self._val_IDs = None

        # Save options
        self.options = options.copy()
        self._df = None

        # Categorize on instantiation and create splits
        if options['train_val_split'] == 'stratified':
            self.categorize()
        self.create_splits()

    @property
    def categories(self):
        if self._df is not None:
            return self._df['cluster'].unique()
        else:
            return []

    @property
    def train_IDs(self):
        if self._train_IDs is None:
            self.create_splits()
        return self._train_IDs

    @property
    def val_IDs(self):
        if self._val_IDs is None:
            self.create_splits()
        return self._val_IDs

    def create_splits(self):
        """Create the ID lists for each split of the dataset
        """
        if self.options['val_split'] > 0.:
            if self.options['train_val_split'] == 'random':
                self._train_IDs, self._val_IDs = train_test_split(self._IDs, test_size=self.options['val_split'],
                                                                  random_state=self.options['random_seed'])
            elif self.options['train_val_split'] == 'stratified':
                strats = self._df['cluster'].values.tolist()
                self._train_IDs, self._val_IDs = train_test_split(self._IDs, test_size=self.options['val_split'],
                                                                  random_state=self.options['random_seed'],
                                                                  stratify=strats)
                if self.options['sampling'] == 'oversampling':
                    df = self._df[self._df['id'].isin(self._train_IDs)]
                    props_all = df.groupby('cluster')['id'].count().reset_index()
                    maj_class, maj_class_count = props_all.iloc[0, 0], props_all.iloc[0, 1]
                    self._train_IDs = df[df['cluster'] == maj_class]['id'].tolist()
                    min_classes = props_all.iloc[1:, 0].tolist()
                    min_classes_counts = props_all.iloc[1:, 1].tolist()
                    for min_class, min_class_count in zip(min_classes, min_classes_counts):
                        quotient, remainder = divmod(maj_class_count, min_class_count)
                        min_class_IDs = df[df['cluster'] == min_class]['id'].tolist()
                        self._train_IDs.extend(np.random.choice(min_class_IDs, remainder, replace=False).tolist())
                        self._train_IDs.extend(min_class_IDs * quotient)
        else:
            self._train_IDs, self._val_IDs = self._IDs, None

        np.random.shuffle(self._train_IDs)
        if self._val_IDs is not None:
            np.random.shuffle(self._val_IDs)

    def categorize(self):
        """Categorize image samples according to chosen criteria.
        """
        # Load CSV with categorized data, if available
        # If not available, categorize data from scratch
        if self.from_csv() is False:
            # Get image details
            details = self.get_images_details(self._IDs)

            # Define dataframe columns
            columns = ['id', 'width', 'height', 'criteria']
            self._df = pd.DataFrame(details, columns=columns)

            # Build clusters
            X = (pd.DataFrame(self._df['criteria'].values.tolist())).as_matrix()
            kmeans = KMeans(n_clusters=self.options['num_clusters']).fit(X)
            clusters = kmeans.predict(X)
            self._df['cluster'] = clusters

            # Save results
            self.to_csv()

    #
    # Image file helpers
    #
    def image_ids_in(self, root_dir, ignore=[]):
        ids = []
        for id in os.listdir(root_dir):
            if id in ignore:
                print('Skipping ID:', id)
            else:
                ids.append(id)
        return ids

    def read_image(self, image_id, space="rgb"):
        # Read file in
        image_file = self._train_folder + '/{}/images/{}.png'.format(image_id, image_id)
        image = imread(image_file)

        # Drop alpha which is not used
        image = image[:, :, :3]

        # Color convert
        if space == "hsv_clusters":
            image = rgb2hsv(image)
        elif space == "hed_clusters":
            image = rgb2hed(image)

        return image

    ###
    ### HSV helpers
    ###
    def get_domimant_colors(self, img, top_colors=2):
        img_l = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
        clt = KMeans(n_clusters=top_colors)
        clt.fit(img_l)
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)
        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()
        return clt.cluster_centers_, hist

    def get_images_details(self, image_ids):
        details = []
        for image_id in tqdm(image_ids, total=len(image_ids), ascii=True, ncols=100, desc='Getting image details'):
            image = self.read_image(image_id, space=self.options['clustering'])
            height, width, channels = image.shape
            dominant_colors, dominant_rates = self.get_domimant_colors(image, top_colors=self.options['num_colors'])
            dominant_colors = dominant_colors.reshape(1, dominant_colors.shape[0] * dominant_colors.shape[1])
            info = (image_id, width, height, dominant_colors.squeeze())
            details.append(info)
        return details


    ###
    ### CSV helpers
    ###
    def to_csv(self):
        """Save clustering info to a CSV
        """
        if os.path.exists(self._IDs_csv):
            os.remove(self._IDs_csv)

        self._df.to_csv(self._IDs_csv, index=False)

    def from_csv(self):
        """Load clustering info from a CSV
        """
        if os.path.exists(self._IDs_csv):
            self._df = pd.read_csv(self._IDs_csv)
            return True
        return False


    ###
    ### Stats
    ###
    def print_proportions(self):
        if self.options['train_val_split'] == 'stratified':
            print('\nVal IDs:')
            df = self._df[self._df['id'].isin(self._val_IDs)]
            props_val = df.groupby('cluster')['id'].count().reset_index()
            props_val['percentage'] = 100 * props_val['id'] / props_val['id'].sum()
            print(props_val)

            print('\nTrain IDs:')
            if self.options['sampling'] == 'normal':
                df = self._df[self._df['id'].isin(self._train_IDs)]
            elif self.options['sampling'] == 'oversampling':
                clusters = []
                for train_ID in self._train_IDs:
                    clusters.append(self._df['cluster'][self._df['id'] == train_ID].values[0])
                df = pd.DataFrame({'id': self._train_IDs, 'cluster': clusters})
            props_train = df.groupby('cluster')['id'].count().reset_index()
            props_train['percentage'] = 100 * props_train['id'] / props_train['id'].sum()
            print(props_train)

        else:
            print('\nIDs: {}, train_IDs: {}, val_IDs: {}'.format(len(self._IDs), len(self._train_IDs), len(self._val_IDs)))


    ###
    ### Debug utils
    ###
    def print_config(self):
        """Display configuration values."""
        print("\nSampler Configuration:")
        for k, v in self.options.items():
            print("  {:20} {}".format(k, v))
        print("  {:20} {}".format('num IDs', len(self._IDs)))

    def plot_images_by_IDs(self, image_ids, images_rows=6, images_cols=8):
        f, axarr = plt.subplots(images_rows, images_cols, figsize=(16, images_rows * 2))
        for row in range(images_rows):
            for col in range(images_cols):
                image_id = image_ids[row * images_cols + col]
                image = self.read_image(image_id)
                height, width, l = image.shape
                ax = axarr[row, col]
                ax.axis('off')
                ax.set_title("%dx%d" % (width, height))
                ax.imshow(image)

    def plot_images_by_category(self, category):
        image_ids = self._df[self._df['cluster'] == category]['id'].values
        self.plot_images_by_IDs(image_ids)


    def archive_images_by_category(self, category, folder):
        image_ids = self._df[self._df['cluster'] == category]['id'].values
        images = []
        for image_id in image_ids:
            images.append(self.read_image(image_id))
        archive_images(images, folder)

# def test_basic_behavior():
#
#     # Instantiate sampler
#     smplr = Sampler()
#
#     # Display sampling configuration
#     smplr.print_config()
#
#     # # Prime samplerDisplay model configuration
#     # smplr.categorize()
#
#     # Stats
#     smplr.print_proportions()
#
#     # # Save images to disk
#     # categories = smplr.categories
#     # for category in tqdm(categories, total=len(categories), ascii=True, ncols=100, desc='Archiving images by category'):
#     #     smplr.archive_images_by_category(category, "c:/temp/category_{}".format(category))
#
# if __name__ == '__main__':
#     test_basic_behavior()
