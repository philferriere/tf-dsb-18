"""
logger.py

Tensor ops-free logger to Tensorboard.

Modifications by Phil Ferriere licensed under the MIT License (see LICENSE for details)

Based on:
  - https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
    Written by Michael Gygli
    License: Copyleft
"""

import tensorflow as tf
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

class TBLogger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir, graph=None):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir, graph=graph)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Args:
            tag: name of the scalar
            value: scalar value to log
            step: training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_images(self, tag, images, step, IDs=None):
        """Logs a list of images.
        Args:
            tag: format for the name of the image summary (will format ID accordingly)
            plots: list of plots
            IDs: list of image IDs
            step: training iteration
        """
        im_summaries = []
        for n in range(len(images)):
            # Write the image to a string
            faux_file = BytesIO() # StringIO()
            if len(images[n].shape) == 3 and images[n].shape[2] == 1:
                image = np.squeeze(images[n], axis=-1)  # (H, W, 1) -> (H, W)
                cmap = 'gray'
            else:
                image = images[n]
                cmap = None
            plt.imsave(faux_file, image, cmap=cmap, format='png') # (?, H, W, ?)
            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=faux_file.getvalue(), height=image.shape[0],
                                       width=image.shape[1])
            # Create a Summary value
            img_tag = tag.format(IDs[n]) if IDs is not None else tag.format(n)
            im_summaries.append(tf.Summary.Value(tag=img_tag, image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()