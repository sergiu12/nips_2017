"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from scipy.misc import imread
from scipy.misc import imsave
import scipy.ndimage

import tensorflow as tf


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def make_mask(img):
    diff_img = np.zeros_like(img)
    for dim in range(3):
        b_img = scipy.ndimage.gaussian_filter(img[:, :, dim], 3)
        diff_img_1d = img[:, :, dim] - b_img

        diff_img_1d = scipy.ndimage.gaussian_filter(diff_img_1d, 3)

        thresh = 0
        #print(diff_img_1d.min(), diff_img_1d.max())
        diff_img_1d[diff_img_1d > thresh] = 255
        diff_img_1d[diff_img_1d <= thresh] = 0
        diff_img[:, :, dim] = diff_img_1d
    return diff_img


def highpassmodify(img, eps=16):
    max_pert = eps
    #print("Max change: %d" % int(max_pert))
    # img_out = np.zeros_like(img)
    img_f = make_mask(img)
    #print(img_f.max(), img_f.min())
    thresh = img_f.mean()
    img_f[img_f <= thresh] = -1
    img_f[img_f > thresh] = 1
    img_f = img_f * max_pert * (-1)
    img_out = np.clip(img + img_f, 0, 255)
    #print((img_out-img).sum())
    #img_out[:50, :50] = 0
    return img_out


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Length of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    #print(filepath)
    with tf.gfile.Open(filepath) as f:
      #images[idx, :, :, :] = imread(f, mode='RGB').astype(np.float) / 255.0
      images[idx, :, :, :] = imread(f, mode='RGB').astype(np.float)
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir, eps=16):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      images[i, :, :, :] = highpassmodify(images[i, :, :, :], eps=eps)
      imsave(f, images[i, :, :, :] / 255.0, format='png')


def main(_):
  eps = FLAGS.max_epsilon
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  for filenames, images in load_images(FLAGS.input_dir, batch_shape):
    save_images(images, filenames, FLAGS.output_dir,eps=eps)
  print(FLAGS.output_dir)

if __name__ == '__main__':
  tf.app.run()
