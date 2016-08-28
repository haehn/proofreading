import numpy as np
import _tifffile as tif
import urllib
import sys

import partition_comparison

class Util(object):
  '''Utilities for the proofreading tests.'''

  @staticmethod
  def load(z):
    '''
    Loads a given slice (z) and returns the image, groundtruth and 
    the initial segmentation.
    '''

    base_url = "https://cdn.rawgit.com/haehn/proofreading/master/data/{0}/{1}.tif?raw=true"

    data = {}

    for what in ['image', 'groundtruth', 'segmentation']:
      current_url = base_url.format(what, z)
      tmpfile = urllib.urlretrieve(current_url)[0]
      data[what] = tif.imread(tmpfile)

    return data['image'], data['groundtruth'], data['segmentation']



  @staticmethod
  def load_all():
    '''
    Loads all slices and returns three volumes containing images,
    groundtruths and the initial segmentations.
    '''
    images = None
    groundtruths = None
    segmentations = None


    for z in range(10):
      Util.load(z)

    return images, groundtruths, segmentations

  @staticmethod
  def variation_of_information(array1, array2, two_d=False):
    '''
    '''
    return partition_comparison.variation_of_information(array1.ravel(), array2.ravel())
