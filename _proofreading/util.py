import numpy as np
import _tifffile as tif
import urllib
import sys
import time

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
    print 'Loading 0%'

    images = []
    groundtruths = []
    segmentations = []

    for z in range(10):
      image, groundtruth, segmentation = Util.load(z)

      images.append(image)
      groundtruths.append(groundtruth)
      segmentations.append(segmentation)
        
      time.sleep(1)
      sys.stdout.write("Loading \r%d%%" % (z+1)*10)
      sys.stdout.flush()        

    images = np.stack(images, axis=0)
    groundtruths = np.stack(groundtruths, axis=0)
    segmentations = np.stack(segmentations, axis=0)

    return images, groundtruths, segmentations

  @staticmethod
  def variation_of_information(array1, array2, two_d=False):
    '''
    '''
    return partition_comparison.variation_of_information(array1.astype(np.uint64).ravel(), 
                                                         array2.astype(np.uint64).ravel())
