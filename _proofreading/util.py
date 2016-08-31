import csv
import matplotlib.pyplot as plt
import numpy as np
import _tifffile as tif
import urllib
from sklearn import metrics
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

    # the coordinates of our subvolume in AC4
    x = 210
    y = 60
    #z = 50 # we dont need since we have already cut the z
    dim_x = dim_y = 400

    for what in ['image', 'groundtruth', 'segmentation']:
      current_url = base_url.format(what, z)
      tmpfile = urllib.urlretrieve(current_url)[0]
      data[what] = tif.imread(tmpfile)

      data[what] = data[what][y:y+dim_y,x:x+dim_x]

    return data['image'], data['groundtruth'], data['segmentation']

  @staticmethod
  def load_all():
    '''
    Loads all slices and returns three volumes containing images,
    groundtruths and the initial segmentations.
    '''
    print 'Loading..'

    images = []
    groundtruths = []
    segmentations = []

    for z in range(10):
      image, groundtruth, segmentation = Util.load(z)

      images.append(image)
      groundtruths.append(groundtruth)
      segmentations.append(segmentation)
        
      time.sleep(1)
      sys.stdout.write("\r%d%%" % ((z+1)*10))
      sys.stdout.flush()        

    images = np.stack(images, axis=0)
    groundtruths = np.stack(groundtruths, axis=0)
    segmentations = np.stack(segmentations, axis=0)

    return images, groundtruths, segmentations

  @staticmethod
  def view(images, golds, segmentations):
    '''
    '''
    z_count = 1
    if images.ndim == 3:
      z_count = images.shape[0]

    for z in range(z_count):
      fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

      text = 'Slice '+str(z)
      fig.text(0,0.5,text)

      image = images
      gold = golds
      segmentation = segmentations
      if z_count > 1:
        image = images[z]
        gold = golds[z]
        segmentation = segmentations[z]

      ax1.axis('off')
      ax1.imshow(image, cmap='gray')

      ax2.axis('off')
      ax2.imshow(gold)

      ax3.axis('off')
      ax3.imshow(segmentation)
    
  #
  # dojo user study
  # Haehn 2014, Design and Evaluation of Interactive Proofreading Tools
  #   for Connectomics, IEEE Vis
  #
  @staticmethod
  def load_users():
    '''
    '''
    user_info_url = "https://cdn.rawgit.com/haehn/proofreading/master/data/dojo_user_study.csv?raw=true"
    tmpfile = urllib.urlretrieve(user_info_url)[0]

    users = []

    with open(tmpfile, 'r') as f:
      csvreader = csv.reader(f)
      for row in csvreader:
        user = {}
        user['id'] = row[0]
        user['sex'] = row[1]
        user['age'] = row[2]
        user['occupation'] = row[3]
        # column 4 is empty
        user['tool'] = row[5]
        users.append(user)

    return users

  @staticmethod
  def load_user_results(id):
    '''
    '''
    base_url = "https://cdn.rawgit.com/haehn/proofreading/master/data/participants/{0}/{1}.tif?raw=true"

    data = []

    for z in range(10):
      current_url = base_url.format(id, z)
      tmpfile = urllib.urlretrieve(current_url)[0]
      data.append(tif.imread(tmpfile))

    data = np.stack(data, axis=0)

    return data

  #
  # measures
  #
  @staticmethod
  def variation_of_information(array1, array2, two_d=False):
    '''
    Meila 2003, Comparing Clusterings by the Variation of Information
    '''
    return partition_comparison.variation_of_information(array1.astype(np.uint64).ravel(), 
                                                         array2.astype(np.uint64).ravel())

  @staticmethod
  def rand_index(array1, array2, two_d=False):
    '''
    Rand 1971, Objective Criteria for the Evaluation of Clustering Methods
    '''
    return metrics.adjusted_rand_score(array1.astype(np.uint64).ravel(), 
                                       array2.astype(np.uint64).ravel())

  @staticmethod
  def edit_distance(gold, segmentation, two_d=False):
    '''
    '''
    min_2d_seg_size = 500
    min_3d_seg_size = 2000


    gt_ids = np.unique(gold.ravel())
    seg_ids = np.unique(segmentation.ravel())

    # count 2d split operations required
    split_count_2d = 0
    for seg_id in seg_ids:
        if seg_id == 0:
            continue
        for zi in range(segmentation.shape[0]):
            gt_counts = np.bincount(gold[zi,:,:][segmentation[zi,:,:]==seg_id])
            if len(gt_counts) == 0:
                continue
            gt_counts[0] = 0
            gt_counts[gt_counts < min_2d_seg_size] = 0
            gt_objects = len(np.nonzero(gt_counts)[0])
            if gt_objects > 1:
                split_count_2d += gt_objects - 1

    # count 3d split operations required
    split_count_3d = 0
    for seg_id in seg_ids:
        if seg_id == 0:
            continue
        gt_counts = np.bincount(gold[segmentation==seg_id])
        if len(gt_counts) == 0:
            continue
        gt_counts[0] = 0
        gt_counts[gt_counts < min_3d_seg_size] = 0
        gt_objects = len(np.nonzero(gt_counts)[0])
        if gt_objects > 1:
            split_count_3d += gt_objects - 1

    # count 3d merge operations required
    merge_count = 0
    for gt_id in gt_ids:
        if gt_id == 0:
            continue
        seg_counts = np.bincount(segmentation[gold==gt_id])
        if len(seg_counts) == 0:
            continue
        seg_counts[0] = 0
        seg_counts[seg_counts < min_3d_seg_size] = 0
        seg_objects = len(np.nonzero(seg_counts)[0])
        if seg_objects > 1:
            merge_count += seg_objects - 1

    return (split_count_2d, split_count_3d, merge_count)

