import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import sparse
from sklearn import metrics
import sys
import tempfile
import _tifffile as tif
import time
import urllib

import partition_comparison

class Util(object):
  '''Utilities for the proofreading tests.'''

  @staticmethod
  def load(z):
    '''
    Loads a given slice (z) and returns the image, groundtruth and 
    the initial segmentation.
    '''

    # add caching of the loaded files
    if tempfile.tempdir:
      # caching only works if we have a valid temp. directory
      tempfolder = os.path.join(tempfile.tempdir, 'dojo_proofreading_study')
      if not os.path.exists(tempfolder):
        os.mkdir(tempfolder)
    else:
      # caching will only work for the current instance
      tempfolder = tempfile.mkdtemp()


    base_url = "https://cdn.rawgit.com/haehn/proofreading/master/data/{0}/{1}.tif?raw=true"

    data = {}

    # the coordinates of our subvolume in AC4
    x = 210
    y = 60
    #z = 50 # we dont need since we have already cut the z
    dim_x = dim_y = 400

    for what in ['image', 'groundtruth', 'segmentation']:
      current_url = base_url.format(what, z)

      #
      current_tempfolder = os.path.join(tempfolder, what)
      if not os.path.exists(current_tempfolder):
        os.mkdir(current_tempfolder)
      current_tempfilename = os.path.join(current_tempfolder, str(z)+'.tif')

      if not os.path.exists(current_tempfilename):
        # we need to download
        tmpfile = urllib.urlretrieve(current_url, filename=current_tempfilename)[0]
      else:
        tmpfile = current_tempfilename

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
    user_info_url = "https://raw.githubusercontent.com/haehn/proofreading/31751cfbebbd01cfd8a5691b2731cae1c8dd3869/data/dojo_user_study.csv"
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
  def evaluate_users(users, golds):
    '''
    '''
    print 'Loading..'

    for i,u in enumerate(users):
        data = Util.load_user_results(u['id'])
        
        # grab measures
        vi = Util.variation_of_information(golds, data)
        ri = Util.rand_index(golds, data)
        ed = Util.edit_distance(golds, data)
        ed = ed[0] + ed[2] # we use number of 2d splits + 3d merges as described in the paper
        adapted_rand_error = Util.adapted_rand_error(golds, data)
        adapted_vi_error = Util.adapted_vi_error(golds, data)
        
        u['vi'] = vi
        u['ri'] = ri
        u['ed'] = ed
        u['rand_error'] = adapted_rand_error
        u['vi_error'] = adapted_vi_error
        
        time.sleep(1)
        sys.stdout.write("\r%d%%" % (((i+1)/float(len(users)))*100))
        sys.stdout.flush()

  @staticmethod
  def load_user_results(id):
    '''
    '''
    base_url = "https://cdn.rawgit.com/haehn/proofreading/master/data/participants/{0}/{1}.tif?raw=true"

    # add caching of the loaded files
    if tempfile.tempdir:
      # caching only works if we have a valid temp. directory
      tempfolder = os.path.join(tempfile.tempdir, 'dojo_proofreading_study')
      if not os.path.exists(tempfolder):
        os.mkdir(tempfolder)
    else:
      # caching will only work for the current instance
      tempfolder = tempfile.mkdtemp()

    data = []

    for z in range(10):
      current_url = base_url.format(id, z)

      #
      current_tempfolder = os.path.join(tempfolder, id)
      if not os.path.exists(current_tempfolder):
        os.mkdir(current_tempfolder)
      current_tempfilename = os.path.join(current_tempfolder, str(z)+'.tif')

      if not os.path.exists(current_tempfilename):
        # we need to download
        tmpfile = urllib.urlretrieve(current_url, filename=current_tempfilename)[0]
      else:
        tmpfile = current_tempfilename

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
    By Seymour Knowles-Barley (Google)
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

  # Evaluation code courtesy of Juan Nunez-Iglesias, taken from
  # https://github.com/janelia-flyem/gala/blob/master/gala/evaluate.py

  @staticmethod
  def adapted_rand_error(gt, seg, all_stats=False):
      """Compute Adapted Rand error as defined by the SNEMI3D contest [1]
      Formula is given as 1 - the maximal F-score of the Rand index
      (excluding the zero component of the original labels). Adapted
      from the SNEMI3D MATLAB script, hence the strange style.
      Parameters
      ----------
      seg : np.ndarray
          the segmentation to score, where each value is the label at that point
      gt : np.ndarray, same shape as seg
          the groundtruth to score against, where each value is a label
      all_stats : boolean, optional
          whether to also return precision and recall as a 3-tuple with rand_error
      Returns
      -------
      are : float
          The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
          where $p$ and $r$ are the precision and recall described below.
      prec : float, optional
          The adapted Rand precision. (Only returned when `all_stats` is ``True``.)
      rec : float, optional
          The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)
      References
      ----------
      [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
      """
      # segA is truth, segB is query
      segA = np.ravel(gt)
      segB = np.ravel(seg)

      # mask to foreground in A
      mask = (segA > 0)
      segA = segA[mask]
      segB = segB[mask]
      n = segA.size  # number of nonzero pixels in original segA

      n_labels_A = np.amax(segA) + 1
      n_labels_B = np.amax(segB) + 1

      ones_data = np.ones(n)

      p_ij = sparse.csr_matrix((ones_data, (segA.ravel(), segB.ravel())),
                               shape=(n_labels_A, n_labels_B),
                               dtype=np.uint64)

      # In the paper where adapted rand is proposed, they treat each background
      # pixel in segB as a different value (i.e., unique label for each pixel).
      # To do this, we sum them differently than others

      B_nonzero = p_ij[:, 1:]
      B_zero = p_ij[:, 0]

      # this is a count
      num_B_zero = B_zero.sum()

      # This is the old code, with conversion to probabilities:
      #
      #  # sum of the joint distribution
      #  #   separate sum of B>0 and B=0 parts
      #  sum_p_ij = ((B_nonzero.astype(np.float32) / n).power(2).sum() +
      #              (float(num_B_zero) / (n ** 2)))
      #  
      #  # these are marginal probabilities
      #  a_i = p_ij.sum(1).astype(np.float32) / n
      #  b_i = B_nonzero.sum(0).astype(np.float32) / n
      #  
      #  sum_a = np.power(a_i, 2).sum()
      #  sum_b = np.power(b_i, 2).sum() + (float(num_B_zero) / (n ** 2))

      # This is the new code, removing the divides by n because they cancel.
      
      # sum of the joint distribution
      #   separate sum of B>0 and B=0 parts
      sum_p_ij = (B_nonzero).power(2).sum() + num_B_zero
                  
      
      # these are marginal probabilities
      a_i = p_ij.sum(1)
      b_i = B_nonzero.sum(0)
      
      sum_a = np.power(a_i, 2).sum()
      sum_b = np.power(b_i, 2).sum() + num_B_zero

      precision = float(sum_p_ij) / sum_b
      recall = float(sum_p_ij) / sum_a

      fScore = 2.0 * precision * recall / (precision + recall)
      are = 1.0 - fScore

      if all_stats:
          return (are, precision, recall)
      else:
          return are

  @staticmethod
  def adapted_vi_error(gt, seg, all_stats=False):
      """Compute Adapted VI error as defined by the SNEMI3D contest [1]
      Formula is given as 1 - the maximal F-score of the Rand index
      (excluding the zero component of the original labels). Adapted
      from the SNEMI3D MATLAB script, hence the strange style.
      Parameters
      ----------
      seg : np.ndarray
          the segmentation to score, where each value is the label at that point
      gt : np.ndarray, same shape as seg
          the groundtruth to score against, where each value is a label
      all_stats : boolean, optional
          whether to also return precision and recall as a 3-tuple with rand_error
      Returns
      -------
      are : float
          The adapted VI error;
      prec : float, optional
          The adapted VI precision. (Only returned when `all_stats` is ``True``.)
      rec : float, optional
          The adapted VI recall.  (Only returned when `all_stats` is ``True``.)
      References
      ----------
      [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
      """
      # segA is truth, segB is query
      segA = np.ravel(gt)
      segB = np.ravel(seg)

      # mask to foreground in A
      mask = (segA > 0)
      segA = segA[mask]
      segB = segB[mask]
      n = segA.size  # number of nonzero pixels in original segA

      n_labels_A = np.amax(segA) + 1
      n_labels_B = np.amax(segB) + 1

      ones_data = np.ones(n)
      # print n, n_labels_A, n_labels_B
      # print segB.ravel().shape
      # return
      # p_ij = np.array((n_labels_A, n_labels_B), dtype=np.uint64)
      # p_ij[:]
      # print ones_data
      p_ij = sparse.csr_matrix((ones_data, (segA.ravel(), segB.ravel())),
                               shape=(n_labels_A, n_labels_B),
                               dtype=np.uint64).astype(np.float64)
      # print p_ij.shape
      # print p_ij
      # p_ij = p_ij.todense()
      # return
      # In the paper where adapted rand is proposed, they treat each background
      # pixel in segB as a different value (i.e., unique label for each pixel).
      # To do this, we sum them differently than others

      B_nonzero = p_ij[:, 1:]
      B_zero = p_ij[:, 0]



      # this is a count
      num_B_zero = float(B_zero.sum())

      # sum of the joint distribution
      #   separate sum of B>0 and B=0 parts
      eps = 1e-15
      # added the .data to make it work with my scipy
      plogp_ij = (B_nonzero.data / n) * (np.log(B_nonzero.data + eps) - np.log(n))
      sum_plogp_ij = plogp_ij.sum() - (num_B_zero / n) * np.log(n)

      # these are marginal probabilities
      a_i = p_ij.sum(1)
      b_i = B_nonzero.sum(0)

      #
      # I added this which is probably pretty inefficient but ok for the small data
      #
      a_i = np.array(np.squeeze(a_i).tolist()[0])
      b_i = np.array(np.squeeze(b_i).tolist()[0])

      sum_aloga_i = ((a_i / n) * (np.log(a_i + eps) - np.log(n))).sum()
      #   separate sum of B>0 and B=0 parts
      sum_blogb_i = ((b_i / n) * (np.log(b_i + eps) - np.log(n))).sum() - (num_B_zero / n) * np.log(n)

      precision = (sum_plogp_ij - sum_aloga_i - sum_blogb_i) / sum_blogb_i
      recall = (sum_plogp_ij - sum_aloga_i - sum_blogb_i) / sum_aloga_i

      fScore = 2.0 * precision * recall / (precision + recall)
      are = 1.0 - fScore

      if all_stats:
          return (are, precision, recall)
      else:
          return are

  #
  # statistics
  #
  @staticmethod
  def group_users_by_tool(users):

    tools = []
    for u in users:
      tools.append(u['tool'])
    tools = sorted(list(set(tools)))

    vi_tools = {}
    ri_tools = {}
    ed_tools = {}
    rand_error_tools = {}
    vi_error_tools = {} 

    for t in tools:
      vi_tools[t] = []
      ri_tools[t] = []
      ed_tools[t] = []
      rand_error_tools[t] = []
      vi_error_tools[t] = []

    for u in users:
        
        vi_tools[u['tool']].append(u['vi'])
        ri_tools[u['tool']].append(u['ri'])
        ed_tools[u['tool']].append(u['ed'])
        rand_error_tools[u['tool']].append(u['rand_error'])
        vi_error_tools[u['tool']].append(u['vi_error'])    

    measures = {}
    measures['vi'] = vi_tools
    measures['ri'] = ri_tools
    measures['ed'] = ed_tools
    measures['rand_error'] = rand_error_tools
    measures['vi_error'] = vi_error_tools

    return measures

  @staticmethod
  def plot_metric(title, baseline, measured_data):

    data = {}
    data[title] = {'Baseline': baseline}

    for d in measured_data:
      data[title][d] = measured_data[d]

    # create the plot
    Util.create_tool_plot(data)


  #
  # visualization
  #
  @staticmethod
  def create_tool_plot(input_data, outputfile=''):
    '''
    Plots a measure across tools with a baseline.

    Input data:
      data = {}
      data['TITLE'] = {'Baseline':BASELINE, 'Dojo':TOOL1, 'Mojo':TOOL2, 'Raveler':TOOL3..}
    '''
    title = input_data.keys()[0]
    baseline = -1
    data = []
    data_labels =[]

    for k in input_data[title].keys():

      if k == 'Baseline':
        baseline = input_data[title][k]
      else:
        data_labels.append(k)
        data.append(input_data[title][k])

    # pair data and labels
    data_pairs = zip(data_labels, data)
    # .. and sort
    data_pairs = sorted(data_pairs, key= lambda t: t[0])

    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(range(3*len(data_pairs)), [baseline]*3*len(data_pairs), 'k:', color='gray')
    b = plt.boxplot([d[1] for d in data_pairs])
    # print range(len(data_pairs)), [d[0] for d in data_pairs]
    b = plt.xticks(range(1,len(data_pairs)+1), [d[0] for d in data_pairs])
    b = plt.ylabel(title, labelpad=20)
