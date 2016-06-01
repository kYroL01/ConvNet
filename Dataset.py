#!/usr/bin/python
# -*- coding: utf-8 -*-

# ======================= DATASET ====================================
#
# Categories:
# 1) Cani
# 2) Gatti
# 3) Cavalli
# 4) Alberi
#
# ================================================================================

import glob
import hashlib
import os.path
import random
import re
import sys
import imghdr as ih

import numpy as np
from six.moves import urllib
import tensorflow as tf
import utils


class Dataset(object):

    # The object dataset must be initialize with a valid directory
    def __init__(self, image_dir):
        
        self.image_dir = image_dir
        
    
    """ Builds a dictionary for the dataset of images.
    - Analyzes the sub folders in the image directory;
    - Splits them into training, testing, and validation sets.
    Args:
          test_percentage:       Integer - percentage of the images for tests.
          validation_percentage: Integer - percentage of images for validation.
    Returns:
          A dictionary containing the lists of images for each label and their paths."""
    def create_dataset(self, test_percentage, validation_percentage):

        ret_dict = {}      # the returned dict
        file_list = []     # list of images for every categories

        sub_dirs = [x[0] for x in os.walk(self.image_dir)] # subdirs for layers

        # Skip root directory and analyze other directories
        is_root_dir = True
        for sub_dir in sub_dirs:
            if is_root_dir:
                is_root_dir = False
                continue

            dir_name = os.path.basename(sub_dir)
            extensions = ['jpg', 'jpeg']

            if dir_name == self.image_dir: # skip root dir
                continue
            print("Looking for images in '" + dir_name + "'")
            for ext in extensions:
                file_glob = os.path.join(self.image_dir, dir_name, '*.' + ext)
                file_list.extend(glob.glob(file_glob)) # add img to file_list

            training_images = []
            testing_images = []
            validation_images = []

            # create lists for every categories
            # and split images to training, testing, validation
            tot_images = 0
            for file_name in file_list:
                # count number of images
                tot_images += 1
                base_name = os.path.basename(file_name)
                hash_name = re.sub(r'_nohash_.*$', '', file_name)
                hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
                percentage_hash = int(hash_name_hashed, 16) % 65536 * (100 / 65535.0)
                if percentage_hash < validation_percentage:
                    validation_images.append(base_name)
                elif percentage_hash < test_percentage \
                    + validation_percentage:
                    testing_images.append(base_name)
                else:
                    training_images.append(base_name)

            ret_dict[dir_name] = {
                'dir': dir_name,
                'training': training_images,
                'testing': testing_images,
                'validation': validation_images,
                }
            
        return ret_dict

    
    """
    Count total number of images
    """
    def get_Num_images(self):
        count = 0
        for dirName, subdirList, fileList in os.walk(self.image_dir):
            for img in fileList:
                count += 1
        return count

    """
    Convert labels from string to nparray of int
    """
    def conv_labels(self, imageDir):

        labels_dict = {
            'Cani': 0,
            'Cavalli': 1,
            'Alberi': 2,
            'Gatti': 3,
        }

        list_labels = labels_dict.values()
        index_offset = np.arange(1) * len(labels_dict)
        labels_one_hot = np.zeros((1, len(labels_dict)))
        labels_one_hot.flat[index_offset + np.array([labels_dict[imageDir]])] = 1
        print("LABELS_ONE_HOT = ", labels_one_hot)

        return labels_one_hot

    
    """
    Convert all the images in the folders in numpy ndarray 
    """
    def convert_to_array():
        
        for dirName, subdirList, fileList in os.walk(imageDir):
            # print('Directory: %s' % dirName)
            # for img in fileList:
            for image, padded, cropped in pad_and_crop_image_dimensions(200, 200, imageDir):
                yield padded, self.conv_labels(imageDir)
                







                # resized = utils.resize_image_with_crop_or_pad(imageDir, 300, 300)



                # filepath = os.path.join(dirName, img)
                # # print("Image = ", filepath)
                # img_arr = load_image(filepath)

                
