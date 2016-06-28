#!/usr/bin/python
# -*- coding: utf-8 -*-

# ======================= DATASET ====================================
#
# Categories:
# 0) Cani
# 1) Cavalli
# 2) Alberi
# 3) Gatti
#
# ================================================================================
import glob
import os.path
import sys
import numpy as np
import tensorflow as tf


IMG_SIZE = 30

LABELS_DICT = {
    'Cani': 0,
    'Cavalli': 1,
    'Alberi': 2,
    'Gatti': 3,
}


"""
Create class Dataset
"""
class Dataset(object):

    # The object dataset must be initialize with a valid directory
    def __init__(self, image_dir):
        
        self.image_dir = image_dir

    """
    Return the dataset as images and labels
    """
    def getDataset(self):
        with tf.Session() as session:
            tf.initialize_all_variables().run()

            for dirName in os.listdir(self.image_dir):
                label = self.convLabels(dirName)
                path = os.path.join(self.image_dir, dirName)
                for img in os.listdir(path):
                    img_path = os.path.join(path, img)
                    if os.path.isfile(img_path) and (img.endswith('jpeg') or
                                                     (img.endswith('jpg'))):
                        img_bytes = tf.read_file(img_path)
                        #img_u8 = tf.image.decode_jpeg(img_bytes, channels=3)
                        img_u8 = tf.image.decode_jpeg(img_bytes, channels=1)
                        img_u8_eval = session.run(img_u8)
                        image = tf.image.convert_image_dtype(img_u8_eval, tf.float32)
                        img_padded_or_cropped = tf.image.resize_image_with_crop_or_pad(image, IMG_SIZE, IMG_SIZE)

                        #img_padded_or_cropped = tf.reshape(img_padded_or_cropped, shape=[IMG_SIZE*IMG_SIZE, 3])
                        img_padded_or_cropped = tf.reshape(img_padded_or_cropped, shape=[IMG_SIZE * IMG_SIZE])

                        yield img_padded_or_cropped.eval(), np.array(label)
    
    
    """
    Count total number of images
    """
    def getNumImages(self):
        count = 0
        for dirName, subdirList, fileList in os.walk(self.image_dir):
            for img in fileList:
                count += 1
        return count

    """
    Convert labels from string to nparray of int
    """
    def convLabels(self, imageDir):
        index_offset = np.arange(1) * len(LABELS_DICT)
        labels_one_hot = np.zeros((1, len(LABELS_DICT)))
        labels_one_hot.flat[index_offset + np.array([LABELS_DICT[imageDir]])] = 1
        print("LABELS_ONE_HOT = ", labels_one_hot)

        return labels_one_hot[0]

    
    """
    Convert all the images in the folders in numpy ndarray 
    """
    def convertToArray(self):
        
        for dirName, subdirList, fileList in os.walk(imageDir):
            # print('Directory: %s' % dirName)
            # for img in fileList:
            for image, padded, cropped in pad_and_crop_image_dimensions(200, 200, imageDir):
                yield padded, self.conv_labels(imageDir)
