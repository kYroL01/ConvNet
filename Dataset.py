#!/usr/bin/python
# -*- coding: utf-8 -*-

# ======================= DATASET ====================================
#
# Categories:
# 3) Cani
# 2) Cavalli
# 1) Alberi
# 0) Gatti
#
# ====================================================================
import glob
import os.path
import sys
import numpy as np
import tensorflow as tf
import logging as log
from timeit import default_timer as timer
import cPickle as pickle
import gzip


## GLOBAL VARIABLES ##
global IMG_SIZE
IMG_SIZE = 224

# TODO: dynamic is better
global LABELS_DICT
LABELS_DICT = {
    'Cani': 3,
    'Cavalli': 2,
    'Alberi': 1,
    'Gatti': 0,
}


"""
Count total number of images
"""
def getNumImages(image_dir):
    count = 0
    for dirName, subdirList, fileList in os.walk(image_dir):
        for img in fileList:
            count += 1
    return count


"""
Return the dataset as images and labels
"""
def convertDataset(image_dir):

    num_labels = len(LABELS_DICT)
    label = np.eye(num_labels)  # Convert labels to one-hot-vector
    i = 0
    
    session = tf.Session()
    init = tf.initialize_all_variables()
    session.run(init)

    log.info("Start processing images (Dataset.py) ")
    start = timer()
    for dirName in os.listdir(image_dir):
        label_i = label[i]
        print("ONE_HOT_ROW = ", label_i)
        i += 1
        # log.info("Execution time of convLabels function = %.4f sec" % (end1-start1))
        path = os.path.join(image_dir, dirName)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            if os.path.isfile(img_path) and (img.endswith('jpeg') or
                                             (img.endswith('jpg'))):
                img_bytes = tf.read_file(img_path)
                img_u8 = tf.image.decode_jpeg(img_bytes, channels=3)
                img_u8_eval = session.run(img_u8)
                image = tf.image.convert_image_dtype(img_u8_eval, tf.float32)
                img_padded_or_cropped = tf.image.resize_image_with_crop_or_pad(image, IMG_SIZE, IMG_SIZE)
                img_padded_or_cropped = tf.reshape(img_padded_or_cropped, shape=[IMG_SIZE * IMG_SIZE, 3])
                yield img_padded_or_cropped.eval(session=session), label_i
    end = timer()
    log.info("End processing images (Dataset.py) - Time = %.2f sec" % (end-start))


def saveDataset(image_dir, file_path):
    with gzip.open(file_path, 'wb') as file:
        for img, label in convertDataset(image_dir):
            pickle.dump((img, label), file)
            

def loadDataset(file_path):
    with gzip.open(file_path) as file:
        while True:
            try:
                yield pickle.load(file)
            except EOFError:
                break

def saveShuffle(l, file_path='images_shuffle.pkl'):
    with gzip.open(file_path, 'wb') as file:
        for img, label in l:
            pickle.dump((img, label), file)
