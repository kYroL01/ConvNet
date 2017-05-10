import Dataset
import os
import sys
import tensorflow as tf
import numpy as np
import logging as log
import timeit
import argparse

from sklearn import metrics

from Dataset import IMG_SIZE
from Dataset import LABELS_DICT

TRAIN_IMAGE_DIR = os.getcwd() + '/small_dataset'
TEST_IMAGE_DIR = os.getcwd() + '/test_dataset'
CKPT_DIR = 'ckpt_dir'
MODEL_CKPT = 'ckpt_dir/model.cktp'

### Parameters for Logistic Regression ###
BATCH_SIZE = 64

### Network Parameters ###
n_input = IMG_SIZE**2
n_classes = 4
n_channels = 3
input_dropout = 0.8
hidden_dropout = 0.5



class ConvNet(object):

    ## Constructor to build the model for the training ##
    def __init__(self, **kwargs):

        params = set(['learning_rate','max_epochs','display_step', 'std_dev', 'dataset_training', 'dataset_test'])

        # initialize all allowed keys to false
        self.__dict__.update((key, False) for key in params)
        # and update the given keys by their given values
        self.__dict__.update((key, value) for key, value in kwargs.iteritems() if key in params)

        if(self.dataset_training != False):
            self.train_imgs_lab = Dataset.loadDataset(self.dataset_training)
        else:
            self.test_imgs_lab = Dataset.loadDataset(self.dataset_test)

        
        # Store layers weight & bias
        self.weights = {
            'wc1': tf.Variable(tf.random_normal([11, 11, n_channels, BATCH_SIZE], stddev=self.std_dev)),
            'wc2': tf.Variable(tf.random_normal([5, 5, BATCH_SIZE, BATCH_SIZE*2], stddev=self.std_dev)),
            'wc3': tf.Variable(tf.random_normal([3, 3, BATCH_SIZE*2, BATCH_SIZE*4], stddev=self.std_dev)),
            'wc4': tf.Variable(tf.random_normal([3, 3, BATCH_SIZE*4, BATCH_SIZE*4], stddev=self.std_dev)),
            'wc5': tf.Variable(tf.random_normal([3, 3, BATCH_SIZE*4, 256], stddev=self.std_dev)),

            'wd': tf.Variable(tf.random_normal([1024, 4096])),
            'wfc': tf.Variable(tf.random_normal([4096, 1024], stddev=self.std_dev)),

            'out': tf.Variable(tf.random_normal([1024, n_classes], stddev=self.std_dev))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([BATCH_SIZE])),
            'bc2': tf.Variable(tf.random_normal([BATCH_SIZE*2])),
            'bc3': tf.Variable(tf.random_normal([BATCH_SIZE*4])),
            'bc4': tf.Variable(tf.random_normal([BATCH_SIZE*4])),
            'bc5': tf.Variable(tf.random_normal([256])),

            'bd': tf.Variable(tf.random_normal([4096])),
            'bfc': tf.Variable(tf.random_normal([1024])),

            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # Graph input
        self.img_pl = tf.placeholder(tf.float32, [None, n_input, n_channels])
        self.label_pl = tf.placeholder(tf.float32, [None, n_classes])
        self.keep_prob_in = tf.placeholder(tf.float32)
        self.keep_prob_hid = tf.placeholder(tf.float32)
        
        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver()



    # Batch function for Training - give the next batch of images and labels
    def BatchIteratorTraining(self, batch_size):
        imgs = []
        labels = []
            
        for img, label in self.train_imgs_lab:
            imgs.append(img)
            labels.append(label)
            if len(imgs) == batch_size:
                yield imgs, labels
                imgs = []
                labels = []
        if len(imgs) > 0:
            yield imgs, labels


    # Batch function for Testing - give the next batch of images and labels
    def BatchIteratorTesting(self, batch_size):
        imgs = []
        labels = []

        for img, label in self.test_imgs_lab:
            imgs.append(img)
            labels.append(label)
            if len(imgs) == batch_size:
                yield imgs, labels
                imgs = []
                labels = []
        if len(imgs) > 0:
            yield imgs, labels


            
    """ 
    Create AlexNet model 
    """
    def conv2d(self, name, l_input, w, b, s):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, s, s, 1], padding='SAME'), b), name=name)

    def max_pool(self, name, l_input, k, s):
        return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME', name=name)

    def norm(self, name, l_input, lsize):
        return tf.nn.lrn(l_input, lsize, bias=2.0, alpha=2e-05, beta=0.75, name=name)

    def alex_net_model(self, _X, _weights, _biases, input_dropout, hidden_dropout):
        # Reshape input picture

        _X = tf.reshape(_X, shape=[-1, IMG_SIZE, IMG_SIZE, 3])

        # Convolution Layer 1
        conv1 = self.conv2d('conv1', _X, _weights['wc1'], _biases['bc1'], s=4)
        print "conv1.shape: ", conv1.get_shape()
        # Max Pooling (down-sampling)
        pool1 = self.max_pool('pool1', conv1, k=3, s=2)
        print "pool1.shape:", pool1.get_shape()
        # Apply Normalization
        norm1 = self.norm('norm1', pool1, lsize=4)
        print "norm1.shape:", norm1.get_shape()
        # Apply Dropout
        dropout1 = tf.nn.dropout(norm1, input_dropout)

        # Convolution Layer 2
        conv2 = self.conv2d('conv2', dropout1, _weights['wc2'], _biases['bc2'], s=1)
        print "conv2.shape:", conv2.get_shape()
        # Max Pooling (down-sampling)
        pool2 = self.max_pool('pool2', conv2, k=3, s=2)
        print "pool2.shape:", pool2.get_shape()
        # Apply Normalization
        norm2 = self.norm('norm2', pool2, lsize=4)
        print "norm2.shape:", norm2.get_shape()
        # Apply Dropout
        dropout2 = tf.nn.dropout(norm2, hidden_dropout)
        print "dropout2.shape:", dropout2.get_shape()

        # Convolution Layer 3
        conv3 = self.conv2d('conv3', dropout2, _weights['wc3'], _biases['bc3'], s=1)
        print "conv3.shape:", conv3.get_shape()

        pool3 = self.max_pool('pool3', conv3, k=3, s=2)
        norm3 = self.norm('norm3', pool3, lsize=4)
        dropout3 = tf.nn.dropout(norm3, hidden_dropout)

        # Convolution Layer 4
        conv4 = self.conv2d('conv4', dropout3, _weights['wc4'], _biases['bc4'], s=1)
        print "conv4.shape:", conv4.get_shape()

        pool4 = self.max_pool('pool4', conv4, k=3, s=2)
        norm4 = self.norm('norm4', pool4, lsize=4)
        dropout4 = tf.nn.dropout(norm4, hidden_dropout)

        # Convolution Layer 5
        conv5 = self.conv2d('conv5', dropout4, _weights['wc5'], _biases['bc5'], s=1)
        print "conv5.shape:", conv5.get_shape()

        pool5 = self.max_pool('pool5', conv5, k=3, s=2)
        print "pool5.shape:", pool5.get_shape()

        # Fully connected layer 1
        pool5_shape = pool5.get_shape().as_list()
        dense = tf.reshape(pool5, [-1, pool5_shape[1] * pool5_shape[2] * pool5_shape[3]])
        print "dense.shape:", dense.get_shape()
        fc1 = tf.nn.relu(tf.matmul(dense, _weights['wd']) + _biases['bd'], name='fc1')  # Relu activation
        print "fc1.shape:", fc1.get_shape()

        # Fully connected layer 2
        fc2 = tf.nn.relu(tf.matmul(fc1, _weights['wfc']) + _biases['bfc'], name='fc2')  # Relu activation
        print "fc2.shape:", fc2.get_shape()

        # Output, class prediction LOGITS
        out = tf.matmul(fc2, _weights['out']) + _biases['out']

        # The function returns the Logits to be passed to softmax and the Softmax for the PREDICTION
        return out

    # Method for training the model and testing its accuracy
    def training(self):

        # Launch the graph
        with tf.Session() as sess:

            ## Construct model: prepare logits, loss and optimizer ##

            # logits: unnormalized log probabilities
            logits = self.alex_net_model(self.img_pl, self.weights, self.biases, self.keep_prob_in, self.keep_prob_hid)

            # loss: cross-entropy between the target and the softmax activation function applied to the model's prediction
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label_pl))
            tf.summary.scalar("cross-entropy_for_loss", loss)
            # optimizer: find the best gradients of the loss with respect to each of the variables
            train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.1).minimize(loss)
            
            print logits.get_shape(), self.label_pl.get_shape()

            ## Evaluate model: the degree to which the result of the prediction conforms to the correct value ##
            
            # list of booleans
            correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(self.label_pl, 1))
            # [True, False, True, True] -> [1,0,1,1] -> 0.75
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

            # Initializing the variables
            init = tf.global_variables_initializer()
            # Run the Op to initialize the variables.
            sess.run(init)
            summary_writer = tf.summary.FileWriter(CKPT_DIR, graph=sess.graph)

            ##################################################################

            # collect imgs for validation
            validation_imgs_batch = [b for i, b in enumerate(self.BatchIteratorTraining(BATCH_SIZE)) if i < 6]

            # Run for epoch
            for epoch in range(self.max_epochs):
                print "epoch = %d" % epoch
                log.info("Epoch %s" % epoch)
                self.train_imgs_lab = Dataset.loadDataset(self.dataset_training) # necessary 'cause of the yeld
                
                # Loop over all batches
                for step, elems in enumerate(self.BatchIteratorTraining(BATCH_SIZE)):
                    print "step = %d" % step
                    ### from iterator return batch lists ###
                    batch_imgs_train, batch_labels_train = elems
                    _, train_acc, train_loss = sess.run([train_step, accuracy, loss], feed_dict={self.img_pl: batch_imgs_train, self.label_pl: batch_labels_train, self.keep_prob_in: 1.0, self.keep_prob_hid: 1.0})
                    if step % self.display_step == 0:
                        log.info("Training Accuracy = " + "{:.5f}".format(train_acc))
                        log.info("Training Loss = " + "{:.6f}".format(train_loss))
                    
            print "Optimization Finished!"

            # Save the models to disk
            save_model_ckpt = self.saver.save(sess, MODEL_CKPT)
            print("Model saved in file %s" % save_model_ckpt)

            ##################################################################

            ### Metrics ###
            y_p = tf.argmax(logits,1) # the value predicted

            target_names = ['class 0', 'class 1', 'class 2', 'class 3']
            list_pred_total = []
            list_true_total = []

            # Accuracy Precision Recall F1-score by VALIDATION IMAGES
            for step, elems in enumerate(validation_imgs_batch):

                batch_imgs_valid, batch_labels_valid = elems
                valid_acc, y_pred = sess.run([accuracy, y_p], feed_dict={self.img_pl: batch_imgs_valid, self.label_pl: batch_labels_valid, self.keep_prob_in: 1.0, self.keep_prob_hid: 1.0})
                log.info("Validation accuracy = " + "{:.5f}".format(valid_acc))
                list_pred_total.extend(y_pred)
                y_true = np.argmax(batch_labels_valid,1)
                list_true_total.extend(y_true)

            # Classification Report (PRECISION - RECALL - F1 SCORE)
            log.info("\n")
            log.info(metrics.classification_report(list_true_total, list_pred_total, target_names=target_names))
            print(metrics.classification_report(list_true_total, list_pred_total, target_names=target_names))

            # Network Input Values
            log.info("Learning Rate " + "{:.4f}".format(self.learning_rate))
            log.info("Number of epochs " + "{:d}".format(self.max_epochs))
            log.info("Standard Deviation " + "{:.2f}".format(self.std_dev))


    
    def prediction(self):
        with tf.Session() as sess:

            # Construct model
            pred = self.alex_net_model(self.img_pl, self.weights, self.biases, self.keep_prob_in, self.keep_prob_hid)

            # Restore model.
            ckpt = tf.train.get_checkpoint_state("ckpt_dir")
            if(ckpt):
                self.saver.restore(sess, MODEL_CKPT)
                print "Model restored"
            else:
                print "No model checkpoint found to restore - ERROR"
                return

            ### M ###
            y_p = tf.argmax(pred,1) # the value predicted

            target_names = ['class 0', 'class 1', 'class 2', 'class 3']
            list_pred_total = []
            list_true_total = []

            # Accuracy Precision Recall F1-score by TEST IMAGES                                              
            for step, elems in enumerate(self.BatchIteratorTesting(BATCH_SIZE)):

                batch_imgs_test, batch_labels_test = elems

                y_pred = sess.run(y_p, feed_dict={self.img_pl: batch_imgs_test, self.keep_prob_in: 1.0, self.keep_prob_hid: 1.0})
                print("batch predict = %d" % len(y_pred))
                list_pred_total.extend(y_pred)
                y_true = np.argmax(batch_labels_test,1)
                print("batch real = %d" % len(y_true))
                list_true_total.extend(y_true)

            # Classification Report (PRECISION - RECALL - F1 SCORE)
            log.info(metrics.classification_report(list_true_total, list_pred_total, target_names=target_names))
            print(metrics.classification_report(list_true_total, list_pred_total, target_names=target_names))

            # Network Input Values
            log.info("Learning Rate " + "{:.4f}".format(self.learning_rate))
            log.info("Number of epochs " + "{:d}".format(self.max_epochs))
            log.info("Standard Deviation " + "{:.2f}".format(self.std_dev))




### MAIN ###
def main():

    np.random.seed(7)

    parser = argparse.ArgumentParser(description='A convolutional neural network for image recognition')
    subparsers = parser.add_subparsers()

    training_args = [
        (['-lr', '--learning-rate'], {'help':'learning rate', 'type':float, 'default':0.001}),
        (['-e', '--max_epochs'], {'help':'max epochs', 'type':int, 'default':100}),
        (['-ds', '--display-step'], {'help':'display step', 'type':int, 'default':10}),
        (['-sd', '--std-dev'], {'help':'std-dev', 'type':float, 'default':1.0}),
        (['-dtr', '--dataset_training'],  {'help':'dataset training file', 'type':str, 'default':'images_shuffled.pkl'})
    ]

    test_args = [
        (['-dts', '--dataset_test'],  {'help':'dataset test file', 'type':str, 'default':'images_test_dataset.pkl'})
    ]

    # parser train
    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(which='train')
    for arg in training_args:
        parser_train.add_argument(*arg[0], **arg[1])

    # parser preprocessing training data
    parser_preprocess = subparsers.add_parser('preprocessing_training')
    parser_preprocess.set_defaults(which='preprocessing_training')
    parser_preprocess.add_argument('-f', '--file', help='output training file', type=str, default='images_dataset.pkl')
    parser_preprocess.add_argument('-s', '--shuffle', help='shuffle training dataset', action='store_true')
    parser_preprocess.set_defaults(shuffle=False)

    # parser preprocessing test data
    parser_preprocess = subparsers.add_parser('preprocessing_test')
    parser_preprocess.set_defaults(which='preprocessing_test')
    parser_preprocess.add_argument('-t', '--test', help='output test file', type=str, default='images_test_dataset.pkl')

    # parser predict
    parser_predict = subparsers.add_parser('predict')
    parser_predict.set_defaults(which='predict')
    for arg in test_args:
        parser_predict.add_argument(*arg[0], **arg[1])

    args = parser.parse_args()

    # FILE LOG
    log.basicConfig(filename='FileLog.log', level=log.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filemode="w")

    # TRAINING & PREDICTION
    if args.which in ('train', 'predict'):
        t = timeit.timeit("Dataset.loadDataset(TRAIN_IMAGE_DIR)", setup="from __main__ import *")

        # create the object ConvNet

        if args.which == 'train':
            # TRAINING
            conv_net = ConvNet(learning_rate=args.learning_rate, max_epochs=args.max_epochs, display_step=args.display_step,
                               std_dev=args.std_dev, dataset_training=args.dataset_training)
            # count total number of imgs in training
            train_img_count = Dataset.getNumImages(TRAIN_IMAGE_DIR)
            log.info("Training set num images = %d" % train_img_count)
            conv_net.training()
        else:
            # PREDICTION
            conv_net = ConvNet(dataset_test=args.dataset_test)
            # count total number of imgs in test
            test_img_count = Dataset.getNumImages(TEST_IMAGE_DIR)
            log.info("Test set num images = %d" % test_img_count)
            conv_net.prediction()

    # PREPROCESSING TRAINING
    elif args.which == 'preprocessing_training':
            if args.shuffle:
                l = [i for i in Dataset.loadDataset('images_dataset.pkl')]
                np.random.shuffle(l)
                Dataset.saveShuffle(l)
            else:
                Dataset.saveDataset(TRAIN_IMAGE_DIR, args.file)

    # PREPROCESSING TEST
    elif args.which == 'preprocessing_test':
            Dataset.saveDataset(TEST_IMAGE_DIR, args.test)



                

if __name__ == '__main__':
    main()
