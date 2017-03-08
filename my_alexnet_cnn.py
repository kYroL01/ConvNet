import Dataset
import os
import sys
import tensorflow as tf
import numpy as np
import logging as log
import timeit
import argparse

from Dataset import IMG_SIZE
from Dataset import LABELS_DICT

IMAGE_DIR = os.getcwd() + '/small_dataset'
PREDICT_IMAGE_DIR = os.getcwd() + '/test_dataset'
CKPT_DIR = 'ckpt_dir'
MODEL_CKPT = 'ckpt_dir/model.cktp'
# Parameters of Logistic Regression
BATCH_SIZE = 64

# Network Parameters
n_input = IMG_SIZE**2
n_classes = 4
n_channels = 3
dropout = 0.8 # Dropout, probability to keep units


class ConvNet(object):

    # Constructor
    def __init__(self, learning_rate, max_epochs, display_step, std_dev, dataset):

        # Initialize params
        self.learning_rate=learning_rate
        self.max_epochs=max_epochs
        self.display_step=display_step
        self.std_dev=std_dev
        self.dataset = dataset
        self.gen_imgs_lab = Dataset.loadDataset(dataset)
        
        # Store layers weight & bias
        self.weights = {
            'wc1': tf.Variable(tf.random_normal([11, 11, n_channels, BATCH_SIZE], stddev=std_dev)),
            'wc2': tf.Variable(tf.random_normal([5, 5, BATCH_SIZE, BATCH_SIZE*2], stddev=std_dev)),
            'wc3': tf.Variable(tf.random_normal([3, 3, BATCH_SIZE*2, BATCH_SIZE*4], stddev=std_dev)),
            'wc4': tf.Variable(tf.random_normal([3, 3, BATCH_SIZE*4, BATCH_SIZE*4], stddev=std_dev)),
            'wc5': tf.Variable(tf.random_normal([3, 3, BATCH_SIZE*4, 256], stddev=std_dev)),

            'wd': tf.Variable(tf.random_normal([1024, 4096])),
            'wfc': tf.Variable(tf.random_normal([4096, 1024], stddev=std_dev)),

            'out': tf.Variable(tf.random_normal([1024, n_classes], stddev=std_dev))
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
        self.keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
        
        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver()

        
    # Batch function - give the next batch of images and labels
    def BatchIterator(self, batch_size):
        imgs = []
        labels = []
            
        for img, label in self.gen_imgs_lab:
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
        return tf.nn.lrn(l_input, lsize, bias=2.0, alpha=0.001 / 9.0, beta=0.75, name=name)

    def alex_net_model(self, _X, _weights, _biases, _dropout):
        # Reshape input picture

        _X = tf.reshape(_X, shape=[-1, IMG_SIZE, IMG_SIZE, 3])

        # Convolution Layer 1
        conv1 = self.conv2d('conv1', _X, _weights['wc1'], _biases['bc1'], 4)
        print "conv1.shape: ", conv1.get_shape()
        # Max Pooling (down-sampling)
        pool1 = self.max_pool('pool1', conv1, k=3, s=2)
        print "pool1.shape:", pool1.get_shape()
        # Apply Normalization
        norm1 = self.norm('norm1', pool1, lsize=4)
        print "norm1.shape:", norm1.get_shape()
        # Apply Dropout
        dropout1 = tf.nn.dropout(norm1, _dropout)

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
        dropout2 = tf.nn.dropout(norm2, _dropout)
        print "dropout2.shape:", dropout2.get_shape()

        # Convolution Layer 3
        conv3 = self.conv2d('conv3', dropout2, _weights['wc3'], _biases['bc3'], s=1)
        print "conv3.shape:", conv3.get_shape()

        pool3 = self.max_pool('pool3', conv3, k=3, s=2)
        norm3 = self.norm('norm3', pool3, lsize=4)
        dropout3 = tf.nn.dropout(norm3, _dropout)

        # Convolution Layer 4
        conv4 = self.conv2d('conv4', dropout3, _weights['wc4'], _biases['bc4'], s=1)
        print "conv4.shape:", conv4.get_shape()

        pool4 = self.max_pool('pool4', conv4, k=3, s=2)
        norm4 = self.norm('norm4', pool4, lsize=4)
        dropout4 = tf.nn.dropout(norm4, _dropout)

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

        softmax_l = tf.nn.softmax(out)

        # The function returns the Logits to be passed to softmax
        return out, softmax_l

    # Method for training the model and testing its accuracy
    def training(self):
        # Launch the graph
        with tf.Session() as sess:
            # Construct model
            logits, prediction = self.alex_net_model(self.img_pl, self.weights, self.biases, self.keep_prob)

            # tf.nn.softmax(...) + cross_entropy(...)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.label_pl))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.1).minimize(loss)

            # Evaluate model
            print logits.get_shape(), self.label_pl.get_shape()
            correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(self.label_pl, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # Initializing the variables
            init = tf.initialize_all_variables()

            # Run the Op to initialize the variables.
            sess.run(init)
            summary_writer = tf.train.SummaryWriter(CKPT_DIR, graph=sess.graph)

            log.info('Dataset created - images list and labels list')
            log.info('Now split images and labels in Training and Test set...')


            ##################################################################

            # collect imgs for test
            tests_imgs_batches = [b for i, b in enumerate(self.BatchIterator(BATCH_SIZE)) if i < 3]

            # Run for epoch
            for epoch in range(self.max_epochs):
                log.info('Epoch %s' % epoch)
                self.gen_imgs_lab = Dataset.loadDataset(self.dataset)
                
                # Loop over all batches
                for step, elems in enumerate(self.BatchIterator(BATCH_SIZE)):
                    
                    ### from itrator return batch lists ###
                    batch_imgs_train, batch_labels_train = elems
                    _, train_acc, train_loss, train_logits = sess.run([optimizer, accuracy, loss, logits], feed_dict={self.img_pl: batch_imgs_train, self.label_pl: batch_labels_train, self.keep_prob: 1.0})
                    log.info("Training Accuracy = " + "{:.5f}".format(train_acc))
                    log.info("Training Loss = " + "{:.6f}".format(train_loss))
                    
            print "Optimization Finished!"

            # Save the models to disk
            save_model_ckpt = self.saver.save(sess, MODEL_CKPT)
            print("Model saved in file %s" % save_model_ckpt)

            # Test accuracy
            for step, elems in enumerate(tests_imgs_batches):
                batch_imgs_test, batch_labels_test = elems
                
                test_acc = sess.run(accuracy, feed_dict={self.img_pl: batch_imgs_test, self.label_pl: batch_labels_test, self.keep_prob: 1.0})
                print "Test accuracy: %.5f" % (test_acc)
                log.info("Test accuracy: %.5f" % (test_acc))

    
    def prediction(self):
        with tf.Session() as sess:

            # Construct model
            _, pred = self.alex_net_model(self.img_pl, self.weights, self.biases, self.keep_prob)

            prediction = tf.argmax(pred,1)

            # Restore model.
            ckpt = tf.train.get_checkpoint_state("ckpt_dir")
            if(ckpt):
                self.saver.restore(sess, MODEL_CKPT)
                print("Model restored")
            else:
                print "No model checkpoint found to restore - ERROR"
                return

            for dirName in os.listdir(PREDICT_IMAGE_DIR):
                path = os.path.join(PREDICT_IMAGE_DIR, dirName)
                for img in os.listdir(path):
                    print "reading image to classify... "
                    img_path = os.path.join(path, img)
                    print("IMG PATH = ", img_path)
                    # check if image is a correct JPG file
                    if(os.path.isfile(img_path) and (img_path.endswith('jpeg') or
                                                     (img_path.endswith('jpg')))):
                        # Read image and convert it
                        img_bytes = tf.read_file(img_path)
                        img_u8 = tf.image.decode_jpeg(img_bytes, channels=3)
                        #img_u8 = tf.image.decode_jpeg(img_bytes, channels=1)
                        img_u8_eval = sess.run(img_u8)
                        image = tf.image.convert_image_dtype(img_u8_eval, tf.float32)
                        img_padded_or_cropped = tf.image.resize_image_with_crop_or_pad(image, IMG_SIZE, IMG_SIZE)
                        img_padded_or_cropped = tf.reshape(img_padded_or_cropped, shape=[IMG_SIZE*IMG_SIZE, 3])
                        #img_padded_or_cropped = tf.reshape(img_padded_or_cropped, shape=[IMG_SIZE * IMG_SIZE])
                        # eval
                        img_eval = img_padded_or_cropped.eval()
                        # Run the model to get predictions
                        predict = sess.run(prediction, feed_dict={self.img_pl: [img_eval], self.keep_prob: 1.})
                        print "ConvNet prediction = %s" % (LABELS_DICT.keys()[LABELS_DICT.values().index(predict)]) # Print the name of class predicted

                    else:
                        print "ERROR IMAGE"

### MAIN ###
def main():

    np.random.seed(7)

    parser = argparse.ArgumentParser(description='A convolutional neural network for image recognition')
    subparsers = parser.add_subparsers()

    common_args = [
        (['-lr', '--learning-rate'], {'help':'learning rate', 'type':float, 'default':0.001}),
        (['-e', '--epochs'], {'help':'epochs', 'type':int, 'default':5}),
        (['-ds', '--display-step'], {'help':'display step', 'type':int, 'default':10}),
        (['-sd', '--std-dev'], {'help':'std-dev', 'type':float, 'default':1.0}),
        (['-d', '--dataset'],  {'help':'dataset file', 'type':str, 'default':'images_dataset.pkl'})
    ]

    # parser train
    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(which='train')
    for arg in common_args:
        parser_train.add_argument(*arg[0], **arg[1])

    # parser preprocessing
    parser_preprocess = subparsers.add_parser('preprocessing')
    parser_preprocess.set_defaults(which='preprocessing')
    parser_preprocess.add_argument('-f', '--file', help='output file', type=str, default='images_dataset.pkl')
    parser_preprocess.add_argument('-s', '--shuffle', help='shuffle dataset', action='store_true')
    parser_preprocess.set_defaults(shuffle=False)

    # parser predict
    parser_predict = subparsers.add_parser('predict')
    parser_predict.set_defaults(which='predict')
    for arg in common_args:
        parser_predict.add_argument(*arg[0], **arg[1])

    args = parser.parse_args()

    log.basicConfig(filename='FileLog.log', level=log.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filemode="w")


    if args.which in ('train', 'predict'):
        t = timeit.timeit("Dataset.loadDataset(IMAGE_DIR)", setup="from __main__ import *")

        # create the object ConvNet
        conv_net = ConvNet(args.learning_rate, args.epochs, args.display_step, args.std_dev, args.dataset)
        if args.which == 'train':
            # TRAINING
            log.info('Start training')                                                                                                  
            conv_net.training()
        else:
            # PREDICTION
            conv_net.prediction()
    # PREPROCESSING
    elif args.which == 'preprocessing':
        if args.shuffle: # shuffle dataset already converted
            l = [i for i in Dataset.loadDataset('images_dataset.pkl')]
            np.random.shuffle(l)
            Dataset.saveShuffle(l)
        else: # create dataset from images
            Dataset.saveDataset(IMAGE_DIR, args.file)

if __name__ == '__main__':
    main()
