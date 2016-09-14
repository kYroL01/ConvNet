import Dataset
import os
import sys
import tensorflow as tf
import numpy as np
import logging as log
import timeit

IMG_SIZE = 224
IMAGE_DIR = os.getcwd() + '/small_dataset'
CKPT_DIR = '/tmp/tf_logs/ConvNet'
MODEL_CKPT = '/tmp/tf_logs/ConvNet/model.cktp'
# Parameters of Logistic Regression
BATCH_SIZE = 20

# Network Parameters
n_input = IMG_SIZE**2
n_classes = 4 
n_channels = 3
dropout = 0.8 # Dropout, probability to keep units


class ConvNet(object):

    # Constructor
    def __init__(self, learning_rate, max_epochs, display_step, std_dev, images, labels):

        # Initialize params
        self.learning_rate=learning_rate
        self.max_epochs=max_epochs
        self.display_step=display_step
        self.std_dev=std_dev
        self.images=images
        self.labels=labels
        
        # Store layers weight & bias
        self.weights = {
            'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96], stddev=std_dev)),
            'wc2': tf.Variable(tf.random_normal([5, 5, 96, 192], stddev=std_dev)),
            'wc3': tf.Variable(tf.random_normal([3, 3, 192, 384], stddev=std_dev)),
            'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384], stddev=std_dev)),
            'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=std_dev)),
            
            'wd': tf.Variable(tf.random_normal([12544, 4096])),
            'wfc': tf.Variable(tf.random_normal([4096, 1024], stddev=std_dev)),
            
            'out': tf.Variable(tf.random_normal([1024, n_classes], stddev=std_dev))
        }
        
        self.biases = {
            'bc1': tf.Variable(tf.random_normal([96])),
            'bc2': tf.Variable(tf.random_normal([192])),
            'bc3': tf.Variable(tf.random_normal([384])),
            'bc4': tf.Variable(tf.random_normal([384])),
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
    def BatchIterator(self, images, labels, batch_size, step):
            s = step*batch_size
            print ("s = ", s)
            print ("end = ", s+batch_size)
            yield images[s:s+batch_size], labels[s:s+batch_size]

            
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
        # log.info("conv1.shape: ", conv1.get_shape())
        # Max Pooling (down-sampling)
        pool1 = self.max_pool('pool1', conv1, k=3, s=2)
        # log.info("pool1.shape:", pool1.get_shape())
        # Apply Normalization
        norm1 = self.norm('norm1', pool1, lsize=4)
        # log.info("norm1.shape:", norm1.get_shape())
        # Apply Dropout
        dropout1 = tf.nn.dropout(norm1, _dropout)    

        # Convolution Layer 2
        conv2 = self.conv2d('conv2', dropout1, _weights['wc2'], _biases['bc2'], s=1)
        # log.info("conv2.shape:", conv2.get_shape())
        # Max Pooling (down-sampling)
        pool2 = self.max_pool('pool2', conv2, k=3, s=2)
        # log.info("pool2.shape:", pool2.get_shape())
        # Apply Normalization
        norm2 = self.norm('norm2', pool2, lsize=4)
        # log.info("norm2.shape:", norm2.get_shape())
        # Apply Dropout
        dropout2 = tf.nn.dropout(norm2, _dropout)
        # log.info("dropout2.shape:", dropout2.get_shape())

        # Convolution Layer 3
        conv3 = self.conv2d('conv3', dropout2, _weights['wc3'], _biases['bc3'], s=1)
        # log.info("conv3.shape:", conv3.get_shape())

        # Convolution Layer 4
        conv4 = self.conv2d('conv4', conv3, _weights['wc4'], _biases['bc4'], s=1)
        # log.info("conv4.shape:", conv4.get_shape())

        # Convolution Layer 5
        conv5 = self.conv2d('conv5', conv4, _weights['wc5'], _biases['bc5'], s=1)
        # log.info("conv5.shape:", conv5.get_shape())
        pool5 = self.max_pool('pool5', conv5, k=3, s=2)
        # log.info("pool5.shape:", pool5.get_shape())

        # Fully connected layer 1
        pool5_shape = pool5.get_shape().as_list()
        dense = tf.reshape(pool5, [-1, pool5_shape[1] * pool5_shape[2] * pool5_shape[3]])
        # log.info("dense.shape:", dense.get_shape())
        fc1 = tf.nn.relu(tf.matmul(dense, _weights['wd']) + _biases['bd'], name='fc1')  # Relu activation
        # log.info("fc1.shape:", fc1.get_shape())
        
        # Fully connected layer 2
        fc2 = tf.nn.relu(tf.matmul(fc1, _weights['wfc']) + _biases['bfc'], name='fc2')  # Relu activation
        # log.info("fc2.shape:", fc2.get_shape())

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

            # TO check # Define loss and optimizer
            # http://stackoverflow.com/questions/33922937/why-does-tensorflow-return-nan-nan-instead-of-probabilities-from-a-csv-file
            # equivalent to
            # tf.nn.softmax(...) + cross_entropy(...)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.label_pl))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

            # Evaluate model
            correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(self.label_pl, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # Initializing the variables
            init = tf.initialize_all_variables()

            # Run the Op to initialize the variables.
            sess.run(init)
            summary_writer = tf.train.SummaryWriter(CKPT_DIR, graph=sess.graph)
            step = 0

            log.info('Dataset created - images list and labels list')
            log.info('Now split images and labels in Training and Test set...')

            # count total number of imgs
            img_count = Dataset.getNumImages(IMAGE_DIR)

            # index for num imgs of training set
            idx = int(4 * img_count/5)

            # Split images and labels
            train_imgs = self.images[0:idx]
            print("len tr imgs = %d"%len(train_imgs))
            train_labels = self.labels[0:idx]
            print("len tr lab = %d"%len(train_labels))
            test_imgs    = self.images[idx:img_count]
            print("len tst imgs = %d"%len(test_imgs))
            test_labels  = self.labels[idx:img_count]
            print("len tst lab = %d"%len(test_labels))

            ##################################################################

            # Run for epoch
            for epoch in range(self.max_epochs):
                avg_loss = 0.
                num_batch = ((idx+1) / BATCH_SIZE) # 8
                print("num_batch %d "%num_batch)
qq                
                # Loop over all batches
                for step in range(num_batch):

                    ### create itrator over batch list ###
                    iter_= self.BatchIterator(train_imgs, train_labels, BATCH_SIZE, step)
                    ### call next() for next batch of imgs and labels ###
                    batch_imgs_train, batch_labels_train = iter_.next()

                    # Fit training using batch data
                    _, single_loss = sess.run([optimizer, loss], feed_dict={self.img_pl: batch_imgs_train, self.label_pl: batch_labels_train, self.keep_prob: dropout})
                    # Compute average loss
                    avg_loss += single_loss

                    # Display logs per epoch step
                    if step % self.display_step == 0:
                        # print "Step %03d - Epoch %03d/%03d loss: %.7f - single_loss %.7f" % (step, epoch, self.max_epochs, avg_loss/step, single_loss)
                        # log.info("Step %03d - Epoch %03d - loss: %.7f - single_loss %.7f" % (step, epoch, avg_loss/step, single_loss))
                        # Calculate training batch accuracy and batch loss
                        train_acc, train_loss = sess.run([accuracy, loss], feed_dict={self.img_pl: batch_imgs_train, self.label_pl: batch_labels_train, self.keep_prob: 1.})
                        print "Training Accuracy = " + "{:.5f}".format(train_acc)
                        log.info("Training Accuracy = " + "{:.5f}".format(train_acc))
                        print "Training Loss = " + "{:.6f}".format(train_loss)
                        log.info("Training Loss = " + "{:.6f}".format(train_loss))

            print "Optimization Finished!"

            print "Accuracy = ", sess.run(accuracy, feed_dict={self.img_pl: batch_imgs_train, self.label_pl: batch_labels_train, self.keep_prob: 1.0})

            # Save the models to disk
            save_model_ckpt = self.saver.save(sess, MODEL_CKPT)
            print("Model saved in file %s" % save_model_ckpt)

            #upgrade num_batch for test images number
            num_batch = (len(test_imgs) / BATCH_SIZE) # 2

            # Test accuracy
            for step in range(num_batch):

                ### nextbatch function for test ###
                iter_= self.BatchIterator(test_imgs, test_labels, BATCH_SIZE, step)
                batch_imgs_test, batch_labels_test = iter_.next()
                test_acc = sess.run(accuracy, feed_dict={self.img_pl: batch_imgs_test, self.label_pl: batch_labels_test, self.keep_prob: 1.0})
                print "Test accuracy: %.5f" % (test_acc)
                log.info("Test accuracy: %.5f" % (test_acc))

            # Classification (two images as example)
            classification = sess.run(tf.argmax(prediction,1), feed_dict={self.img_pl: [test_imgs[0]], self.keep_prob: 1.0})
            print "ConvNet prediction (in training) = ", classification
            classification = sess.run(tf.argmax(prediction,1), feed_dict={self.img_pl: [test_imgs[22]], self.keep_prob: 1.0})
            print "ConvNet prediction (in training) = ", classification

                
    def prediction(self, img_path):
        with tf.Session() as sess:

            # Construct model
            _, pred = self.alex_net_model(self.img_pl, self.weights, self.biases, self.keep_prob)

            prediction = tf.argmax(pred,1)

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

                # Restore model.
                ckpt = tf.train.get_checkpoint_state("/tmp/")
                if(ckpt):
                    self.saver.restore(sess, MODEL_CKPT)
                    print("Model restored")
                else:
                    print "No model checkpoint found to restore - ERROR"
                    return

                # Run the model to get predictions
                predict = sess.run(prediction, feed_dict={self.img_pl: [img_eval], self.keep_prob: 1.})
                print "ConvNet prediction = ", predict

            else:
                print "ERROR IMAGE"


### MAIN ###
def main():

    # args from command line:
    # 1) learning_rate
    # 2) max_epochs
    # 3) display_step
    # 4) std_dev
    learning_rate = float(sys.argv[1])
    max_epochs = int(sys.argv[2])
    display_step = int(sys.argv[3])
    std_dev = float(sys.argv[4])

    log.basicConfig(filename='FileLog.log', level=log.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filemode="w")

    # generator object of imgs and labels from getDataset()
    imgs_labels_gen = Dataset.getDataset(IMAGE_DIR)
    
    # convert the generator object returned from dataset.getDataset() in list of tuple
    imgs_labels = list(imgs_labels_gen)
    images, labels = zip(*imgs_labels)

    t = timeit.timeit("Dataset.getDataset(IMAGE_DIR)", setup="from __main__ import *")
    log.info("Execution time of Dataset.getDataset(IMAGE_DIR) (__main__) = %.4f sec" % t)
    
    # create the object ConvNet
    conv_net = ConvNet(learning_rate, max_epochs, display_step, std_dev, images, labels)

    # TRAINING
    conv_net.training()

    # PREDICTION
    for dirName in os.listdir(IMAGE_DIR):
        path = os.path.join(IMAGE_DIR, dirName)
        for img in os.listdir(path):
            print "reading image to classify... "
            img_path = os.path.join(path, img)
            conv_net.prediction(img_path)
            print("IMG PATH = ", img_path)


if __name__ == '__main__':
    main()
    
