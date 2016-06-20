from Dataset import Dataset
import os
import tensorflow as tf
import numpy as np

IMG_SIZE = 30
BATCH_SIZE = 32
IMAGE_DIR = os.getcwd() + '/small_dataset'

dataset = Dataset(IMAGE_DIR)

# Parameters
learning_rate = 0.001
max_epoch = 200000
display_step = 20

# Network Parameters
n_input = IMG_SIZE**2
n_classes = 4 
n_channels = 3
dropout = 0.8 # Dropout, probability to keep units

# Graph input
img_pl = tf.placeholder(tf.float32, [None, n_input, # n_channels
])
label_pl = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


class ConvNet(object):

    # Constructor
    def __init__(self):
        # Store layers weight & bias
        self.weights = {
            'wc1': tf.Variable(tf.random_normal([3, 3, 1, BATCH_SIZE])),
            'wc2': tf.Variable(tf.random_normal([3, 3, BATCH_SIZE, BATCH_SIZE*2])),
            'wc3': tf.Variable(tf.random_normal([3, 3, BATCH_SIZE*2, BATCH_SIZE*4])),
            'wd': tf.Variable(tf.random_normal([4*4*BATCH_SIZE*4, BATCH_SIZE*16])),
            'wfc': tf.Variable(tf.random_normal([BATCH_SIZE*16, BATCH_SIZE*16])),
            'out': tf.Variable(tf.random_normal([BATCH_SIZE*16, n_classes]))
        }
        self.biases = {
            'bc1': tf.Variable(tf.random_normal([BATCH_SIZE])),
            'bc2': tf.Variable(tf.random_normal([BATCH_SIZE*2])),
            'bc3': tf.Variable(tf.random_normal([BATCH_SIZE*4])),
            'bd': tf.Variable(tf.random_normal([BATCH_SIZE*16])),
            'bfc': tf.Variable(tf.random_normal([BATCH_SIZE*16])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

    # Return the next batch of size batch_size
    def nextBatch(self, imgs, labels, step, batch_size):
        s = step*batch_size
        return imgs[s:s+batch_size], labels[s:s+batch_size]

    """ 
    Create AlexNet model 
    """
    def conv2d(self, name, l_input, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

    def max_pool(self, name, l_input, k):
        return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

    def norm(self, name, l_input, lsize=4):
        return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

    def alex_net_model(self, _X, _weights, _biases, _dropout):
        # Reshape input picture
        _X = tf.reshape(_X, shape=[-1, IMG_SIZE, IMG_SIZE, 1])

        # Convolution Layer
        conv1 = self.conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
        # Max Pooling (down-sampling)
        pool1 = self.max_pool('pool1', conv1, k=2)
        # Apply Normalization
        norm1 = self.norm('norm1', pool1, lsize=4)
        # Apply Dropout
        norm1 = tf.nn.dropout(norm1, _dropout)

        # Convolution Layer
        conv2 = self.conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
        # Max Pooling (down-sampling)
        pool2 = self.max_pool('pool2', conv2, k=2)
        # Apply Normalization
        norm2 = self.norm('norm2', pool2, lsize=4)
        # Apply Dropout
        norm2 = tf.nn.dropout(norm2, _dropout)

        # Convolution Layer
        conv3 = self.conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
        # Max Pooling (down-sampling)
        pool3 = self.max_pool('pool3', conv3, k=2)
        # Apply Normalization
        norm3 = self.norm('norm3', pool3, lsize=4)
        # Apply Dropout
        norm3 = tf.nn.dropout(norm3, _dropout)

        # Fully connected layer
        dense = tf.reshape(norm3, [-1, _weights['wd'].get_shape().as_list()[0]])         # Reshape conv3 output to fit dense layer input
        fc1 = tf.nn.relu(tf.matmul(dense, _weights['wd']) + _biases['bd'], name='fc1')  # Relu activation
        fc2 = tf.nn.relu(tf.matmul(fc1, _weights['wfc']) + _biases['bfc'], name='fc2')    # Relu activation

        # Output, class prediction
        out = tf.matmul(fc2, _weights['out']) + _biases['out']
        return out

    def training(self):

        # Construct model
        pred = self.alex_net_model(img_pl, self.weights, self.biases, keep_prob)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, label_pl))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(label_pl,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Initializing the variables
        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            step = 0

            imgs = []
            labels = []
            for img, label in dataset.getDataset():
                imgs.append(img)
                labels.append(label)

            print 'Dataset created - images list and labels list'

            for epoch in xrange(max_epoch):
                for step in xrange((len(imgs)/BATCH_SIZE) +1):

                    batch_imgs, batch_labels = self.nextBatch(imgs, labels, step, BATCH_SIZE)

                    # Fit training using batch data
                    sess.run(optimizer, feed_dict={img_pl: batch_imgs, label_pl: batch_labels, keep_prob: dropout})

                    if step % display_step == 0:
                        print 'calculating accuracy'
                        # Calculate batch accuracy
                        acc = sess.run(accuracy, feed_dict={img_pl: batch_imgs, label_pl: batch_labels, keep_prob: 1.})
                        # Calculate batch loss
                        loss = sess.run(cost, feed_dict={img_pl: batch_imgs, label_pl: batch_labels, keep_prob: 1.})
                        print "Iter " + str(step*BATCH_SIZE) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)

            print "Optimization Finished!"

            # Save the models to disk
            save_model_ckpt = saver.save(sess, "/tmp/model.ckpt")
            print("Model saved in file %s" % save_model_ckpt)

    def test(self):
        pass
    #TODO: add test
        #print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})


    def predict(self):
        # Restore model from disk.
        # saver.restore(sess, "/tmp/model.ckpt")
        # print("Model restored")
        pass



### MAIN ###
def main():

    # create the object ConvNet
    conv_net = ConvNet()

    # TRAINING
    conv_net.training()

    # TESTING
    conv_net.test()

    # PREDICTION
    conv_net.prediction()


if __name__ == '__main__':
    main()
    
