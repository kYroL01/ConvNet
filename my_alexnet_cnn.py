from Dataset import Dataset
import tensorflow as tf
import utils

# create dataset object
dataset = Dataset(utils.image_dir)
# create dictionary data structure for the images
dataset_dict = dataset.create_dataset(utils.test_percentage, utils.validation_percentage)

# Images Parameters
IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = 4   # classes: Normal - Porn_difficult - Porn_easy - Violence
BATCH_SIZE = 10
TOT_IMAGES = dataset.get_Num_images()

# Model Parameters
learning_rate = 0.01
max_epoch = 2560  # ~ 40 iterations
step = 20
dropout = 0.8     # Dropout, probability to keep units


### FUNCTIONS FOR CREATE THE MODEL ###

# Convolutional function
def conv2d(name, img_input, w, b, f):

    print(img_input.get_shape())
    print(w.get_shape())
    
    # return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img_input, w, strides=[1, f, f, 3], padding='SAME'), b), name=name)
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img_input, w, strides=[1, f, f, 1], padding='SAME'), b), name=name)

# Max Pooling function
def max_pool(name, img_input, k):
    return tf.nn.max_pool(img_input, ksize=[1, k, k, 1], strides=[1, k, k, 3], padding='SAME', name=name)

# Normalization function
def norm(name, img_input, size=4):
    return tf.nn.lrn(img_input, size, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


# Implementation of AlexNet model:
# 3 convolutional layers and 2 fully connected layers
def model(_img, _weights, _biases, _dropout):
    # Reshape input picture
    _img = tf.reshape(_img, shape=[-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    # Convolution Layer 1
    conv1 = conv2d('conv1', _img, _weights['wc1'], _biases['bc1'], f=4)
    # Max Pooling (down-sampling)
    pool1 = max_pool('pool1', conv1, k=2)
    # Apply Normalization
    norm1 = norm('norm1', pool1, size=4)
    # Apply Dropout
    norm1 = tf.nn.dropout(norm1, _dropout)

    # Convolution Layer 2
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'], f=1)
    # Max Pooling (down-sampling)
    pool2 = max_pool('pool2', conv2, k=2)
    # Apply Normalization
    norm2 = norm('norm2', pool2, size=4)
    # Apply Dropout
    norm2 = tf.nn.dropout(norm2, _dropout)

    # Convolution Layer 3
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'], f=1)

    # Convolution Layer 4
    conv4 = conv2d('conv4', conv3, _weights['wc4'], _biases['bc4'], f=1)

    # Convolution Layer 5
    conv5 = conv2d('conv5', conv4, _weights['wc5'], _biases['bc5'], f=1)
    # Max Pooling (down-sampling)
    pool5 = max_pool('pool5', conv5, k=2)

    # Reshape to a 2-D matrix conv3 output to fit dense layer input
    reshape = tf.reshape(pool5, [-1, _weights['wfc1'].get_shape().as_list()[0]])
    # Fully connected layer1
    fc1 = tf.nn.relu(tf.matmul(reshape, _weights['wfc1']) + _biases['bfc1'], name='fc1')
    # Fully connected layer2
    fc2 = tf.nn.relu(tf.matmul(fc1, _weights['wfc2']) + _biases['bfc2'], name='fc2')
    # Output, class prediction
    out_pred = tf.matmul(fc2, _weights['out']) + _biases['out']
    
    return out_pred



""" Calculates the loss from the logits and the labels.
  -args: 
         logits = Logits tensor (float)
         labels: Labels tensor (int32)
  -ret:
         loss: Loss tenso (float)
"""
def loss(logits, labels):
    # converted to 64-bit integers
    labels = tf.to_int64(labels)
    # produce 1-hot labels from the labels_placeholder and compare the output logits with those 1-hot labels.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels), name = 'loss')
    return loss



""" Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Optimizer returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
  -args:
         loss: Loss tensor, from loss().
         learning_rate: The learning rate to use for gradient descent.
  -ret:
         optim: The Optimizer for training.
  """
def training(loss, learning_rate):
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    return optim



""" Evaluate the quality of the logits at predicting the label.
  -args:
         logits: Logits tensor (float)
         labels: Labels tensor (int32)
  -ret:
         A scalar int32 tensor with the number of examples (out of batch_size)
         that were predicted correctly.
"""
def evaluation(logits, labels):
    # returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    sum_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
    
    return sum_correct



""" Fills the feed_dict for training the given step.
  -args:
         ph_images: images placeholder, from placeholder_inputs().
         ph_labels: labels placeholder, from placeholder_inputs().
  -ret:
         feed_dict: The feed dictionary mapping from placeholders to values.
"""
def fill_feed_dict(ph_images, ph_labels, val_dropout, start, end):
    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` samples.
    images_feed, labels_feed = dataset.convert_to_array()
    feed_dict = {
        ph_images: [resize_image_with_crop_or_pad(img, 300, 300) for img in images_feed[start:end]],
        ph_labels: labels_feed[start:end],
        keep_prob: val_dropout
    }
    return feed_dict



""" Run the training graph for a number of step
  -args:
         ph_images:  images placeholder, from placeholder_inputs().
         ph_labels:  labels placeholder, from placeholder_inputs().
         _keep_prob: probability for the dropbout
         _weights:   weights matrix for the softmax regression
         _biases:    biases vector for the softmax regression
"""
def run_training_graph(ph_images, ph_labels, _keep_prob, _weights, _biases):

    # Build a Graph that computes predictions from the inference model.
    with tf.Graph().as_default():

        ### call model() function
        logits = model(ph_images, _weights, _biases, _keep_prob)

        ### Call loss() function
        global loss
        loss = loss(logits, ph_labels)

        ### Call training() function
        optimizer = training(loss, learning_rate)

        ### Call evaluation() function to compare the logits to the labels during evaluation.
        eval_correct = evaluation(logits, ph_labels)

        # PREDICTIONS for the training. It is use the Softmax function
        train_prediction = tf.nn.softmax(logits)

        # Evaluate the model:
        correct_pred = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(ph_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Launch the graph
        with tf.Session() as sess:

            # Initializing the variables
            init = tf.initialize_all_variables()
            sess.run(init)
            
            # Create a SummaryWriter to output summaries and the Graph into log files
            summary_writer_logs = tf.train.SummaryWriter('/tmp/tensorflow_logs', sess.graph)
            
            # After everything is built, start the training loop.
            # while (step * batch_size) < max_steps:
            for epoch in range(max_epoch):
                start_time = time.time()

                # Train in batches
                for start in range(0, TOT_IMAGES, BATCH_SIZE):
                    
                    end = start + BATCH_SIZE
                    
                    feed_dict_train = fill_feed_dict(ph_images, ph_labels, dropout, start, end)
                    feed_dict_accloss = fill_feed_dict(ph_images, ph_labels, 1., start, end)

                    # Feed training using batch data.
                    # This dictionary maps the batch data (as a numpy array) to the
                    # node to be fed.
                    sess.run(optimizer, feed_dict = feed_dict_train)

                    # every step (20) we calculate accuracy and loss
                    if epoch % step == 0:
                        # Calculate batch accuracy
                        acc = sess.run(accuracy, feed_dict = feed_dict_accloss)
                        # Calculate batch loss
                        loss_b = sess.run(loss, feed_dict = feed_dict_accloss)
                        
                        print "Iteration " + str(epoch * BATCH_SIZE) + ", Loss = " + "{:.6f}".format(loss_b) + ", Training Accuracy = " + "{:.5f}".format(acc)

            # TO DO #
            # Calculate accuracy for TEST images

            # Update the events file
            summary_str = sess.run(summary_op)

            # Save the models to disk
            save_model_ckpt = saver.save(sess, "/tmp/model.ckpt")
            print("Model saved in file %s" % save_model_ckpt)


### MAIN ###
def main():

    # tf Graph input """ THE SOFTMAX REGRESSION """
    train_images_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    print("SHAPE TRAIN IMG = ", train_images_node.get_shape())

    train_labels_node = tf.placeholder(tf.int32, shape=(BATCH_SIZE, NUM_CLASSES))

    print("SHAPE TRAIN LAB = ", train_labels_node.get_shape())

    keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

    # Weight & bias
    weights = {
        'wc1': tf.get_variable("wc1", [11, 11, 3, 96], initializer=tf.random_normal_initializer(stddev=0.1)),
        'wc2': tf.get_variable("wc2", [5, 5, 48, 256], initializer=tf.random_normal_initializer(stddev=0.1)),      
        'wc3': tf.get_variable("wc3", [3, 3, 256, 384], initializer=tf.random_normal_initializer(stddev=0.1)),
        'wc4': tf.get_variable("wc4", [3, 3, 192, 384], initializer=tf.random_normal_initializer(stddev=0.1)),
        'wc5': tf.get_variable("wc5", [3, 3, 192, 296], initializer=tf.random_normal_initializer(stddev=0.1)),      
        'wfc1': tf.get_variable("wfc1", [2*2*296, 4096], initializer=tf.random_normal_initializer(stddev=0.1)),        
        'wfc2': tf.get_variable("wfc2", [4096, 4096], initializer=tf.random_normal_initializer(stddev=0.1)),        
        'out': tf.get_variable("out", [4096, NUM_CLASSES], initializer=tf.random_normal_initializer(stddev=0.1))
        
        # 'wc1': tf.Variable(tf.truncated_normal([NUM_CLASSES, NUM_CLASSES, NUM_CHANNELS, 64], stddev=0.1)),
        # 'wc2': tf.Variable(tf.truncated_normal([NUM_CLASSES, NUM_CLASSES, BATCH_SIZE, 128], stddev=0.1)),
        # 'wc3': tf.Variable(tf.truncated_normal([NUM_CLASSES, NUM_CLASSES, 128, 256], stddev=0.1)),
        # 'wfc1': tf.Variable(tf.truncated_normal([4*4*256, 1024], stddev=0.1)),
        # 'wfc2': tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.1)),
        # 'out': tf.Variable(tf.truncated_normal([1024, NUM_CLASSES], stddev=0.1))
    }
    
    biases = {
        'bc1': tf.Variable(tf.zeros([96])),
        'bc2': tf.Variable(tf.constant(0.1, shape=[256])),
        'bc3': tf.Variable(tf.zeros([384])),
        'bc4': tf.Variable(tf.constant(0.1, shape=[384])),
        'bc5': tf.Variable(tf.zeros([296])),
        'bfc1': tf.Variable(tf.zeros([4096])),
        'bfc2': tf.Variable(tf.zeros([4096])),
        'out': tf.Variable(tf.zeros([NUM_CLASSES]))
    }

    ### Call run_training_graph() function
    run_training_graph(train_images_node, train_labels_node, keep_prob, weights, biases)

    # Restore model from disk.
    saver.restore(sess, "/tmp/model.ckpt")
    print("Model restored")
    # Do some work with the model    

if __name__ == '__main__':
    main()
