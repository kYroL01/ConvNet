import Dataset
import os
import sys
import math
import timeit
import argparse
import tensorflow as tf
import numpy as np
import logging as log
import matplotlib.pyplot as plt

from sklearn import metrics

from Dataset import IMG_SIZE, LABELS_DICT

TRAIN_IMAGE_DIR = os.getcwd() + '/dataset'
TEST_IMAGE_DIR = os.getcwd() + '/test_dataset'
CKPT_DIR = 'ckpt_dir'
MODEL_CKPT = 'ckpt_dir/model.ckpt'

### Parameters for Logistic Regression ###
BATCH_SIZE = 64

### Network Parameters ###
n_input = IMG_SIZE**2
n_classes = 4
n_channels = 3
input_dropout = 0.8
hidden_dropout = 0.5
std_dev = 0.1 #math.sqrt(2/n_input) # http://cs231n.github.io/neural-networks-2/#init


class AlexNetModel(tf.keras.Model):
    """AlexNet CNN model using Keras API"""

    def __init__(self, n_classes=4, input_dropout=0.8, hidden_dropout=0.5):
        super(AlexNetModel, self).__init__()

        # Convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(BATCH_SIZE, (11, 11), strides=4, padding='same',
                                            activation='relu', name='conv1',
                                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=std_dev))
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='pool1')
        self.norm1 = tf.keras.layers.BatchNormalization(name='norm1')
        self.dropout1 = tf.keras.layers.Dropout(1 - input_dropout, name='dropout1')

        self.conv2 = tf.keras.layers.Conv2D(BATCH_SIZE*2, (5, 5), strides=1, padding='same',
                                            activation='relu', name='conv2',
                                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=std_dev))
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='pool2')
        self.norm2 = tf.keras.layers.BatchNormalization(name='norm2')

        self.conv3 = tf.keras.layers.Conv2D(BATCH_SIZE*4, (3, 3), strides=1, padding='same',
                                            activation='relu', name='conv3',
                                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=std_dev))
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='pool3')
        self.norm3 = tf.keras.layers.BatchNormalization(name='norm3')
        self.dropout3 = tf.keras.layers.Dropout(1 - hidden_dropout, name='dropout3')

        self.conv4 = tf.keras.layers.Conv2D(BATCH_SIZE*4, (3, 3), strides=1, padding='same',
                                            activation='relu', name='conv4',
                                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=std_dev))
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='pool4')
        self.norm4 = tf.keras.layers.BatchNormalization(name='norm4')
        self.dropout4 = tf.keras.layers.Dropout(1 - hidden_dropout, name='dropout4')

        self.conv5 = tf.keras.layers.Conv2D(256, (3, 3), strides=1, padding='same',
                                            activation='relu', name='conv5',
                                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=std_dev))
        self.pool5 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='pool5')

        # Flatten layer
        self.flatten = tf.keras.layers.Flatten()

        # Fully connected layers
        self.fc1 = tf.keras.layers.Dense(4096, activation='relu', name='fc1',
                                         kernel_initializer=tf.keras.initializers.RandomNormal(stddev=std_dev))

        self.fc2 = tf.keras.layers.Dense(2*2*256, activation='relu', name='fc2',
                                         kernel_initializer=tf.keras.initializers.RandomNormal(stddev=std_dev))
        self.dropout7 = tf.keras.layers.Dropout(1 - hidden_dropout, name='dropout7')

        # Output layer
        self.out = tf.keras.layers.Dense(n_classes, name='output',
                                         kernel_initializer=tf.keras.initializers.RandomNormal(stddev=std_dev))

    def call(self, inputs, training=False):
        """Forward pass of the model"""
        # Reshape input to image format
        x = tf.reshape(inputs, shape=[-1, IMG_SIZE, IMG_SIZE, 3])

        # Conv block 1
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.norm1(x, training=training)
        x = self.dropout1(x, training=training)

        # Conv block 2
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.norm2(x, training=training)

        # Conv block 3
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.norm3(x, training=training)
        x = self.dropout3(x, training=training)

        # Conv block 4
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.norm4(x, training=training)
        x = self.dropout4(x, training=training)

        # Conv block 5
        x = self.conv5(x)
        x = self.pool5(x)

        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout7(x, training=training)

        # Output layer (logits)
        x = self.out(x)

        return x


class ConvNet(object):

    ## Constructor to build the model for the training ##
    def __init__(self, **kwargs):

        params = set(['learning_rate','max_epochs','display_step','dataset_training','dataset_test'])

        # initialize all allowed keys to false
        self.__dict__.update((key, False) for key in params)
        # and update the given keys by their given values
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in params)

        if(self.dataset_training != False):
            self.train_imgs_lab = Dataset.loadDataset(self.dataset_training)
        else:
            self.test_imgs_lab = Dataset.loadDataset(self.dataset_test)

        # Create the Keras model
        self.model = AlexNetModel(n_classes=n_classes, input_dropout=input_dropout,
                                  hidden_dropout=hidden_dropout)


    # Batch function for Training - give the next batch of images and labels
    def BatchIteratorTraining(self, batch_size):
        imgs = []
        labels = []

        for img, label in self.train_imgs_lab:
            imgs.append(img)
            labels.append(label)
            if len(imgs) == batch_size:
                yield np.array(imgs), np.array(labels)
                imgs = []
                labels = []
        if len(imgs) > 0:
            yield np.array(imgs), np.array(labels)


    # Batch function for Testing - give the next batch of images and labels
    def BatchIteratorTesting(self, batch_size):
        imgs = []
        labels = []

        for img, label in self.test_imgs_lab:
            imgs.append(img)
            labels.append(label)
            if len(imgs) == batch_size:
                yield np.array(imgs), np.array(labels)
                imgs = []
                labels = []
        if len(imgs) > 0:
            yield np.array(imgs), np.array(labels)


    # Method for training the model and testing its accuracy
    def training(self):

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=0.1),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(CKPT_DIR):
            os.makedirs(CKPT_DIR)

        # Setup callbacks
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=CKPT_DIR, histogram_freq=1)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_CKPT,
            save_weights_only=True,
            save_best_only=False,
            verbose=1
        )

        # Collect validation data (first 6 batches)
        validation_imgs_batch = [b for i, b in enumerate(self.BatchIteratorTraining(BATCH_SIZE)) if i < 6]
        val_imgs = np.concatenate([batch[0] for batch in validation_imgs_batch], axis=0)
        val_labels = np.concatenate([batch[1] for batch in validation_imgs_batch], axis=0)

        # Training loop
        for epoch in range(self.max_epochs):
            print("epoch = %d" % epoch)
            log.info("Epoch %s" % epoch)
            self.train_imgs_lab = Dataset.loadDataset(self.dataset_training)  # necessary 'cause of the yield

            # Loop over all batches
            for step, (batch_imgs_train, batch_labels_train) in enumerate(self.BatchIteratorTraining(BATCH_SIZE)):
                print("step = %d" % step)

                # Train on batch
                history = self.model.fit(
                    batch_imgs_train,
                    batch_labels_train,
                    batch_size=len(batch_imgs_train),
                    epochs=1,
                    verbose=0,
                    callbacks=[tensorboard_callback] if step == 0 else []
                )

                train_acc = history.history['accuracy'][0]
                train_loss = history.history['loss'][0]

                if step % self.display_step == 0:
                    log.info("Training Accuracy = " + "{:.5f}".format(train_acc))
                    log.info("Training Loss = " + "{:.6f}".format(train_loss))

        print("Optimization Finished!")

        # Save the model weights
        self.model.save_weights(MODEL_CKPT)
        print("Model saved in file %s" % MODEL_CKPT)

        ### Metrics ###
        target_names = ['class 0', 'class 1', 'class 2', 'class 3']
        list_pred_total = []
        list_true_total = []

        # Accuracy Precision Recall F1-score by VALIDATION IMAGES
        for step, (batch_imgs_valid, batch_labels_valid) in enumerate(validation_imgs_batch):

            # Get predictions
            predictions = self.model.predict(batch_imgs_valid, verbose=0)
            y_pred = np.argmax(predictions, axis=1)

            # Calculate accuracy
            y_true = np.argmax(batch_labels_valid, axis=1)
            valid_acc = np.mean(y_pred == y_true)

            log.info("Validation accuracy = " + "{:.5f}".format(valid_acc))
            list_pred_total.extend(y_pred)
            list_true_total.extend(y_true)

        # Classification Report (PRECISION - RECALL - F1 SCORE)
        log.info("\n")
        log.info(metrics.classification_report(list_true_total, list_pred_total, target_names=target_names))

        # Network Input Values
        log.info("Learning Rate " + "{:.4f}".format(self.learning_rate))
        log.info("Number of epochs " + "{:d}".format(self.max_epochs))

        print(metrics.classification_report(list_true_total, list_pred_total, target_names=target_names))

        # ROC curve
        if len(list_true_total) > 0 and len(list_pred_total) > 0:
            fpr, tpr, _ = metrics.roc_curve(list_true_total, list_pred_total)

            plt.figure()
            plt.plot(fpr, tpr, label='ROC curve')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Recognition ROC curve')
            plt.legend(loc="lower right")
            plt.show()


    def prediction(self):

        # Load model weights
        if os.path.exists(MODEL_CKPT + '.index'):
            self.model.load_weights(MODEL_CKPT)
            print("Model restored")
        else:
            print("No model checkpoint found to restore - ERROR")
            return

        ### Metrics ###
        target_names = ['class 0', 'class 1', 'class 2', 'class 3']
        list_pred_total = []
        list_true_total = []

        # Accuracy Precision Recall F1-score by TEST IMAGES
        for step, (batch_imgs_test, batch_labels_test) in enumerate(self.BatchIteratorTesting(BATCH_SIZE)):

            # Get predictions
            predictions = self.model.predict(batch_imgs_test, verbose=0)
            y_pred = np.argmax(predictions, axis=1)

            print("batch predict = %d" % len(y_pred))
            list_pred_total.extend(y_pred)

            y_true = np.argmax(batch_labels_test, axis=1)
            print("batch real = %d" % len(y_true))
            list_true_total.extend(y_true)

        # Classification Report (PRECISION - RECALL - F1 SCORE)
        log.info('\n')
        log.info(metrics.classification_report(list_true_total, list_pred_total, target_names=target_names))

        # Network Input Values
        log.info("Learning Rate " + "{:.4f}".format(self.learning_rate if self.learning_rate else 0.0))
        log.info("Number of epochs " + "{:d}".format(self.max_epochs if self.max_epochs else 0))

        print(metrics.classification_report(list_true_total, list_pred_total, target_names=target_names))




### MAIN ###
def main():

    np.random.seed(7)

    parser = argparse.ArgumentParser(description='A convolutional neural network for image recognition')
    subparsers = parser.add_subparsers()

    training_args = [
        (['-lr', '--learning-rate'], {'help':'learning rate', 'type':float, 'default':0.001}),
        (['-e', '--max_epochs'], {'help':'max epochs', 'type':int, 'default':100}),
        (['-ds', '--display-step'], {'help':'display step', 'type':int, 'default':10}),
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
            conv_net = ConvNet(learning_rate=args.learning_rate, max_epochs=args.max_epochs,
                               display_step=args.display_step, dataset_training=args.dataset_training)
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
