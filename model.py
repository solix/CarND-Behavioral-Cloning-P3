import random

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

reader1 = pd.read_csv('./data/driving_log.csv', usecols=['center', 'left', 'right', 'steering'])
reader2 = pd.read_csv('./final_d/recover_track/driving_log.csv', usecols=['center', 'left', 'right', 'steering'])
reader3 = pd.read_csv('./valid_track/driving_log.csv', usecols=['center', 'left', 'right', 'steering'])



def loadRecoveryData():
    #loading data
    imgs = []
    labels = []
    for  index, row in reader2.iterrows():
        for i in range(3):
            source =  row['center']
            token = source.split('/')
            local_path = './final_d/recover_track/IMG/'
            file_path = token[-1]
            local_path = local_path+file_path
            img = cv2.imread(local_path)
            imgs.append(img)
        steering = float(row['steering'])
        labels.append(steering)
        labels.append(steering + 0.25)
        labels.append(steering - 0.25)
    for  index, row in reader1.iterrows():
        for i in range(3):
            source =  row['center']
            token = source.split('/')
            local_path = './data/IMG/'
            file_path = token[-1]
            local_path = local_path+file_path
            img = cv2.imread(local_path)
            imgs.append(img)
        steering = float(row['steering'])
        labels.append(steering)
        labels.append(steering + 0.25)
        labels.append(steering - 0.25)


    return np.array(imgs),np.array(labels)


def loadValidData():
    #loading data given
    imgs = []
    labels = []
    for  index, row in reader3.iterrows():
        for i in range(3):
            source =  row['center']
            token = source.split('/')
            local_path = './final_d/2track/IMG/'
            file_path = token[-1]
            local_path = local_path+file_path
            img = cv2.imread(local_path)
            imgs.append(img)
        steering = float(row['steering'])
        labels.append(steering)
        labels.append(steering + 0.25)
        labels.append(steering - 0.25)
    return np.array(imgs),np.array(labels)


# Model is inspired by nvidia cnn model with a different tweaks
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D, Reshape, Activation
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
# define flags for epoch and batchsize
from keras import backend as K
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
import matplotlib.pyplot as plt


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 11, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")
flags.DEFINE_float('learning_rate', 0.0001, "The batch size.")


# def train_generator(features, labels, batch_size=32):
#
#  # Create empty arrays to contain batch of features and labels#
#
#     batch_features = np.zeros((batch_size,160,320,3))
#     print("batch_feature shape is {}".format(batch_features.shape))
#     batch_labels = np.zeros((batch_size))
#     datagen = ImageDataGenerator(
#          rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         horizontal_flip=True)
#
#     while True:
#         for i in range(batch_size):
#             #choose random index in features
#             index= random.choice(len(features),1)
#             batch_features[i] = features[index]
#             batch_labels[i] = labels[index]
#             datagen.fit(batch_features[i])
#
#         yield datagen


# def train_generator(features, labels, batch_size=FLAGS.batch_size):
#     # Create empty arrays to contain batch of features and labels#
#
#     batch_features = np.zeros((batch_size, 160, 320, 3))
#     print("batch_feature shape is {}".format(batch_features.shape))
#     batch_labels = np.zeros((batch_size))
#     datagen = ImageDataGenerator(
#         rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         horizontal_flip=True)
#
#     while True:
#         for i in range(batch_size):
#             # choose random index in features
#             index = random.choice(len(features), 1)
#             batch_features[i] = features[index]
#             batch_labels[i] = labels[index]
#             datagen.fit(batch_features[i])
#
#     yield datagen

def plothistory (history_object):

### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def main(_):
    X_valid, y_valid = loadValidData()
    X_train, y_train = loadRecoveryData()
    # inspired from Nvidia
    print('Build model...')
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,20), (0, 0))))  # also supports shape inference using `-1` as dimension
    model.add(GaussianNoise(sigma=0.05))
    model.add(Convolution2D(3, 5, 5,border_mode='valid', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(24, 5, 5, border_mode='valid',subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, border_mode='valid',subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3,border_mode='valid'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3,border_mode='valid', ))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam(lr=FLAGS.learning_rate))
    print("Model summary:\n", model.summary())
    datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
    datagen.fit(X_train,augment=True)
    for i in range(1,FLAGS.epochs):
        # model.fit(X_train, y_train, validation_split=0.3, shuffle=True, nb_epoch=i, batch_size=FLAGS.batch_size,verbose = 1)
        # fits the model on batches with real-time data augmentation:
        model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                            samples_per_epoch=len(X_train)//32, nb_epoch=i)
        model_no = 'model_M'+str(i)+'.h5'
        model.save(model_no)
        print("Model is saves as {}".format(model_no))


if __name__ == '__main__':
    tf.app.run()