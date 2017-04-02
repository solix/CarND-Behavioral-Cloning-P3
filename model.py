import numpy as np
import pandas as pd
import csv
import cv2
import tensorflow as tf
from random import shuffle
from sklearn.model_selection import train_test_split
import preprocess


reader1 = pd.read_csv('./5laps/driving_log.csv', usecols=['center', 'left', 'right', 'steering'])




def remove_unwanted_data_with_bad_angels(dataset):
    #lets remove unwanted angels see what happens
    for data in dataset:
        print(data['steering'])
        angel = data['steering']
        if (angel > 0.80 or angel < 0.80):
            dataset.remove(data)
        elif np.math.isclose(angel, 0, abs_tol=0.001):
            dataset.remove(data)
    return dataset

#load csv file and append it to dataset
def load_dataset(file_path):
    dataset = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            try:
                dataset.append({'center': line[0], 'left': line[1], 'right': line[2], 'steering': float(line[3]),
                                'throttle': float(line[4]), 'brake': float(line[5]), 'speed': float(line[6])})
            except:
                continue  # some images throw error during loading
    return dataset




def plothistory (history_object):
### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def moving_average(a, n=3) :
    # from http://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
#augment data to balance the data
def load_and_augment_image(image):
        # select a value between 0 and 2 to swith between center, left and right image
    index = np.random.randint(3)


    if (index == 0):
        image_file = image['left'].strip()
        angle_offset = 2
    elif (index == 1):
        image_file = image['center'].strip()
        angle_offset = 0.
    elif (index == 2):
        image_file = image['right'].strip()
        angle_offset = - 2

    steering_angle = image['steering'] + angle_offset
    # print(image_file)
    image = cv2.imread('./5laps/'+image_file)
    image, steering_angle = preprocess.random_transform(image, steering_angle)
    return image, steering_angle

def generator_batch(dataset,batch_size=32):
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, 160, 320, 3))
    print("batch_feature shape is {}".format(batch_features.shape))
    batch_labels = np.zeros((batch_size, 1))
    while True:
        for i in range(batch_size):
            # select a random image from the dataset
            image_index = np.random.randint(len(dataset))
            image_data = dataset[image_index]
            feature, label = load_and_augment_image(image_data)
            # choose random index in features
            batch_features[i] = feature
            batch_labels[i] = label
        yield batch_features, batch_labels
    yield batch_images,batch_labels


#load data and split to train and validation
fila_path = './5laps/driving_log.csv'
dataset = load_dataset(fila_path)
dataset = remove_unwanted_data_with_bad_angels(dataset)
print("Loaded {} samples from file {}".format(len(dataset), fila_path))
print("\nPartitioning the dataset ...")
shuffle(dataset)
X_train, X_validation = train_test_split(dataset, test_size=0.2)
print("X_train has {} elements.".format(len(X_train)))
print("X_validation has {} elements.".format(len(X_validation)))
print("Partitioning the dataset complete.")

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




def main(_):
    # inspired from Nvidia
    print('Build model...')
    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160, 320, 3)))
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
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(100))
    # model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(10))
    # model.add(Dropout(0.75))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam(lr=FLAGS.learning_rate))
    print("Model summary:\n", model.summary())

    training_gen = generator_batch(X_train)
    val_gen = generator_batch(X_validation)
    for i in range(FLAGS.epochs):
        # model.fit([], [], validation_split=0.3, shuffle=True, nb_epoch=i, batch_size=FLAGS.batch_size,verbose = 1)
        model.fit_generator(training_gen,samples_per_epoch=1000,nb_epoch=1,validation_data=val_gen, nb_val_samples=len(X_validation))
        model_no = 'model_T'+str(i)+'.h5'
        model.save(model_no)
        print("Model is saves as {}".format(model_no))


if __name__ == '__main__':
    tf.app.run()