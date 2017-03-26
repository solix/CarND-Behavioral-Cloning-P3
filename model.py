import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.utils import shuffle


reader = pd.read_csv('./data/driving_log.csv', usecols=['center', 'left', 'right', 'steering'])
drive_reader = pd.read_csv('./track1/drive/driving_log.csv', usecols=['center', 'left', 'right', 'steering'])
recovery_reader = pd.read_csv('./track1/recovery/driving_log.csv', usecols=['center', 'left', 'right', 'steering'])
imgs = []
labels = []
for  index, row in reader.iterrows():
    print(row['center'], row['steering'])
    for i in range(3):
       source =  row[i]
       token = source.split('/')
       local_path = './data/IMG/'
       file_path = token[-1]
       local_path = local_path+file_path
       img = cv2.imread(local_path)
       imgs.append(img)
    steering = float(row['steering'])
    labels.append(steering)
    labels.append(steering + 0.25)
    labels.append(steering - 0.3)

for index, row in drive_reader.iterrows():
    # print(row['center'], row['steering'])
    for i in range(3):
        source =  row[i]
        token = source.split('/')
        local_path = './track1/drive/IMG/'
        file_path = token[-1]
        local_path = local_path+file_path
        img = cv2.imread(local_path)
        imgs.append(img)
    steering = float(row['steering'])
    labels.append(steering)
    labels.append(steering + 0.25)
    labels.append(steering - 0.3)

for index, row in recovery_reader.iterrows():
    # print(row['center'], row['steering'])
    source =  row["center"]
    token = source.split('/')
    local_path = './track1/recovery/IMG/'
    file_path = token[-1]
    local_path = local_path+file_path
    img = cv2.imread(local_path)
    imgs.append(img)
    steering = float(row['steering'])
    labels.append(steering)


X_train = np.array(imgs)
y_train = np.array(labels)
  
print(len(X_train), 'number of training data features')
print(len(y_train), 'number of labeled data')


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

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 11, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")
flags.DEFINE_float('learning_rate', 0.0001, "The batch size.")


def main(_):
    # inspired from Nvidia
    print('Build model...')
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((100,20), (0, 0))))  # also supports shape inference using `-1` as dimension
    model.add(GaussianNoise(sigma=0.2))
    model.add(Convolution2D(3, 5, 5, subsample=(2, 2),W_regularizer=l2(.001 )))
    model.add(ELU())
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), W_regularizer= l2(.001)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2),W_regularizer=l2(.001)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, W_regularizer=l2(.001)))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(100,W_regularizer=l2(.001)))
    model.add(ELU())
    model.add(Dense(10,W_regularizer=l2(.001)))
    model.add(ELU())
    model.add(Dense(1,W_regularizer=l2(.001)))

    model.compile(loss='mse', optimizer=Adam(lr=FLAGS.learning_rate))
    print("Model summary:\n", model.summary())

    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=FLAGS.epochs, batch_size=FLAGS.batch_size,verbose = 1)
    model.save('model.h5')
    print("Model is saves as model.h5")


if __name__ == '__main__':
    tf.app.run()
