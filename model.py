import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


reader = pd.read_csv('./data/driving_log.csv', usecols=['center', 'left', 'right', 'steering'])
reader2 = pd.read_csv('./my_data_2/driving_log.csv', usecols=['center', 'left', 'right', 'steering'])
imgs = []
labels = []
for  index, row in reader.iterrows():
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
    labels.append(steering + 0.2)
    labels.append(steering - 0.2)

# for  index, row in reader2.iterrows():
#     for i in range(3):
#         source =  row[i]
#         token = source.split('/')
#         local_path = './my_data_2/IMG/'
#         file_path = token[-1]
#         local_path = local_path+file_path
#         img = cv2.imread(local_path)
#         imgs.append(img)
#     steering = float(row['steering'])
#     labels.append(steering)
#     labels.append(steering + 0.2)
#     labels.append(steering - 0.2)

X_train = np.array(imgs)
y_train = np.array(labels)


augmented_imgs = []
augmented_steerings= []

for  img, msr in zip(imgs,labels):
    augmented_imgs.append(img)
    augmented_steerings.append(msr)
    augmented_imgs.append(cv2.flip(img,1))
    augmented_steerings.append(msr*-1.0)
# X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.33)

print(len(X_train), 'number of training data features')
print(len(y_train), 'number of training labeles')

datagen = ImageDataGenerator(
    )

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
    # inspired from Nvidia
    print('Build model...')
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,20), (0, 0))))  # also supports shape inference using `-1` as dimension
    model.add(GaussianNoise(sigma=0.5))
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
    # datagen.fit(X_train)
    # history = model.fit_generator(datagen.flow(X_train,y_train,batch_size=FLAGS.batch_size),samples_per_epoch=len(X_train),nb_epoch=FLAGS.epochs,validation_data=(X_valid,y_valid),verbose=1)
    # plothistory(history)
    model.save('model.h5')
    print("Model is saves as model.h5")


if __name__ == '__main__':
    tf.app.run()
