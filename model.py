import numpy as np
import pandas as pd
import cv2
import tensorflow as tf



reader = pd.read_csv('./data/driving_log.csv', usecols=['center', 'left', 'right', 'steering'])

type(reader)

imgs = []
labels = []
for index, row in reader.iterrows():
    # print(row['center'], row['steering'])
    file_path = './data/' + row['center']
    img = cv2.imread(file_path)
    steering = float(row['steering'])
    imgs.append(img)
    labels.append(steering)

augmented_images , augmented_measurements = [],[]

for image , mesure in zip(imgs,labels):
    augmented_images.append(image)
    augmented_measurements.append(mesure)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(mesure*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)



#Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Dropout,Cropping2D
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D,MaxPooling2D

#define flags for epoch and batchsize
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 10, "The number of epochs.")
flags.DEFINE_integer('batch_size', 60, "The batch size.")
flags.DEFINE_float('learning_rate', 0.0001, "The batch size.")


def main(_):

    model = Sequential()
    model.add(Lambda(lambda x:x/255.0 - 0.5,input_shape = (160,320,3)))
    model.add(Cropping2D((50,20),(0,0)))
    model.add(Convolution2D(6,3,3,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(240))
    model.add(Dense(120))
    model.add(Dense(1))

    model.compile(loss = 'mse' , optimizer = Adam(lr=FLAGS.learning_rate))
    model.fit(X_train, y_train, validation_split = 0.2, shuffle = True , nb_epoch = FLAGS.epochs , batch_size=FLAGS.batch_size )
    model.save('model.h5')


if __name__ == '__main__':
    tf.app.run()