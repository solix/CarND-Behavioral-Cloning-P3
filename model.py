import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.utils import shuffle
from keras import backend as K





reader = pd.read_csv('./data/driving_log.csv', usecols=['center', 'left', 'right', 'steering'])

type(reader)

imgs = []
labels = []
rows = ['center', 'left', 'right']
for index, row in reader.iterrows():
    # print(row['center'], row['steering'])
    for i in range(3):
	file_path = './data/' + row[i]
    	img = cv2.imread(file_path)
        imgs.append(img)
    steering = float(row['steering'])
    labels.append(steering)
    labels.append(steering*0.1)
    labels.append(steering*-0.1)
augmented_images , augmented_measurements = [],[]

for image , mesure in zip(imgs,labels):
    augmented_images.append(image)
    augmented_measurements.append(mesure)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(mesure*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


print(len(X_train), 'number of training data features')
print(len(y_train), 'number of labeled data')

#Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Dropout,Cropping2D, Reshape,Activation
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D,MaxPooling2D,ZeroPadding2D

#define flags for epoch and batchsize
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 10, "The number of epochs.")
flags.DEFINE_integer('batch_size', 60, "The batch size.")
flags.DEFINE_float('learning_rate', 0.0001, "The batch size.")


def main(_):
     #inspired from Nvidia
    print('Build model...')
    model = Sequential()
    model.add(Lambda(lambda x:x/255.0 - 0.5,input_shape = (66,200,3)))
    model.add(Cropping2D(cropping=((50,20),(0,0))))  # also supports shape inference using `-1` as dimension
    print(model.output) #shape=(?, 66, 200, 3)
    model.add(Convolution2D(3,5,5,sub_sample=(2,2),activation='relu'))
    model.add(Dropout(0.25))
    model.add(Convolution2D(24, 5, 5, sub_sample=(2,2), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Convolution2D(36, 5, 5,sub_sample = (2,2), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Convolution2D(48, 3, 3, activation='relu' ,border_mode='same'))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))



    model.compile(loss = 'mse' , optimizer = Adam(lr=FLAGS.learning_rate) )
    print("Model summary:\n", model.summary())


    model.fit(X_train, y_train, validation_split = 0.2, shuffle = True , nb_epoch = FLAGS.epochs , batch_size=FLAGS.batch_size )
    model.save('model.h5')
    print("Model is saves as model.h5")


if __name__ == '__main__':
    tf.app.run()
