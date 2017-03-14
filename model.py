import numpy as np
import pandas as pd
import cv2

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

X_train = np.array(imgs)
y_train = np.array(labels)


from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape = (160,320,3)))
model.add(Dense(1))

model.compile(loss = 'mse' , optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True , nb_epoch = 7 )
model.save('model.h5')
