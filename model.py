import csv
import cv2
import numpy as np

#path1 is default data from Udacity and path2 is data created by Author
#Images from all 3 cameras considered for training

#path1 = '../CarND-Behavioral-Cloning-P3/data/'
path2= '../CarND-Behavioral-Cloning-P3/myData/'

images = []
measurments = []

#path = path1
path = path2
lines = []
with open(path+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

for line in lines[:11293]:
    image = cv2.imread(path+'IMG/'+(line[0].split('/')[-1]))
    images.append(image)
    measurment = float(line[3])
    measurments.append(measurment)
    image = cv2.imread(path+'IMG/'+(line[1].split('/')[-1]))
    images.append(image)
    measurments.append(measurment)
    image = cv2.imread(path+'IMG/'+(line[2].split('/')[-1]))
    images.append(image)
    measurments.append(measurment)

X_train = np.array(images)
y_train = np.array(measurments)
print(X_train.shape)

#required keras component imported

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()

#input data normalisation
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

#Architecture from Nvidia is implemented
model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten()) 
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.1, shuffle=True, epochs=2)

model.save('model.h5')