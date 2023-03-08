from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import numpy as np
import cv2

LABELS = ["Burger", "Dimsun", "Ramen", "Sushi"]


#Read data from file (download at https://github.com/neokarn/computer_vision/blob/master/data.csv)
N = 30
x_train = np.zeros((N,50,50,1),'float')
y_train = np.zeros((N),'float')
count = 0
for class_id in LABELS:
    for im_id in range(1,11):
        im = cv2.imread("thainum123/train/"+str(class_id)+"/"+str(im_id)+".bmp",cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im,(50,50))
        x_train[count,:,:,0] = im/255.
        y_train[count] = class_id-1
        count += 1

#Create model by using sequential structure
model = Sequential()
model.add(Dense(5, input_dim=5, activation='tanh'))
model.add(Dense(5, activation='tanh'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()



#Train Model
x_train = data[0:100,0:5]
y_train = data[0:100,5]
y_train = to_categorical(y_train)

x_val = data[100:120,0:5]
y_val = data[100:120,5]
y_val = to_categorical(y_val)

#save model
checkpoint = ModelCheckpoint('my_model.h5',
                             verbose=1,
                             monitor='val_accuracy',
                             mode='max',
                             save_best_only = True)

h = model.fit(x_train, y_train,
              epochs=200, batch_size=5,
              validation_data=(x_val,y_val),
              callbacks=[checkpoint])

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.legend(['train', 'val'])


plt.show()