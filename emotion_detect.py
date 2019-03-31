# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 12:06:20 2019

@author: saloni
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Activation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam

from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import h5py

f=h5py.File('Images_data.hdf5','r')
a=list(f.keys())[0]
x_data=list(f[a])
x_data=np.asarray(x_data)

f2=h5py.File('Images_label.hdf5','r')
a=list(f2.keys())[0]
y_data=list(f2[a])
y_data=np.asarray(y_data)

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data, test_size=0.1, random_state=42)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(x_train)

activation = 'relu'
kernel_regularizer_mlp = l2(7e-4)
dropout_conv = 0.5 #Fraction of units to DROP, i.e. set to 0. for no dropout

model = Sequential()
model.add(Conv2D(60, (3, 3), padding='same', input_shape=(350,350,1))) #note that conv layers have no regularizer, but they do have dropout
model.add(BatchNormalization(axis=3)) #here axis has a different meaning from numpy axis
# Here, axis takes mean and std over every other axis, i.e. every individual element in axis=3 (i.e. features) is normalized
model.add(Activation(activation))
model.add(Conv2D(60, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout_conv))

model.add(Conv2D(125, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation(activation))
model.add(Conv2D(125, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout_conv))

model.add(Conv2D(250, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation(activation))
model.add(Conv2D(250, (3, 3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(500, activation=activation, kernel_regularizer=kernel_regularizer_mlp))
# model.add(Dropout(0.8))
model.add(Dense(100, activation='softmax', kernel_regularizer=kernel_regularizer_mlp))

# print(model.summary())
#plot_model(model, to_file='emotion_model.png', show_shapes=True, show_layer_names=False)
# SVG(model_to_dot(model).create(prog='dot', format='svg'))
optimizer = Adam()
loss = 'categorical_crossentropy'
metrics = ['accuracy']
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

batch_size = 256
epochs = 50
history = model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test))

score = model.evaluate(x_test,y_test, batch_size=batch_size)
print('Test accuracy = {0}'.format(100*score[1]))

model.save('model_emotion.h5')


