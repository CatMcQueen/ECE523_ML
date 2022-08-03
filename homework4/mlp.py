import numpy as np
import pandas as pd
from keras.models import Sequential 
from keras.layers import Dense
import keras
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

num_classes = 10 

# reshape data to be 1D
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test[2])

# get it to 0 - 255
x_train = x_train.astype('float32')/255.0
x_test  = x_test.astype('float32')/255.0

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test, num_classes)

input_shape = (x_train.shape[1] * x_train.shape[2])

model = Sequential()
model.add(Dense(1000, input_dim=input_shape, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.comile(loss='categorical_crossentropy', optimizer='adam', metrics=['loss', 'accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=300, verbose=1, validation_split=0.2)

test_results = model.evaluate(x_test, y_test, verbose=1)
print('Test results -- Loss: {} - Accuracy: {}'.format(test_results[0], test_results[1]))


