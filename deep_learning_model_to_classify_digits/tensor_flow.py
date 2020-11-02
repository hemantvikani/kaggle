# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#this model is used to identify digits using neural network


import pandas as pd

from tensorflow import keras



from tensorflow.keras import layers 

img_rows,img_cols = 28,28
num_classes = 10

def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label,num_classes)
    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape (num_images,img_rows,img_cols,1)
    out_x = x_shaped_array / 255
    return out_x,out_y
train_file = "train.csv"
raw_data = pd.read_csv(train_file)
x,y = data_prep(raw_data)
model = keras.Sequential()
model.add(layers.Conv2D(20,kernel_size=(3,3),activation = 'relu',input_shape=(img_rows,img_cols,1)))
model.add(layers.Conv2D(20, kernel_size=(3,3),activation= 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(10,activation = 'softmax'))
model.compile(loss = keras.losses.categorical_crossentropy,optimizer='adam',
              metrics = ['accuracy'])
model.fit(x,y,batch_size=128,epochs = 2,validation_split = 0.2)

test_file = "test.csv"
raw_data1 = pd.read_csv(test_file)
n_images = raw_data1.shape[0]
x_as_array1 = raw_data1.values[:,:]
x_shaped_array1 = x_as_array1.reshape (n_images,img_rows,img_cols,1)
test_x = x_shaped_array1 / 255

p = model.predict(test_x)


    