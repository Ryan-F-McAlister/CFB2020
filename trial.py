# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:15:53 2020

@author: rynmc
"""

import tensorflow as tf
from tensorflow import keras

x1=5
x2=13
x=tf.constant([x1,x2], shape=(1,2))

model=keras.Sequential()
model.add(keras.layers.Input((1,2)))
model.add(keras.layers.Embedding(20, 5))
model.add(keras.layers.Reshape(target_shape=(1,-1)))
model.add(keras.layers.Dense(1))
y=model(x)
model.summary()
print(y)



