import h5py
import tensorflow.keras as keras
import numpy as np


# Idea: Few shot inference of fonts from a couple of examples
# How to build tower network in keras? Soln: add

input1 = keras.layers.Input(shape=(64, 64, 8))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(64, 64, 8))
x2 = keras.layers.Dense(8, activation='relu')(input2)
input3 = keras.layers.Input(shape=(64, 64, 8))
x3 = keras.layers.Dense(8, activation='relu')(input3)
input4 = keras.layers.Input(shape=(64, 64, 8))
x4 = keras.layers.Dense(8, activation='relu')(input4)
input5 = keras.layers.Input(shape=(64, 64, 8))
x5 = keras.layers.Dense(8, activation='relu')(input5)

added = keras.layers.Add()([x1, x2, x3, x4, x5])
fc1 = keras.layers.Dense(10, activation='relu')(added)
fc2 = keras.layers.Dense(10, activation='relu')(fc1)
fc3 = keras.layers.Dense(10, activation='relu')(fc2)
out = keras.layers.Dense(1, activation='sigmoid')(fc3)

model = keras.models.Model(inputs=[input1, input2, input3, input4, input5], outputs=out)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# keras.utils.plot_model(model, to_file='model.png')

# Load data and create examples (A, B, C, D, E) where E may be or may be a part
# of the font.