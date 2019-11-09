import h5py
import tensorflow.keras as keras
import numpy as np


# Idea: Few shot inference of fonts from a couple of examples
# How to build tower network in keras? Soln: add

input1 = keras.layers.Input(shape=(64, 64), name='input1')
i1f = keras.layers.Flatten()(input1)
x1 = keras.layers.Dense(8, activation='relu')(i1f)
input2 = keras.layers.Input(shape=(64, 64), name='input2')
i2f = keras.layers.Flatten()(input2)
x2 = keras.layers.Dense(8, activation='relu')(i2f)
input3 = keras.layers.Input(shape=(64, 64), name='input3')
i3f = keras.layers.Flatten()(input3)
x3 = keras.layers.Dense(8, activation='relu')(i3f)
input4 = keras.layers.Input(shape=(64, 64), name='input4')
i4f = keras.layers.Flatten()(input4)
x4 = keras.layers.Dense(8, activation='relu')(i4f)
input5 = keras.layers.Input(shape=(64, 64), name='input5')
i5f = keras.layers.Flatten()(input5)
x5 = keras.layers.Dense(8, activation='relu')(i5f)

added = keras.layers.Add()([x1, x2, x3, x4, x5])
fc1 = keras.layers.Dense(10, activation='relu')(added)
fc2 = keras.layers.Dense(10, activation='relu')(fc1)
fc3 = keras.layers.Dense(10, activation='relu')(fc2)
out = keras.layers.Dense(1, activation='sigmoid')(fc3)

model = keras.models.Model(inputs=[input1, input2, input3, input4, input5], outputs=out)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Debug
# keras.utils.plot_model(model, to_file='model.png')

f = h5py.File('fonts-25-discrim-task.hdf5', 'r')
labels = f['labels'][:].reshape(2200, 1)
examples = f['examples']

# This is kind of expensive; is there a better way to structure the data to prevent slicing like this?
print('Extracting data...')
examples1 = examples[:, 0, :, :]
examples2 = examples[:, 0, :, :]
examples3 = examples[:, 0, :, :]
examples4 = examples[:, 0, :, :]
examples5 = examples[:, 0, :, :]

print('Training model...')
model.fit(x={'input1': examples1, 'input2': examples2, 'input3': examples3, 'input4': examples4, 'input5': examples5}, y=labels, shuffle=False, epochs=100)

f = h5py.File('fonts-25-test-discrim-task.hdf5', 'r')
labels = f['labels'][:].reshape(2200, 1)
examples = f['examples']

# This is kind of expensive; is there a better way to structure the data to prevent slicing like this?
print('Extracting test data...')
examples1 = examples[:, 0, :, :]
examples2 = examples[:, 0, :, :]
examples3 = examples[:, 0, :, :]
examples4 = examples[:, 0, :, :]
examples5 = examples[:, 0, :, :]

model.evaluate(x={'input1': examples1, 'input2': examples2, 'input3': examples3, 'input4': examples4, 'input5': examples5}, y=labels)
# model.predict()

# Load data and create examples (A, B, C, D, E) where E may be or may be a part
# of the font.