import h5py
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import pickle

np.random.seed(1)
tf.random.set_seed(1)

import PIL, PIL.Image

WORK_PATH = './gen-task/'

# Idea: tower architecture on basis letters to get autoencoding,
# feed through dense layers to retrieve all 26 letters (caps)

"""
Model definition: tower network
"""

# Input
X = keras.layers.Input(shape=(4, 64, 64), name='input')

# Lambda layers pull out the 4 basis characters
x1 = keras.layers.Lambda(lambda x: x[:, 0, :, :], output_shape=(64, 64), name='x1')(X)
x2 = keras.layers.Lambda(lambda x: x[:, 1, :, :], output_shape=(64, 64), name='x2')(X)
x3 = keras.layers.Lambda(lambda x: x[:, 2, :, :], output_shape=(64, 64), name='x3')(X)
x4 = keras.layers.Lambda(lambda x: x[:, 3, :, :], output_shape=(64, 64), name='x4')(X)

# Flatten the images into 64 * 64 dimensional vectors
x1 = keras.layers.Flatten()(x1)
x2 = keras.layers.Flatten()(x2)
x3 = keras.layers.Flatten()(x3)
x4 = keras.layers.Flatten()(x4)

# The towers consist of a fully connected layer
x1 = keras.layers.Dense(16, activation='relu')(x1)
x2 = keras.layers.Dense(16, activation='relu')(x2)
x3 = keras.layers.Dense(16, activation='relu')(x3)
x4 = keras.layers.Dense(16, activation='relu')(x4)

# Concatenates the towers together and feed through fully connected layers
added = keras.layers.Concatenate()([x1, x2, x3, x4])
fc = keras.layers.Dense(200, activation='relu')(added)
fc = keras.layers.Dense(200, activation='relu')(fc)
fc = keras.layers.Dense(26 * 64 * 64, activation='relu')(fc)

# Reshape for 2D convolution and upsample
fc = keras.layers.Reshape((26, 64, 64))(fc) 
fc = keras.layers.Conv2DTranspose(26, data_format='channels_first', kernel_size=(4, 4), padding='same', activation='relu')(fc)
fc = keras.layers.Conv2DTranspose(26, data_format='channels_first', kernel_size=(4, 4), padding='same', activation='relu')(fc)

# out = keras.layers.Reshape((26, 64, 64))(fc)
out = fc

model = keras.models.Model(inputs=X, outputs=out)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[])
model.summary()

FILENAME = 'fonts-50'

def dump_history(history):
    with open('{}/history.pickle'.format(WORK_PATH), 'wb') as f:
        pickle.dump(history, f)
        print('Dumped history.')

# Open and prepare training set
train = h5py.File('./gen-task-dsets/gen-task-{}-train.hdf5'.format(FILENAME), 'r')
outputs = train['outputs'][:]
basis = train['basis'][:]

print('Training model on {} fonts...'.format(train['basis'].shape[0]))
history = model.fit(x=basis, y=outputs, epochs=50) # See Keras docs for the history object
dump_history(history.history)

def display_picture(arr, idx):
    img = PIL.Image.fromarray(np.hstack([arr[idx] for idx in range(26)]))
    if img.mode != 'L':
        img = img.convert('L')
    img.show()
    img.save('ex{}.png'.format(idx))

# Open and prepare test set
test = h5py.File('./gen-task-dsets/gen-task-{}-test.hdf5'.format(FILENAME), 'r')
outputs = test['outputs'][:]
basis = test['basis'][:]

print('Testing model on {} fonts...'.format(test['basis'].shape[0]))
loss = model.evaluate(x=basis, y=outputs)

# View classified examples from the test set
predictions = model.predict(basis)
display_picture(predictions[0], 0)