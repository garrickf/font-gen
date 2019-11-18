import h5py
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

np.random.seed(1)
tf.random.set_seed(1)

import PIL, PIL.Image

# Idea: Few shot inference of fonts from a couple of examples
# How to build tower network in keras? Soln: add

"""
Model definition: tower network
"""

# Input
X = keras.layers.Input(shape=(5, 64, 64), name='input')

# Lambda layers pull out the 5 individual characters to test on (faster than preprocessing data)
x1 = keras.layers.Lambda(lambda x: x[:, 0, :, :], output_shape=(64, 64), name='x1')(X)
x2 = keras.layers.Lambda(lambda x: x[:, 1, :, :], output_shape=(64, 64), name='x2')(X)
x3 = keras.layers.Lambda(lambda x: x[:, 2, :, :], output_shape=(64, 64), name='x3')(X)
x4 = keras.layers.Lambda(lambda x: x[:, 3, :, :], output_shape=(64, 64), name='x4')(X)
x5 = keras.layers.Lambda(lambda x: x[:, 4, :, :], output_shape=(64, 64), name='x5')(X)

# Flatten the images into 64 * 64 dimensional vectors
x1 = keras.layers.Flatten()(x1)
x2 = keras.layers.Flatten()(x2)
x3 = keras.layers.Flatten()(x3)
x4 = keras.layers.Flatten()(x4)
x5 = keras.layers.Flatten()(x5)

# The towers consist of a fully connected layer
x1 = keras.layers.Dense(16, activation='relu')(x1)
x2 = keras.layers.Dense(16, activation='relu')(x2)
x3 = keras.layers.Dense(16, activation='relu')(x3)
x4 = keras.layers.Dense(16, activation='relu')(x4)
x5 = keras.layers.Dense(16, activation='relu')(x5)

# Concatenates the towers together and feed through 3 fully connected layers
added = keras.layers.Add()([x1, x2, x3, x4, x5])
fc1 = keras.layers.Dense(20, activation='relu')(added)
fc2 = keras.layers.Dense(20, activation='relu')(fc1)
fc3 = keras.layers.Dense(20, activation='relu')(fc2)

# We use a sigmoidal activation to get a probability b/t 0 and 1
out = keras.layers.Dense(1, activation='sigmoid')(fc3)

model = keras.models.Model(inputs=X, outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_accuracy'])
model.summary()

# Plotting
# keras.utils.plot_model(model, to_file='model.png')

FILENAME = 'fonts-25'

# Open and prepare training set
train = h5py.File('discrim-task-{}-train.hdf5'.format(FILENAME), 'r')
labels = train['labels'][:].reshape(-1, 1) # Reshape to be dims (num_examples, 1)
examples = train['examples'][:]

print('Training model on {} examples...'.format(train['labels'].shape[0]))
history = model.fit(x=examples, y=labels, epochs=25) # See Keras docs for the history object

# Open and prepare test set
test = h5py.File('discrim-task-{}-test.hdf5'.format(FILENAME), 'r')
labels = test['labels'][:].reshape(-1, 1)
examples = test['examples'][:]

print('Testing model on {} examples...'.format(test['labels'].shape[0]))
loss = model.evaluate(x=examples, y=labels)

# View classified examples from the test set
predictions = model.predict(examples)
predictions = predictions.reshape(-1)
labels = labels.reshape(-1)
pred_labels = np.round(predictions).astype(int)

# Get indexes of false positives and false negatives
true_positives = np.where(np.logical_and(labels == 1, pred_labels == 1))[0]
true_negatives = np.where(np.logical_and(labels == 0, pred_labels == 0))[0]
false_positives = np.where(np.logical_and(labels == 1, pred_labels == 0))[0]
false_negatives = np.where(np.logical_and(labels == 0, pred_labels == 1))[0]

def display_picture(arr, idx, typ, pred):
    img = PIL.Image.fromarray(np.hstack((arr[0], arr[1], arr[2], arr[3], arr[4])))
    img.show()
    img.save('ex{}-{}-{}.png'.format(idx, typ, str(pred)[2:5]))

input('Go to true positives and true negatives...')

idx = true_positives[0]
display_picture(examples[idx], idx, 'tp', predictions[idx])
print('Example {} (label: {}, predicted: {} ({}))'.format(idx, labels[idx], pred_labels[idx], predictions[idx]))

idx = true_negatives[0]
display_picture(examples[idx], idx, 'tn', predictions[idx])
print('Example {} (label: {}, predicted: {} ({}))'.format(idx, labels[idx], pred_labels[idx], predictions[idx]))

input('Go to false positives and false negatives...')

print('false_positives: {}, false_negatives: {}'.format(false_positives.shape[0], false_negatives.shape[0]))

for i in range(5):
    idx = false_positives[i]
    display_picture(examples[idx], idx, 'fp', predictions[idx])
    print('Example {} (label: {}, predicted: {} ({}))'.format(idx, labels[idx], pred_labels[idx], predictions[idx]))

for i in range(5):
    idx = false_negatives[i]
    display_picture(examples[idx], idx, 'fn', predictions[idx])
    print('Example {} (label: {}, predicted: {} ({}))'.format(idx, labels[idx], pred_labels[idx], predictions[idx]))

input('Press anything to exit...')
