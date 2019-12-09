import h5py
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import pickle
import argparse
import util

np.random.seed(1)
tf.random.set_seed(1)

import PIL, PIL.Image

WORK_PATH = './infer-task/'
EPOCHS = 60

"""
exp3
run command: python nn_infer.py -i fonts-jpn-all -e 1 -t -dir inf-exp-3 -w ./gen-task/fonts-all-2908_exp2_d2019-12-01_2h-3m_weights

add -p to predict only
python nn_infer.py -i fonts-system -e 0 -p -dir inf-sysfonts-task -w ./infer-task/fonts-jpn-all_exp1_d2019-12-01_20h-25m_weights

add -s to save weights
"""

def generate_task(infile, experiment, run_test_set, save_weights, weightfile, tmp_dir, predict_only):
    """
    Helper functions
    """
    namespace = util.namespace(infile, experiment)

    def display_picture(arr, name):
        """
        Displays 46 hiragana.
        """
        img = PIL.Image.fromarray(np.hstack([arr[idx] for idx in range(46)]))
        if img.mode != 'L':
            img = img.convert('L')
        # img.show() # Debug (disable when running)
        img.save('./{}/{}{}.png'.format(tmp_dir, namespace, name))

    def display_basis(arr, name):
        """
        Displays basis letters.
        """
        img = PIL.Image.fromarray(np.hstack([arr[idx] for idx in range(4)]))
        if img.mode != 'L':
            img = img.convert('L')
        # img.show() # Debug (disable when running)
        img.save('./{}/{}{}.png'.format(tmp_dir, namespace, name))

    def dump_history(history):
        with open('{}/{}history.pickle'.format(WORK_PATH, namespace), 'wb') as f:
            pickle.dump(history, f)
            print('Dumped history.')

    """
    Model definition: tower network on basis letters to get autoencoding, feed 
    through shared dense layers to retrieve all 26 letters (caps)
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
    neurons = 32
    x1 = keras.layers.Dense(neurons, activation='relu')(x1)
    x2 = keras.layers.Dense(neurons, activation='relu')(x2)
    x3 = keras.layers.Dense(neurons, activation='relu')(x3)
    x4 = keras.layers.Dense(neurons, activation='relu')(x4)

    # Concatenates the towers together and feed through fully connected layers
    added = keras.layers.Concatenate()([x1, x2, x3, x4])
    neurons = 400
    num_chars = 46
    fc = keras.layers.Dense(neurons, activation='relu')(added)
    fc = keras.layers.Dense(neurons, activation='relu')(fc)

    if (experiment == 1):
        fc = keras.layers.Dense(26 * 64 * 64, activation='relu')(fc) # Use same layer as in original gen task

    fc = keras.layers.Dense(num_chars * 64 * 64, activation='relu', name='hiragana_dense')(fc)

    # Reshape for 2D convolution and upsample
    fc = keras.layers.Reshape((num_chars, 64, 64))(fc)
    fc = keras.layers.Conv2DTranspose(num_chars, data_format='channels_first', kernel_size=(4, 4), padding='same', activation='relu', name='hiragana_transpose')(fc)

    out = fc

    model = keras.models.Model(inputs=X, outputs=out)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[])
    model.summary()

    # Plotting
    # keras.utils.plot_model(model, to_file='model.png')
    # exit()

    if weightfile is not None:
        print('Loading weightfile...')
        model.load_weights('{}.hdf5'.format(weightfile), by_name=True)

    if predict_only:
        """
        Does just prediciton on the data.
        """ 
        train = h5py.File('./infer-task-dsets/infer-task-{}.hdf5'.format(infile), 'r')
        basis = train['basis'][:]

        predictions = model.predict(basis)
        for i, (p, b) in enumerate(zip(predictions, basis)):
            display_picture(p, 'predict{}'.format(i))
            display_basis(b, 'basis{}'.format(i))

        exit()

    # Open and prepare training set
    train = h5py.File('./infer-task-dsets/infer-task-{}-train.hdf5'.format(infile), 'r')
    outputs = train['outputs'][:]
    basis = train['basis'][:]

    print('Training model on {} fonts...'.format(train['basis'].shape[0]))

    class ImageHistory(keras.callbacks.Callback):
        """
        Runs predict on the model and test set to visualize how the 
        NN is learning. In a Keras callback, we have access to model
        and params as class properties.
        """
        def __init__(self):
            super() # Parent class constructor
            self.image_idx = 0

        def on_train_begin(self, logs={}):
            predictions = self.model.predict(basis[:1])
            display_picture(predictions[0], 'train-viz-{}'.format(self.image_idx))
            self.image_idx += 1

        def on_batch_end(self, batch, logs={}):
            predictions = self.model.predict(basis[:1])
            display_picture(predictions[0], 'train-viz-{}'.format(self.image_idx))
            self.image_idx += 1

    history = model.fit(x=basis, y=outputs, epochs=EPOCHS, batch_size=512, callbacks=[ImageHistory()]) # See Keras docs for the history object
    dump_history(history.history)
    if save_weights:
        model.save_weights('./infer-task/{}weights.hdf5'.format(namespace))


    # Open and prepare val set
    test = h5py.File('./infer-task-dsets/infer-task-{}-val.hdf5'.format(infile), 'r')
    outputs = test['outputs'][:]
    basis = test['basis'][:]

    print('Validating model on {} fonts...'.format(test['basis'].shape[0]))
    loss = model.evaluate(x=basis, y=outputs)

    # View classified examples from the validation set
    predictions = model.predict(basis)
    display_picture(predictions[0], 'val')
    display_basis(basis[0], 'basis-val')

    if run_test_set:
        # Open and prepare test set
        test = h5py.File('./infer-task-dsets/infer-task-{}-test.hdf5'.format(infile), 'r')
        outputs = test['outputs'][:]
        basis = test['basis'][:]

        print('Testing model on {} fonts...'.format(test['basis'].shape[0]))
        loss = model.evaluate(x=basis, y=outputs)

        # View classified examples from the validation set
        predictions = model.predict(basis)
        display_picture(predictions[0], 'test')
        display_basis(basis[0], 'basis-test')


def parse_args():
    DEFAULT_FILENAME = 'fonts-50'
    parser = argparse.ArgumentParser(description='Run generative task.')
    parser.add_argument('--infile', '-i', default=DEFAULT_FILENAME, help='Name of data infile.')
    parser.add_argument('--experiment', '-e', default=0, type=int, help='Experiment number.')
    parser.add_argument('--test', '-t', action='store_true', help='Run test set (default: False).')
    parser.add_argument('--save_weights', '-s', action='store_true', help='Save weights (default: False).')
    parser.add_argument('--load_weights', '-w', help='Load weights from hdf5 file (default: None).')
    parser.add_argument('--tmp_dir', '-dir', default='tmp', help='Temp dir to dump info (default: tmp).')
    parser.add_argument('--predict', '-p', action='store_true', help='Predict only.')

    return parser.parse_args()


def main():
    args = parse_args()
    generate_task(args.infile, args.experiment, args.test, args.save_weights, args.load_weights, args.tmp_dir, args.predict)
    

if __name__ == '__main__':
    main()
