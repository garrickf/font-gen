import h5py
import PIL, PIL.Image
import numpy as np
import random

# TODO: Change this file to the desired input file name
FILENAME = 'fonts-50'

# Set random seed to make datasets reproducible. This script may get rerun a few times.
random.seed(1)
np.random.seed(1)

input_f = h5py.File('./datasets/{}.hdf5'.format(FILENAME), 'r')
dset = input_f['fonts']
num_fonts = dset.shape[0] # shape of (num_fonts, num_letters, 64, 64)

A_idx = 26
H_idx = 33
Q_idx = 42
J_idx = 35
other_letter_idxs = [i for i in range(26, 26*2) if i not in [A_idx, H_idx, Q_idx, J_idx]]

y_same = 1
y_not = 0

files = {
    'train': h5py.File('./disc-task-dsets/disc-task-{}-train.hdf5'.format(FILENAME), 'w'),
    'val': h5py.File('./disc-task-dsets/disc-task-{}-val.hdf5'.format(FILENAME), 'w'),
    'test': h5py.File('./disc-task-dsets/disc-task-{}-test.hdf5'.format(FILENAME), 'w'),
}
all_img_dset, all_labels_dset, all_di = {}, {}, {}
all_img_dset['train'] = files['train'].create_dataset('examples', (1, 5, 64, 64), chunks=(1, 5, 64, 64), maxshape=(None, 5, 64, 64), dtype='u1')
all_img_dset['val'] = files['val'].create_dataset('examples', (1, 5, 64, 64), chunks=(1, 5, 64, 64), maxshape=(None, 5, 64, 64), dtype='u1')
all_img_dset['test'] = files['test'].create_dataset('examples', (1, 5, 64, 64), chunks=(1, 5, 64, 64), maxshape=(None, 5, 64, 64), dtype='u1')
all_labels_dset['train'] = files['train'].create_dataset('labels', (1,), chunks=(1,), maxshape=(None,), dtype='int32')
all_labels_dset['val'] = files['val'].create_dataset('labels', (1,), chunks=(1,), maxshape=(None,), dtype='int32')
all_labels_dset['test'] = files['test'].create_dataset('labels', (1,), chunks=(1,), maxshape=(None,), dtype='int32')
all_di['train'], all_di['val'], all_di['test'] = 0, 0, 0

# Determine train/test/val split
train_split = 0.7
test_split = 0.15
val_split = 0.15
assert(train_split + test_split + val_split == 1)

def get_split():
    """
    Randomly assigns train, val, or test based on above-defined splits.
    """
    cases = {
        train_split: 'train',
        train_split + val_split: 'val',
        train_split + val_split + test_split: 'test',
    }
    chance = random.random()
    for threshold in cases:
        if chance < threshold:
            return cases[threshold]

for font_idx in range(num_fonts):
    a = dset[font_idx, A_idx]
    h = dset[font_idx, H_idx]
    q = dset[font_idx, Q_idx]
    j = dset[font_idx, J_idx]

    # Generate same examples
    print('Generating same examples...')
    for idx in other_letter_idxs:
        letter = dset[font_idx, idx]
        # Create a training example with shape (5, 64, 64) in tuple (images, label)
        images = np.array([a, h, q, j, letter])
        
        # Resize datasets and store
        group = get_split()
        img_dset, labels_dset, di = all_img_dset[group], all_labels_dset[group], all_di[group]

        img_dset.resize((di+1, *images.shape))
        labels_dset.resize((di+1,))
        img_dset[di] = images
        labels_dset[di] = y_same
        all_di[group] += 1
        files[group].flush()

    # Generate not examples. If we generate (26 - 4) correct examples per font,
    # we need the same number of negative examples
    print('Generating not examples...')
    not_font_idxs = [i for i in range(25) if i != font_idx]
    not_examples = np.array([(font_idx, char_idx) for font_idx in not_font_idxs for char_idx in other_letter_idxs])
    not_sample_idxs = np.random.choice(len(not_examples), size=(26 - 4), replace=False)
    not_samples = not_examples[not_sample_idxs]
    assert(not_samples.shape[0] == 26 - 4)

    for not_idx, char_idx in not_samples:
        letter = dset[not_idx, char_idx]
        images = np.array([a, h, q, j, letter])

        # Resize datasets and store
        group = get_split()
        img_dset, labels_dset, di = all_img_dset[group], all_labels_dset[group], all_di[group]

        img_dset.resize((di+1, *images.shape))
        labels_dset.resize((di+1,))
        img_dset[di] = images
        labels_dset[di] = y_not
        all_di[group] += 1
        files[group].flush()

        # Debug
        # img = PIL.Image.fromarray(np.hstack((a, h, q, j, letter)))
        # img.show()
        # exit()
    print('Finished font {}'.format(font_idx))

print('Number of train: {}, Number of val: {}, Number of test: {}'.format(all_di['train'], all_di['val'], all_di['test']))

for f in files.values():
    f.close()
