import h5py
import PIL, PIL.Image
import numpy as np
import random

# TODO: Change this file to the desired input file name
FILENAME = 'fonts-all-2908'

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
all_caps = [i for i in range(26, 26*2)]

files = {
    'train': h5py.File('./gen-task-dsets/gen-task-{}-train.hdf5'.format(FILENAME), 'w'),
    'val': h5py.File('./gen-task-dsets/gen-task-{}-val.hdf5'.format(FILENAME), 'w'),
    'test': h5py.File('./gen-task-dsets/gen-task-{}-test.hdf5'.format(FILENAME), 'w'),
}
all_img_dset, all_output_dset, all_di = {}, {}, {}
all_img_dset['train'] = files['train'].create_dataset('basis', (1, 4, 64, 64), chunks=(1, 4, 64, 64), maxshape=(None, 4, 64, 64), dtype='u1')
all_img_dset['val'] = files['val'].create_dataset('basis', (1, 4, 64, 64), chunks=(1, 4, 64, 64), maxshape=(None, 4, 64, 64), dtype='u1')
all_img_dset['test'] = files['test'].create_dataset('basis', (1, 4, 64, 64), chunks=(1, 4, 64, 64), maxshape=(None, 4, 64, 64), dtype='u1')
all_output_dset['train'] = files['train'].create_dataset('outputs', (1, 26, 64, 64), chunks=(1, 26, 64, 64), maxshape=(None, 26, 64, 64), dtype='u1')
all_output_dset['val'] = files['val'].create_dataset('outputs', (1, 26, 64, 64), chunks=(1, 26, 64, 64), maxshape=(None, 26, 64, 64), dtype='u1')
all_output_dset['test'] = files['test'].create_dataset('outputs', (1, 26, 64, 64), chunks=(1, 26, 64, 64), maxshape=(None, 26, 64, 64), dtype='u1')
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

"""
Generate all examples.
"""
for font_idx in range(num_fonts):
    a = dset[font_idx, A_idx]
    h = dset[font_idx, H_idx]
    q = dset[font_idx, Q_idx]
    j = dset[font_idx, J_idx]

    # Generate example for font
    print('Generating font example...')
    
    # Create a basis letters with shape (4, 64, 64)
    basis = np.array([a, h, q, j])

    # Create output
    output = np.array([dset[font_idx, idx] for idx in all_caps])
    
    # Resize datasets and store
    group = get_split()
    img_dset, output_dset, di = all_img_dset[group], all_output_dset[group], all_di[group]

    img_dset.resize((di+1, *basis.shape))
    output_dset.resize((di+1, *output.shape))
    img_dset[di] = basis
    output_dset[di] = output
    all_di[group] += 1
    files[group].flush()

    # Debug: input example
    # img = PIL.Image.fromarray(np.hstack((a, h, q, j)))
    # img.show()
    # exit()
    # Debug: output example
    # img = PIL.Image.fromarray(np.hstack((output[idx] for idx in range(26))))
    # img.show()
    # exit()
    print('Finished font {}'.format(font_idx))

print('Number of train: {}, Number of val: {}, Number of test: {}'.format(all_di['train'], all_di['val'], all_di['test']))

for f in files.values():
    f.close()