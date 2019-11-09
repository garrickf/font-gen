import h5py
import PIL, PIL.Image
import numpy as np

input_f = h5py.File('fonts-25.hdf5', 'r')
dset = input_f['fonts']

A_idx = 26
H_idx = 33
Q_idx = 42
J_idx = 35
other_letter_idxs = [i for i in range(26, 26*2) if i not in [A_idx, H_idx, Q_idx, J_idx]]

y_same = 1
y_not = 0

# Create new file to place examples and labels into
f = h5py.File('fonts-25-discrim-task.hdf5', 'w')
img_dset = f.create_dataset('examples', (1, 5, 64, 64), chunks=(1, 5, 64, 64), maxshape=(None, 5, 64, 64), dtype='u1')
labels_dset = f.create_dataset('labels', (1,), chunks=(1,), maxshape=(None,), dtype='int32')
di = 0

for font_idx in range(10):
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
        img_dset.resize((di+1, *images.shape))
        labels_dset.resize((di+1,))
        img_dset[di] = images
        labels_dset[di] = y_same
        di += 1
        f.flush()

    # Generate not examples
    print('Generating not examples...')
    for not_idx in range(10):
        if not_idx == font_idx: continue
        for idx in other_letter_idxs:
            letter = dset[not_idx, idx]
            images = np.array([a, h, q, j, letter])

            # Resize datasets and store
            img_dset.resize((di+1, *images.shape))
            labels_dset.resize((di+1,))
            img_dset[di] = images
            labels_dset[di] = y_not
            di += 1
            f.flush()

        # Debug
        # img = PIL.Image.fromarray(np.hstack((a, h, q, j, letter)))
        # img.show()
    print('Finished font {}', font_idx)

f.close()

# Note:
# img = PIL.Image.fromarray(dset[1, 1])
