"""
test_file.py
---
Tests to sanity check a generated hdf5 file.
"""

import h5py

f = h5py.File('fonts.hdf5', 'r')

# Keys (see datasets)
print(list(f.keys()))

# Extract a dataset
dset = f['fonts']

# Print out info
print('dset.shape', dset.shape)
print('dset.dtype', dset.dtype)