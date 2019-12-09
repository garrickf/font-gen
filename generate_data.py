"""
generate_data.py
---
Generates image data from font files. Adapted from @erikbern, link:
https://github.com/erikbern/deep-fonts/blob/master/create_dataset.py
"""

import h5py
import PIL, PIL.ImageFont, PIL.Image, PIL.ImageDraw, PIL.ImageChops, PIL.ImageOps
import os
import random
import string
import numpy as np
import sys

# Dims of the images we want
w, h = 64, 64
w0, h0 = 256, 256

# Mode 'L' is 8-bit pixels, black and white
blank = PIL.Image.new('L', (w0*3, h0*3), 255)

# The characters we want to generate (added Japanese hiragana)
# jpn_hiragana = 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん'
all_chars = string.ascii_letters + string.digits #+ jpn_hiragana

def generate_examples_from(fontname):
    imgs, data = [], []

    # Track these to scale down fonts later while preserving vertical alignment
    max_width = 0
    min_top = float('inf')
    max_bottom = float('-inf')

    # Load a TrueType or OpenType font file, specify size (in points)
    font = PIL.ImageFont.truetype(fontname, min(w0, h0))

    for char in all_chars:
        img = PIL.Image.new('L', (w0*3, h0*3), 255)

        draw = PIL.ImageDraw.Draw(img)
        draw.text((w0, h0), char, font=font)

        # Get the difference between the image and a blank image
        diff = PIL.ImageChops.difference(img, blank)
        lx, ly, hx, hy = diff.getbbox()
        left, top, right, bottom = lx, ly, hx, hy

        # Track bounding box edges to calculate scaling factor
        min_top = min(min_top, top)
        max_bottom = max(max_bottom, bottom)
        max_width = max(max_width, right - left)
        imgs.append((left, right, img))

    # Compute the scale factor to scale the largest dimension to the bounding box size
    # we want
    scale_factor = min(1.0 * h / (max_bottom - min_top), 1.0 * w / max_width)

    for left, right, img in imgs:
        # A character is cropped by it own left and right bounds and the largest 
        # vertical bounds we've seen (to handle tall characters)
        img = img.crop((left, min_top, right, max_bottom))

        # Compute new dimensions and scale
        new_width = (right - left) * scale_factor
        new_height = (max_bottom - min_top) * scale_factor
        img = img.resize((int(new_width), int(new_height)), PIL.Image.ANTIALIAS)

        # Expand to square, centering character as needed
        img_sq = PIL.Image.new('L', (w, h), 255)
        offset_x = (w - new_width) / 2
        offset_y = (h - new_height) / 2
        img_sq.paste(img, (int(offset_x), int(offset_y)))

        # Convert to numpy array
        example = np.array(img_sq.getdata()).reshape((h, w))
        example = 255 - example
        data.append(example)

        # Debug
        # img_sq.show()
    return np.array(data)


def all_fonts(d='./'):
    for dirpath, dirname, filenames in os.walk(d):
        for filename in filenames:
            if filename.endswith('.ttf') or filename.endswith('.otf') or filename.endswith('.ttc'):
                yield os.path.join(dirpath, filename)

# Usage: python generate_data.py ./fonts-system ./fonts-system
f = h5py.File('{}.hdf5'.format(sys.argv[2]), 'w')
dset = f.create_dataset('fonts', (1, len(all_chars), h, w), chunks=(1, len(all_chars), h, w), maxshape=(None, len(all_chars), h, w), dtype='u1')

i = 0
for fontname in all_fonts(d=sys.argv[1]):
    try:
        data = generate_examples_from(fontname)
    except: # IOError:
        print('Error reading {}'.format(fontname))
        continue

    print('<{}> Extracted shape: {}'.format(fontname, data.shape))
    
    dset.resize((i+1, len(all_chars), h, w))
    dset[i] = data
    i += 1
    f.flush()
    # Early stop
    # if i == 50: break

print('extracted: ', i)
f.close()
