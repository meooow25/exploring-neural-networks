# -*- coding: utf-8 -*-
"""
Module for loading image and label data into numpy arrays.
Data is from the MNIST database (http://yann.lecun.com/exdb/mnist/).
File format specifications can be found on the webpage.
"""

import gzip
import numpy as np

def readint32(f):
    return int.from_bytes(f.read(4), byteorder='big')

def load_images(path):
    print('Loading images from', path)
    f = gzip.open(path)
    magic = readint32(f)
    assert magic == 2051
    count = readint32(f)
    rows = readint32(f)
    cols = readint32(f)
    # print(count, rows, cols)
    images = [np.frombuffer(f.read(rows*cols), dtype='u1') for i in range(count)]
    assert not f.read()
    print('Loaded successfully')
    return images

def load_labels(path):
    print('Loading labels from', path)
    f = gzip.open(path)
    magic = readint32(f)
    assert magic == 2049
    count = readint32(f)
    # print(count)
    labels = [int.from_bytes(f.read(1), byteorder='big') for i in range(count)]
    assert not f.read()
    print('Loaded successfully')
    return labels