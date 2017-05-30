__author__ 'SherlockLiao'

import mxnet as mx
import numpy as np
import gzip
import struct
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def read_data(label_file, image_file):
    with gzip.open(label_file) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(image_file, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label),
                                                                   rows, cols)
    return (label, image)


(train_lbl, train_img) = read_data('./data/train-labels-idx1-ubyte.gz',
                                   './data/train-images-idx3-ubyte.gz')
(val_lbl, val_img) = read_data('./data/t10k-labels-idx1-ubyte.gz',
                               './data/t10k-images-idx3-ubyte.gz')


train_iter = mx.io.NDArrayIter(data=train_img, label=train_lbl,
                               batch_size=128, shuffle=True,
                               data_name='img',
                               label_name='softmax_label')

val_iter = mx.io.NDArrayIter(data=val_img, label=val_lbl,
                             batch_size=128, shuffle=False,
                             data_name='img',
                             label_name='softmax_label')

# define structure
data = mx.sym.Variable(name='img')
label = mx.sym.Variabel(name='softmax_label')
