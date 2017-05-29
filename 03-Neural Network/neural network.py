__author__ = 'SherlockLiao'

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


def to4d(x):
    x = x / 255
    x = x.reshape(x.shape[0], -1)
    return x

train_iter = mx.io.NDArrayIter(data=to4d(train_img), label=train_lbl,
                           batch_size=128, shuffle=True)
val_iter = mx.io.NDArrayIter(data=to4d(val_img), label=val_lbl,
                         batch_size=128, shuffle=False)


# define structure
img = mx.sym.Variable(name='data')
label = mx.sym.Variable(name='softmax_label')
fc1 = mx.sym.FullyConnected(data=img, name='fc1', num_hidden=300)
fc2 = mx.sym.FullyConnected(data=fc1, name='fc2', num_hidden=100)
fc3 = mx.sym.FullyConnected(data=fc2, name='fc3', num_hidden=10)
softmax_out = mx.sym.SoftmaxOutput(data=fc3, label=label)

# define model
neural_network = mx.mod.Module(
            context=mx.cpu(0),
            symbol=softmax_out,
            data_names=['data'],
            label_names=['softmax_label']
)

save = mx.callback.do_checkpoint('./model_save/neural_network')
speed = mx.callback.Speedometer(128, 200)
# train model
neural_network.fit(train_data=train_iter, eval_data=val_iter,
                   num_epoch=100, eval_metric=['acc', 'ce'],
                   optimizer='sgd',
                   optimizer_params={'learning_rate': 1e-2},
                   epoch_end_callback=save
)
