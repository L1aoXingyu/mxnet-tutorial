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
    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    return x

train_iter = mx.io.NDArrayIter(data=to4d(train_img), label=train_lbl,
                               batch_size=128, shuffle=True,
                               data_name='img',
                               label_name='softmax_label')

val_iter = mx.io.NDArrayIter(data=to4d(val_img), label=val_lbl,
                             batch_size=128, shuffle=False,
                             data_name='img',
                             label_name='softmax_label')

# define structure
data = mx.sym.Variable(name='img')
label = mx.sym.Variable(name='softmax_label')
conv1 = mx.sym.Convolution(data=data, num_filter=6, kernel=(3, 3),
                           stride=(1, 1), pad=(1, 1), name='conv1')
ac1 = mx.sym.Activation(data=conv1, act_type='relu', name='ac1')
pool1 = mx.sym.Pooling(data=ac1, pool_type='max', kernel=(2, 2),
                       stride=(2, 2), name='pool1')
conv2 = mx.sym.Convolution(data=pool1, num_filter=16, kernel=(5, 5),
                           stride=(1, 1), name='conv2')
ac2 = mx.sym.Activation(data=conv2, act_type='relu', name='ac2')
pool2 = mx.sym.Pooling(data=ac2, pool_type='max', kernel=(2, 2),
                       stride=(2, 2), name='pool2')
flatten = mx.sym.Flatten(data=pool2, name='flatten')
fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=120, name='fc1')
fc2 = mx.sym.FullyConnected(data=fc1, num_hidden=84, name='fc2')
fc3 = mx.sym.FullyConnected(data=fc2, num_hidden=10, name='fc3')
softmax = mx.sym.SoftmaxOutput(data=fc3, label=label)
mx.viz.plot_network(symbol=softmax)
# define model
save = mx.callback.do_checkpoint('./model_save/cnn')
cnn = mx.mod.Module(
        context=mx.cpu(),
        symbol=softmax,
        data_names=['img'],
        label_names=['softmax_label']
)

cnn.fit(train_data=train_iter, eval_data=val_iter,
        num_epoch=100, eval_metric=['ce', 'acc'],
        optimizer='sgd',
        optimizer_params={'learning_rate': 1e-2},
        # epoch_end_callback=save
        )
