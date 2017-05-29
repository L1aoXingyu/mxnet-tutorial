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


def to4array(x):
    x = x / 255
    x = x.reshape(x.shape[0], -1)
    return x


train_iter = mx.io.NDArrayIter(data=to4array(train_img), label=train_lbl,
                               batch_size=128, shuffle=True, label_name='log')

val_iter = mx.io.NDArrayIter(data=to4array(val_img),
                             label=val_lbl, batch_size=128,
                             shuffle=False, label_name='log')

# define network
checkpoint = mx.callback.do_checkpoint('./save_model/logistic_model')
data = mx.sym.Variable(name='data')
label = mx.sym.Variable(name='log')
fc = mx.sym.FullyConnected(data=data, name='logstic', num_hidden=10)
logistic = mx.sym.SoftmaxOutput(data=fc, label=label)

# define model
model = mx.mod.Module(
    context=mx.gpu(0),
    symbol=logistic,
    data_names=['data'],
    label_names=['log'],
)

model.fit(train_data=train_iter, eval_data=val_iter, num_epoch=50,
          eval_metric=['acc', 'ce'],
          optimizer='sgd',
          optimizer_params=(('learning_rate', 1e-3),),
          epoch_end_callback=checkpoint)
