import logging
import sys

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(message)s',
                    stream=sys.stdout)

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
batch_size = 4
total_epochs = 200

x_train = np.array(
    [[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], [9.779], [6.182], [7.59],
     [2.167], [7.042], [10.791], [5.313], [7.997], [3.1]],
    dtype=np.float32)

y_train = np.array(
    [[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], [3.366], [2.596], [2.53],
     [1.221], [2.827], [3.465], [1.65], [2.904], [1.3]],
    dtype=np.float32)

train_iter = mx.io.NDArrayIter(x_train, y_train, batch_size=batch_size, shuffle=True, label_name='reg_label')

# define network structure
data = mx.sym.var('data')
label = mx.sym.var('reg_label')
reg = mx.sym.FullyConnected(data, num_hidden=1, name='fc')
reg = mx.sym.LinearRegressionOutput(data=reg, label=label)

mod = mx.mod.Module(reg, data_names=['data'], label_names=['reg_label'], context=ctx)
mod.fit(train_iter,
        optimizer='sgd',
        optimizer_params={'learning_rate': 0.01, 'momentum': 0.9},
        num_epoch=total_epochs,
        eval_metric='mse',
        batch_end_callback=mx.callback.Speedometer(batch_size, frequent=1),
        epoch_end_callback=mx.callback.do_checkpoint('linear', period=10))

# predict result
mod.forward(mx.io.DataBatch(data=[mx.nd.array(x_train)], label=[mx.nd.array(y_train)]), is_train=False)
predict = mod.get_outputs()[0].asnumpy()

plt.plot(x_train, y_train, 'ro', label='actual')
plt.plot(x_train, predict, label='predicted line')
plt.show()
