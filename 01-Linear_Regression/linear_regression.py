import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

train_iter = mx.io.NDArrayIter(data=x_train, label=y_train, batch_size=1,
                               shuffle=True, label_name='linear_label')

# define structure
data = mx.sym.Variable(name='data')
label = mx.sym.Variable(name='linear_label')
linear = mx.sym.FullyConnected(data=data, name='fc', num_hidden=1)
linear_out = mx.sym.LinearRegressionOutput(data=linear, label=label)

# define model
checkpoint = mx.callback.do_checkpoint('./save_model/linear_model', period=5)
model = mx.mod.Module(
    context=mx.gpu(0),
    symbol=linear_out,
    data_names=['data'],
    label_names=['linear_label']
)

model.fit(train_iter, num_epoch=20, eval_metric='mse',
          epoch_end_callback=checkpoint)

new_data = mx.io.NDArrayIter(x_train)
pred = model.predict(new_data)

plt.plot(x_train, y_train, 'ro', label='actual')
plt.plot(x_train, pred.asnumpy(), label='predicted line')
plt.show()
