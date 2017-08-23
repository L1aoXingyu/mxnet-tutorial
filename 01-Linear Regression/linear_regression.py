import mxnet as mx
from mxnet import gluon as g
import numpy as np
import matplotlib.pyplot as plt

ctx = mx.gpu()
batch_size = 4
x_train = np.array(
    [[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], [9.779], [6.182], [7.59],
     [2.167], [7.042], [10.791], [5.313], [7.997], [3.1]],
    dtype=np.float32)

y_train = np.array(
    [[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], [3.366], [2.596], [2.53],
     [1.221], [2.827], [3.465], [1.65], [2.904], [1.3]],
    dtype=np.float32)

x_train = mx.nd.array(x_train)
y_train = mx.nd.array(y_train)

# define network structure
linear_model = g.nn.Sequential()
with linear_model.name_scope():
    linear_model.add(g.nn.Dense(1))

linear_model.collect_params().initialize(mx.init.Normal(sigma=0.01), ctx=ctx)
criterion = g.loss.L2Loss()
optimizer = g.Trainer(linear_model.collect_params(), 'sgd',
                      {'learning_rate': 1e-4})

epochs = 1000
for e in range(epochs):
    x = x_train.as_in_context(ctx)
    y = y_train.as_in_context(ctx)
    with mx.autograd.record():
        output = linear_model(x)
        loss = criterion(output, y)
    loss.backward()
    optimizer.step(x.shape[0])
    running_loss = mx.nd.mean(loss).asscalar()
    print('{}'.format(running_loss))

with mx.autograd.record(train_mode=False):
    predict = linear_model(x_train.as_in_context(ctx))

plt.plot(x_train.asnumpy(), y_train.asnumpy(), 'ro', label='actual')
plt.plot(x_train.asnumpy(), predict.asnumpy(), label='predicted line')
plt.show()

linear_model.save_params('linear.params')