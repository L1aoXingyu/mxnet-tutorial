__author__ = 'SherlockLiao'

import time

import mxnet as mx
import mxnet.gluon as g
import numpy as np

# define hyperparameters
batch_size = 100
learning_rate = 1e-3
epochs = 20
step = 300
ctx = mx.gpu()


# define data transform
def data_transform(data, label):
    return mx.nd.transpose(data.astype(np.float32) / 255,
                           (2, 0, 1)), label.astype(np.float32)


# define dataset and dataloader
train_dataset = g.data.vision.MNIST(transform=data_transform)
test_dataset = g.data.vision.MNIST(train=False, transform=data_transform)

train_loader = g.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_loader = g.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)


# define model
class rnn(g.Block):
    def __init__(self, hidden, n_layer, n_class):
        super(rnn, self).__init__()
        with self.name_scope():
            self.lstm = g.rnn.LSTM(hidden, n_layer)
            self.classifier = g.nn.Dense(n_class)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = out[out.shape[0] - 1, :, :]
        out = self.classifier(out)
        return out


model = rnn(128, 2, 10)

model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

criterion = g.loss.SoftmaxCrossEntropyLoss()
optimizer = g.Trainer(model.collect_params(), 'adam',
                      {'learning_rate': learning_rate})

# start train
for e in range(epochs):
    print('*' * 10)
    print('epoch {}'.format(e + 1))
    since = time.time()
    moving_loss = 0.0
    moving_acc = 0.0
    for i, (img, label) in enumerate(train_loader, 1):
        b, c, h, w = img.shape
        img = img.reshape((b, h, w))
        img = mx.nd.transpose(img, (2, 0, 1))
        img = img.as_in_context(ctx)
        label = label.as_in_context(ctx)
        h0 = mx.nd.zeros(shape=(2, b, 128), ctx=ctx)
        c0 = mx.nd.zeros(shape=(2, b, 128), ctx=ctx)
        with g.autograd.record():
            output = model(img, [h0, c0])
            loss = criterion(output, label)
        loss.backward()
        optimizer.step(b)
        # =========== keep average loss and accuracy ==============
        moving_loss += mx.nd.mean(loss).asscalar()
        predict = mx.nd.argmax(output, axis=1)
        acc = mx.nd.mean(predict == label).asscalar()
        moving_acc += acc

        if i % step == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                i, len(train_loader), moving_loss / step, moving_acc / step))
            moving_loss = 0.0
            moving_acc = 0.0
    test_loss = 0.0
    test_acc = 0.0
    total = 0.0
    for img, label in test_loader:
        b, c, h, w = img.shape
        img = img.reshape((b, h, w))
        img = mx.nd.transpose(img, (2, 0, 1))
        img = img.as_in_context(ctx)
        label = label.as_in_context(ctx)
        h0 = mx.nd.zeros(shape=(2, b, 128), ctx=ctx)
        c0 = mx.nd.zeros(shape=(2, b, 128), ctx=ctx)
        output = model(img, [h0, c0])
        loss = criterion(output, label)
        test_loss += mx.nd.sum(loss).asscalar()
        predict = mx.nd.argmax(output, axis=1)
        test_acc += mx.nd.sum(predict == label).asscalar()
        total += b
    print('Test Loss: {:.6f}, Test Acc: {:.6f}'.format(test_loss / total,
                                                       test_acc / total))
    print('Time: {:.1f} s'.format(time.time() - since))

model.save_params('./lstm.params')