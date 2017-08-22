import mxnet as mx
import mxnet.gluon as g
import numpy as np
import time

# define hyperparameters
batch_size = 64
learning_rate = 1e-3
epochs = 100
ctx = mx.gpu()


# define data transform
def data_transform(data, label):
    return data.astype(np.float32) / 255, label.astype(np.float32)


# define dataset and dataloader
train_dataset = g.data.vision.MNIST(transform=data_transform)
test_dataset = g.data.vision.MNIST(train=False, transform=data_transform)

train_loader = g.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_loader = g.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)

# define model
logistic_model = g.nn.Sequential()
with logistic_model.name_scope():
    logistic_model.add(g.nn.Dense(10))

logistic_model.collect_params().initialize(
    mx.init.Xavier(rnd_type='gaussian'), ctx=ctx)

criterion = g.loss.SoftmaxCrossEntropyLoss()
optimizer = g.Trainer(logistic_model.collect_params(), 'sgd',
                      {'learning_rate': learning_rate})

# start train
for e in range(epochs):
    print('epoch {}'.format(e + 1))
    print('*' * 10)
    since = time.time()
    running_loss = 0.0
    running_acc = 0.0
    for i, (img, label) in enumerate(train_loader, 1):
        img = img.as_in_context(ctx).reshape((-1, 28 * 28))
        label = label.as_in_context(ctx)
        with g.autograd.record():
            output = logistic_model(img)
            loss = criterion(output, label)
        loss.backward()
        optimizer.step(batch_size)
        # =========== keep average loss and accuracy ==============
        running_loss += mx.nd.mean(loss).asscalar()
        predict = mx.nd.argmax(output, axis=1)
        num_correct = mx.nd.mean(predict == label).asscalar()
        running_acc += num_correct
        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                e + 1, epochs, running_loss / i, running_acc / i))