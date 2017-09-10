__author__ = 'SherlockLiao'

import os

import mxnet as mx
import numpy as np
import torch
from mxnet import gluon
from torchvision.utils import save_image

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3
ctx = mx.gpu()


def transform(data, label):
    return (data.astype('float32') / 255 - 0.5) / 0.5, label.astype('float32')


mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)

dataloader = gluon.data.DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True)


class autoencoder(gluon.Block):
    def __init__(self):
        super(autoencoder, self).__init__()
        with self.name_scope():
            self.encoder = gluon.nn.Sequential('encoder_')
            with self.encoder.name_scope():
                self.encoder.add(gluon.nn.Dense(128, activation='relu'))
                self.encoder.add(gluon.nn.Dense(64, activation='relu'))
                self.encoder.add(gluon.nn.Dense(12, activation='relu'))
                self.encoder.add(gluon.nn.Dense(3))

            self.decoder = gluon.nn.Sequential('decoder_')
            with self.decoder.name_scope():
                self.decoder.add(gluon.nn.Dense(12, activation='relu'))
                self.decoder.add(gluon.nn.Dense(64, activation='relu'))
                self.decoder.add(gluon.nn.Dense(128, activation='relu'))
                self.decoder.add(gluon.nn.Dense(28 * 28, activation='tanh'))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder()
model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
criterion = gluon.loss.L2Loss()
optimizer = gluon.Trainer(model.collect_params(), 'adam',
                          {'learning_rate': learning_rate,
                           'wd': 1e-5})

for epoch in range(num_epochs):
    running_loss = 0.0
    n_total = 0.0
    for data in dataloader:
        img, _ = data
        img = img.reshape((img.shape[0], -1)).as_in_context(ctx)

        with gluon.autograd.record():
            output = model(img)
            loss = criterion(output, img)
        loss.backward()
        optimizer.step(img.shape[0])
        running_loss += mx.nd.sum(loss).asscalar()
        n_total += img.shape[0]
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, running_loss / n_total))
    if epoch % 10 == 0:
        pic = to_img(torch.FloatTensor(output.asnumpy()))
        save_image(pic, './mlp_img/{}_autoencoder.png'.format(epoch))
model.save_params('./simple_autoencoder.params')
